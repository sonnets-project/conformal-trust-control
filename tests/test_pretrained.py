"""Check published weights, inference accounting and data-only evaluation."""
from pathlib import Path
import importlib.util
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from checkpoint_catalog import PRETRAINED, verify_catalog
from verify_release import release_files

HAS_TORCH = importlib.util.find_spec('torch') is not None
if HAS_TORCH:
    import torch
    import evaluate_pretrained as runner
    import episode_evaluation as evaluation
    import residual_policy
    from ctc_runtime import select_contract_paths, replay_benchmark
    from test_runtime import quotes


class CheckpointCatalogTests(unittest.TestCase):
    def test_all_published_weights_and_no_private_arrays_are_in_release(self):
        catalog = verify_catalog()
        weights = {Path('pretrained') / row['path'] for row in catalog['checkpoints']}
        paths = {path.relative_to(ROOT) for path in release_files()}
        self.assertEqual({path for path in paths if path.suffix == '.pt'}, weights)
        self.assertEqual(len(weights), 120)
        self.assertFalse(any(path.suffix in {'.parquet', '.npz', '.npy', '.pth'} for path in paths))

    def test_corrupted_or_unlisted_checkpoints_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            copied = Path(temporary) / 'pretrained'
            shutil.copytree(PRETRAINED, copied)
            first = verify_catalog(copied)['checkpoints'][0]
            extra = copied / 'unlisted.pt'
            extra.write_bytes(b'extra')
            with self.assertRaisesRegex(ValueError, 'differ from the catalog'):
                verify_catalog(copied)
            extra.unlink()
            (copied / first['path']).write_bytes(b'truncated')
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                verify_catalog(copied)


@unittest.skipUnless(HAS_TORCH, 'Install requirements-training.txt for model evaluation tests')
class PretrainedEvaluationTests(unittest.TestCase):
    def test_all_120_models_load_for_inference(self):
        torch.set_num_threads(1)
        states = torch.tensor([[1., 1., -.5, .2, .5, .01, .1, -.01]])
        for entry in verify_catalog()['checkpoints']:
            with self.subTest(checkpoint=entry['path']):
                model = runner.load_policy(entry)
                self.assertEqual(sum(p.numel() for p in model.parameters()), 4801)
                self.assertFalse(model.training)
                self.assertFalse(any(p.requires_grad for p in model.parameters()))
                self.assertTrue(torch.isfinite(model(states)).all())
        entry = dict(verify_catalog()['checkpoints'][0], seed=99)
        with self.assertRaisesRegex(ValueError, 'metadata mismatch'):
            runner.load_policy(entry)

    def test_evaluation_preserves_weights_and_bs_fallback(self):
        frame, rates = quotes()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            rates.to_csv(directory / 'rates.csv', index=False)
            evaluation.set_rates(directory / 'rates.csv')
            replay_benchmark(select_contract_paths(frame), rates).to_csv(directory / 'episodes.csv', index=False)
            episodes = evaluation._episodes_from_csv(directory / 'episodes.csv')
            pack = evaluation.pack_episodes(episodes)
            model = runner.load_policy(verify_catalog()['checkpoints'][0])
            before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
            with patch.object(residual_policy, 'train_residual', side_effect=AssertionError('Training is forbidden')):
                ledger = runner.evaluate_policy(model, 1, episodes, episodes, pack, pack)
            np.testing.assert_array_equal(ledger['dp_lam'][0.], np.zeros(len(episodes)))
            np.testing.assert_array_equal(ledger['d_lam'][0.], np.zeros(len(episodes)))
            for lam in runner.controller.LAM_GRID:
                np.testing.assert_array_equal(ledger['dp_lam'][lam], ledger['vadp'][lam])
            # Independently account for the positions, option PnL and trading costs.
            positions = torch.clamp(pack['b_pos'] + model(pack['states']), -1.5, 1.5).numpy()
            baseline = pack['b_pos'].numpy()
            def pnl(position):
                previous = np.r_[np.float32(0), position[:-1]] * (1-pack['is_first'].numpy())
                steps = (position*(pack['S_new']-pack['S_old']).numpy()
                         + (pack['P_new']-pack['P_old']).numpy()
                         - abs(position-previous)*pack['S_old'].numpy()*.001)
                return np.array([steps[pack['ep_id'].numpy()==index].sum() for index in range(len(episodes))])
            np.testing.assert_allclose(ledger['dp_lam'][1.], (pnl(positions)-pnl(baseline))*1e4,
                                       rtol=1e-5, atol=1e-5)
            for name, tensor in model.state_dict().items():
                self.assertTrue(torch.equal(tensor, before[name]))

    def test_panel_rates_fill_missing_dates_without_replacing_existing_rates(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, output = Path(temporary)/'source.csv', Path(temporary)/'rates.csv'
            pd.DataFrame({'Date':['2017-10-06'], '1y':[.012]}).to_csv(source, index=False)
            panel = pd.DataFrame({'date':pd.to_datetime(['2017-10-06', '2017-10-09']), 'r':[.02, .013]})
            runner.prepare_rates(panel, source, output)
            result = pd.read_csv(output)
            self.assertEqual(result['1y'].tolist(), [.012, .013])
            self.assertEqual(len(pd.read_csv(source)), 1)


if __name__ == '__main__':
    unittest.main()
