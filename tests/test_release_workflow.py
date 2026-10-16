from pathlib import Path
import json
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from paper_results import compare_frames, verify_references
from run_experiments import commands, STAGES
from verify_release import release_files, notebook_issues


class ReleaseWorkflowTests(unittest.TestCase):
    def test_notebook_execution_metadata_is_rejected_even_without_outputs(self):
        clean = {'metadata': {'kernelspec': {'display_name': 'Python 3', 'name': 'python3'},
                              'language_info': {'name': 'python'}},
                 'cells': [{'cell_type': 'code', 'source': ['print(1)'], 'metadata': {},
                            'outputs': [], 'execution_count': None}]}
        self.assertEqual(notebook_issues(clean), [])
        clean['cells'][0]['metadata'] = {'executionInfo': {'user': {'displayName': 'Example'},
                                                          'timestamp': 1234567890}}
        self.assertTrue(any('metadata' in issue for issue in notebook_issues(clean)))
        clean['cells'][0]['metadata'] = {}
        clean['metadata']['widgets'] = {'state': 'saved session'}
        self.assertTrue(any('notebook metadata' in issue for issue in notebook_issues(clean)))

    def test_experiment_order_and_regime_commands(self):
        self.assertLess(STAGES.index('controls'), STAGES.index('three-arm'))
        self.assertLess(STAGES.index('common-budget'), STAGES.index('analysis'))
        self.assertEqual(len(commands('controls')), 4)
        self.assertIn('--skip-comparison', commands('analysis')[0][1])

    def test_runtime_directories_are_not_release_files(self):
        paths = [p.relative_to(ROOT) for p in release_files()]
        self.assertTrue(paths)
        self.assertFalse(any(p.parts[0] == '.runtime' for p in paths))
        self.assertFalse(any(p.parts[:2] == ('new-reward', 'checkpoints') for p in paths))

    def test_comparison_accepts_row_reordering_but_rejects_changed_numbers(self):
        expected = verify_references()["table4_aggressive.csv"]
        actual = expected.iloc[::-1].copy()
        self.assertEqual(compare_frames("table4_aggressive.csv", actual, expected)["status"], "PASS")
        actual.loc[actual.index[0], "breach"] += .01
        with self.assertRaisesRegex(ValueError, "Numerical mismatch"):
            compare_frames("table4_aggressive.csv", actual, expected)

    def test_existing_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as output:
            sentinel = Path(output) / "keep.txt"
            sentinel.write_text("preserve me")
            result = subprocess.run([sys.executable, str(ROOT / "scripts/reproduce_paper.py"),
                                     "--output", output], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Output already exists", result.stderr)
            self.assertEqual(sentinel.read_text(), "preserve me")


if __name__ == "__main__":
    unittest.main()
