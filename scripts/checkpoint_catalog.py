"""Verify the published checkpoint catalog without importing PyTorch."""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRETRAINED = ROOT / 'pretrained'
REGIMES = ('crash_test2018', 'crash_test2020', 'mix_test2022', 'mix_test2024')
ARMS = ('cons', 'aggr')
KAPPAS = (1, 2, 3)
SEEDS = (0, 1, 2, 3, 4)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_catalog(directory=PRETRAINED):
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest.get('format_version') != 1 or manifest.get('checkpoint_count') != 120:
        raise ValueError('Unsupported checkpoint catalog format or count.')
    expected = set(itertools.product(REGIMES, ARMS, KAPPAS, SEEDS))
    entries = manifest['checkpoints']
    identities = [(entry['regime'], entry['arm'], entry['kappa'], entry['seed']) for entry in entries]
    if len(entries) != 120 or len(set(identities)) != 120 or set(identities) != expected:
        raise ValueError('Checkpoint catalog must contain exactly four regimes, two arms, three kappas and five seeds.')
    if {path.relative_to(directory).as_posix() for path in directory.rglob('*.pt')} != {entry['path'] for entry in entries}:
        raise ValueError('Checkpoint files differ from the catalog.')
    for entry in entries:
        name = f"{entry['regime']}/{entry['arm']}_k{entry['kappa']}_seed{entry['seed']}.pt"
        if entry['path'] != name:
            raise ValueError(f'Unexpected checkpoint path: {entry["path"]}')
        path = directory / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f'Checkpoint missing or symlinked: {name}')
        if path.stat().st_size != entry['size_bytes'] or sha256(path) != entry['sha256']:
            raise ValueError(f'Checkpoint checksum mismatch: {name}')
    return manifest
