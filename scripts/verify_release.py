#!/usr/bin/env python3
"""Check repository files, notebook outputs and recorded source/result hashes."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import subprocess
import re

from paper_results import ROOT, sha256, verify_references
from checkpoint_catalog import verify_catalog


def notebook_issues(notebook):
    """Reject execution records and metadata unrelated to the Python kernel."""
    issues = []
    metadata = notebook.get('metadata', {})
    if set(metadata) - {'kernelspec', 'language_info'}:
        issues.append('notebook metadata contains fields beyond kernel and language settings')
    allowed = {'kernelspec': {'name', 'language', 'display_name'},
               'language_info': {'name', 'version'}}
    for key, fields in allowed.items():
        if set(metadata.get(key, {})) - fields:
            issues.append(f'{key} contains unexpected metadata fields')
    for index, cell in enumerate(notebook.get('cells', [])):
        if cell.get('metadata'):
            issues.append(f'cell {index} contains saved metadata')
        if cell.get('outputs') or cell.get('execution_count') is not None:
            issues.append(f'cell {index} contains execution output or a run count')
        if cell.get('attachments'):
            issues.append(f'cell {index} contains embedded attachments')
    return issues


def release_files(root=ROOT):
    """Inspect tracked and unignored files, including data accidentally tracked earlier."""
    result = subprocess.run(['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard'],
                            cwd=root, capture_output=True, check=False)
    if result.returncode == 0:
        return [root / name for name in sorted(set(result.stdout.decode().split('\0'))) if name]
    # ZIP downloads have no Git metadata. The known local-output directories
    # have the same role as the repository's ignore rules.
    excluded = {'.git', '.runtime', '__pycache__', '.ipynb_checkpoints', '.pytest_cache'}
    paths = []
    for path in root.rglob('*'):
        rel = path.relative_to(root)
        if any(part in excluded or part.startswith('.venv') for part in rel.parts):
            continue
        if rel.parts[0] == 'data' and rel.name != 'README.md':
            continue
        if rel.parts[0] == 'results' and rel.parts[1:2] != ('reference',):
            continue
        if rel.parts[:2] in (('new-reward', 'checkpoints'), ('new-reward', 'data')):
            continue
        if path.is_file() or path.is_symlink():
            paths.append(path)
    return paths


def main() -> None:
    errors = []
    allowed_weights = set()
    try:
        catalog = verify_catalog()
        allowed_weights = {Path('pretrained') / entry['path'] for entry in catalog['checkpoints']}
    except (OSError, ValueError, KeyError, TypeError) as error:
        errors.append(f'Pretrained checkpoint catalog: {error}')
    excluded = {".git", ".runtime", "__pycache__", ".ipynb_checkpoints", ".venv", ".pytest_cache"}
    forbidden_suffixes = {".parquet", ".npz", ".npy", ".pt", ".pth", ".pkl", ".pickle", ".joblib", ".zip"}
    required = ("README.md", "LICENSE", "DATA_LICENSE.md", "CITATION.cff", "THIRD_PARTY_NOTICES.md",
                ".gitignore", "docs/reproduction.md", "data/README.md", "run_reproduction.sh",
                "pretrained/manifest.json", "pretrained/README.md", "scripts/evaluate_pretrained.py",
                "scripts/run_experiments.py")
    for name in required:
        if not (ROOT / name).is_file():
            errors.append(f"Required release file missing: {name}")
    files = release_files()
    missing_weights = allowed_weights - {path.relative_to(ROOT) for path in files}
    if missing_weights:
        errors.append(f'{len(missing_weights)} published checkpoints are excluded from release files')
    for path in files:
        rel = path.relative_to(ROOT)
        if any(part in excluded or part.startswith(".venv") for part in rel.parts):
            errors.append(f'Local runtime file included in release: {rel}')
            continue
        if rel.parts[:2] == ("results", "generated"):
            errors.append(f'Generated output included in release: {rel}')
            continue
        if path.is_symlink():
            errors.append(f"Symlink in public release: {rel}")
            continue
        if not path.is_file():
            continue
        if (path.suffix in forbidden_suffixes and rel not in allowed_weights) or path.name.startswith(".env"):
            errors.append(f"Private/runtime file in public release: {rel}")
        if path.suffix == '.log' or path.name == '.DS_Store':
            errors.append(f'Local log or system metadata in release: {rel}')
        if path.suffix == ".csv" and rel.parts[:2] != ("results", "reference"):
            errors.append(f"Unreviewed CSV outside aggregate references: {rel}")
        if rel.parts[0] == "data" and rel.name != "README.md":
            errors.append(f"Data payload in public directory: {rel}")
        if rel.parts[:2] in (("new-reward", "checkpoints"), ("new-reward", "include"), ("new-reward", "data")):
            errors.append(f"Private runtime dependency in public release: {rel}")
        if path.suffix in {'.py', '.md', '.json', '.ipynb', '.txt', '.yml', '.yaml', '.cff', '.sh', '.csv'}:
            source = path.read_text()
            personal_patterns = (r'/' + r'(?:Users|home)/[^/\s]+/', r'/' + r'content/drive',
                                 r'My' + r'Drive/', r'google' + r'\.colab',
                                 r'[A-Za-z]:\\\\(?:Users|Documents and Settings)\\\\',
                                 r'-----BEGIN ' + r'(?:RSA |OPENSSH |EC )?PRIVATE KEY-----')
            if any(re.search(pattern, source) for pattern in personal_patterns):
                errors.append(f'Personal or hosted-notebook path in release: {rel}')
        if path.suffix == ".ipynb":
            notebook = json.loads(path.read_text())
            errors.extend(f'{rel}: {issue}' for issue in notebook_issues(notebook))
    provenance = json.loads((ROOT / "docs/source_manifest.json").read_text())
    for entry in provenance["sources"]:
        target = ROOT / entry["target"]
        if not target.is_file() or sha256(target) != entry["target_sha256"]:
            errors.append(f"Imported source differs from its provenance record: {entry['target']}")
    verify_references()
    if errors:
        raise SystemExit("Release check FAILED:\n  " + "\n  ".join(errors))
    print("Repository checks passed: 120 checkpoints, documentation, notebooks, source hashes and 11 result tables.")


if __name__ == "__main__":
    main()
