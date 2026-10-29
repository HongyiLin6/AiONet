"""Verify parameters and implementation fixed before test-cache construction."""
import json
from pathlib import Path
from experiment_io import ROOT,sha

def verify_frozen():
    path=ROOT/'manifests/evaluation_choices_frozen.json'
    value=json.loads(path.read_text())
    package=Path(__file__).resolve().parents[2]
    problems=[p for p,h in value['frozen_files_sha256'].items() if not (ROOT/p).is_file() or sha(ROOT/p)!=h]
    problems += [p for p,h in value.get('code_sha256',{}).items() if not (package/p).is_file() or sha(package/p)!=h]
    if problems:raise RuntimeError('Frozen artifacts changed: '+', '.join(problems))
    return dict(all_frozen_files_unchanged=True,verified_files=len(value['frozen_files_sha256']),frozen_manifest_sha256=sha(path))
