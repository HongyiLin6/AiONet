"""Check documented runtime requirements and English-language source text."""
from pathlib import Path
import re
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT=Path(__file__).resolve().parents[1]


def requirements(path):
    return [Requirement(line.strip()) for line in path.read_text().splitlines() if line.strip() and not line.startswith('#')]


def test_installer_pins_satisfy_vendored_runtime_metadata():
    pins={canonicalize_name(r.name):next(iter(r.specifier)).version for r in requirements(ROOT/'requirements-full.txt')}
    for requirement in requirements(ROOT/'training/vendor/metrovision/mmdetection3d/requirements/runtime.txt'):
        name=canonicalize_name(requirement.name)
        assert name in pins,'Declare the direct runtime dependency: '+name
        assert pins[name] in requirement.specifier


def test_release_documentation_and_python_comments_are_english():
    paths=[ROOT/'README.md',ROOT/'SOURCE_ORIGIN.md',ROOT/'THIRD_PARTY.md',*ROOT.glob('docs/*.md'),*ROOT.rglob('*.py')]
    for path in paths:
        assert not re.search(r'[\u3400-\u9fff]',path.read_text(encoding='utf-8-sig')),str(path)
