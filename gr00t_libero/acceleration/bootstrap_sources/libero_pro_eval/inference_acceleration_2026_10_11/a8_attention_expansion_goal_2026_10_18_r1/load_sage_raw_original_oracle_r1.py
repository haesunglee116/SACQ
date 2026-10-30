"""Load pinned original math/raw bindings without duplicate torch custom-op registration."""
import hashlib
import importlib
import importlib.machinery
import json
import sys
import types
from pathlib import Path


def load(receipt_ref):
    p = Path(receipt_ref['path'])
    assert hashlib.sha256(p.read_bytes()).hexdigest() == receipt_ref['sha256']
    receipt = json.loads(p.read_text())
    path = Path(receipt['pythonpath'])/'sageattention'
    name = 'sageattention_raw_original_oracle'
    assert name not in sys.modules
    package = types.ModuleType(name)
    package.__path__ = [str(path)]
    package.__file__ = str(path/'__init__.py')
    package.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
    sys.modules[name] = package
    for arch in ['sm80', 'sm89']:
        raw = importlib.import_module(name+'._qattn_'+arch)
        assert str(Path(raw.__file__).resolve()).startswith(str(path.resolve())+'/')
        sys.modules[name+'.'+arch+'_compile'] = raw
        setattr(package, arch+'_compile', raw)
    # This build contains no SM90 binary. Never import its decorator module.
    sys.modules[name+'.sm90_compile'] = None
    core = importlib.import_module(name+'.core')
    assert core.SM89_ENABLED and not core.SM90_ENABLED
    assert core.sm89_compile is sys.modules[name+'._qattn_sm89']
    assert Path(core.__file__).resolve() == (path/'core.py').resolve()
    return core
