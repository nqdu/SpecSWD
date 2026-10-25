"""Exercise optional backend imports in fresh Python processes."""

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]


def run(source):
    subprocess.run([sys.executable, '-c', source], cwd=ROOT, check=True)


def test_base_import_does_not_require_torch():
    run('''
import importlib.abc
import sys
class BlockTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise ImportError('torch intentionally unavailable')
sys.meta_path.insert(0, BlockTorch())
import specd
from specd import DispersionTable, constant_q_response
assert 'torch' not in sys.modules
assert 'specd.lib.libswd' not in sys.modules
''')


def test_torch_solve_does_not_require_native_extensions():
    run('''
import importlib.abc
import sys
class BlockNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname == 'specd.lib' or fullname.startswith('specd.lib.'):
            raise ImportError('native extensions intentionally unavailable')
sys.meta_path.insert(0, BlockNative())
from specd.torch import SpecWorkSpace
ws=SpecWorkSpace().initialize('love',[0,35,35],[2.8,2.8,3.2],
                            vsh=[3.3,3.3,5.5],vsv=[3.,3.,5.])
ws.compute_egn(.1)[0].real.backward()
assert ws.parameters['vsh'].grad is not None
assert 'specd.lib.libswd' not in sys.modules
''')
