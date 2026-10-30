"""Same fixed background installer; approved uniform3bit L/A overlay."""
import model_install_gptq_baseline as original
# The r2 adapter compiles under its own filename; use the immutable initial
# function text and apply the exact r2 fixed-background correction explicitly.
import ast
from pathlib import Path
p=Path(original.__file__)
tree=ast.parse(p.read_text())
node=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='install')
source=ast.get_source_segment(p.read_text(),node)
for old,new in [("DIT_BITS='16'","DIT_BITS='4'"),("'gptq2_baseline'","'gptq3_baseline'"),("r['bits'] == 2","r['bits'] == 3"),("pure_LA_GPTQ2","pure_LA_GPTQ3")]:
    assert source.count(old)==1,old
    source=source.replace(old,new)
exec(compile(source,__file__,'exec'),original.__dict__)
install=original.install
