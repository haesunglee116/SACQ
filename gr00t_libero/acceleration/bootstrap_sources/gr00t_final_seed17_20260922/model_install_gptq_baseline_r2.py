"""Enable fixed VLSA HQQ4 construction; L/A remain explicitly forced16 until overlay."""
import inspect
import model_install_gptq_baseline as original
source=inspect.getsource(original.install)
assert source.count("DIT_BITS='16'")==1
source=source.replace("DIT_BITS='16'", "DIT_BITS='4'")
exec(compile(source,__file__,'exec'),original.__dict__)
install=original.install
