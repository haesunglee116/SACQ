"""Static runtime equivalence checks; no simulator or GPU is initialized."""
import ast
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import unittest
from test_configuration import stable_ast

ROOT=Path(__file__).resolve().parents[1]/'gr00t_libero'
SOURCE=ROOT/'sources/gr00t_final_seed17_20260922'


class RuntimeSourceTests(unittest.TestCase):
    def check_function(self, filename, key):
        tree=ast.parse((SOURCE/filename).read_text())
        main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
        main.args.kwonlyargs=[]
        main.args.kw_defaults=[]
        if key=='simulator':
            class OriginalDevice(ast.NodeTransformer):
                def visit_Name(self,node):
                    if node.id=='egl_device':
                        return ast.copy_location(ast.Constant(value='0'),node)
                    return node
            main=OriginalDevice().visit(main)
        digest=hashlib.sha256(json.dumps(stable_ast(main),sort_keys=True).encode()).hexdigest()
        expected=json.loads((ROOT/'profiles/runtime_ast_baseline.json').read_text())['functions'][key]
        self.assertEqual(digest,expected)

    def test_rollout_operations_match_submitted_composition(self):
        self.check_function('rollout_runtime.py','rollout')

    def test_simulator_operations_match_submitted_composition(self):
        self.check_function('simulator_runtime.py','simulator')

    def test_runtime_modules_import_without_model_dependencies(self):
        for filename,params in (
            ('rollout_runtime.py',{'PROFILE','ACCEL','PROFILE_WRAPPER'}),
            ('simulator_runtime.py',{'PROFILE','egl_device'}),
        ):
            spec=importlib.util.spec_from_file_location('runtime_under_test',SOURCE/filename)
            module=importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.assertEqual(set(inspect.signature(module.main).parameters),params)

    def test_fitter_math_matches_submission(self):
        path=ROOT/'sources/libero_pro_eval/quantization_analysis/workshop_fixedalloc_refit_20260919/source/dit_fitter.py'
        tree=ast.parse(path.read_text())
        main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
        self.assertEqual([a.arg for a in main.args.kwonlyargs],['strict_metrics','checkpoint'])
        main.args.kwonlyargs=[];main.args.kw_defaults=[]
        class RemoveStrictHook(ast.NodeTransformer):
            def visit_Assign(self,node):
                if len(node.targets)==1 and ast.unparse(node.targets[0])=='farm.CKPT':
                    if isinstance(node.value,ast.IfExp):node.value=node.value.orelse
                return self.generic_visit(node)
            def visit_If(self,node):
                if isinstance(node.test,ast.Name) and node.test.id=='strict_metrics':
                    return None
                return self.generic_visit(node)
        tree=RemoveStrictHook().visit(tree)
        digest=hashlib.sha256(json.dumps(stable_ast(tree),sort_keys=True).encode()).hexdigest()
        self.assertEqual(digest,'223925eea7de704a71642c20ca36182943ec9b2f2712fc1606e87e5b6e755933')

    def test_model_manifest_uses_declared_counts_and_precision(self):
        path=ROOT/'sources/libero_transfer_quantstats_20260922/gr00t_rollout_model.py'
        tree=ast.parse(path.read_text())
        function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='validate_manifest')
        namespace={}
        exec(compile(ast.Module(body=[function],type_ignores=[]),'<manifest-test>','exec'),namespace)
        validate=namespace['validate_manifest']
        valid=dict(expected_sacq_count=338,expected_gptq_count=2,
                   gptq=[dict(layer='a',bits=3),dict(layer='b',bits=4)])
        validate(valid)
        with self.assertRaises(ValueError):
            validate(dict(valid,expected_gptq_count=105))
        with self.assertRaises(ValueError):
            validate(dict(valid,gptq=[dict(layer='a',bits=3),dict(layer='a',bits=4)]))
        with self.assertRaises(ValueError):
            validate(dict(valid,gptq=[dict(layer='a',bits=True),dict(layer='b',bits=4)]))

    def test_active_wrappers_do_not_execute_source_strings(self):
        for filename in ('rollout_worker.py','simulator_profile.py','calibration_worker.py'):
            tree=ast.parse((SOURCE/filename).read_text())
            for node in ast.walk(tree):
                if isinstance(node,ast.Call):
                    self.assertNotIn(ast.unparse(node.func),('exec','eval','inspect.getsource'))


if __name__=='__main__':
    unittest.main()


class FitterCheckpointTests(unittest.TestCase):
    def test_explicit_checkpoint_survives_argument_parsing(self):
        import argparse
        import sys
        from types import SimpleNamespace
        from unittest.mock import patch
        path=ROOT/'sources/libero_pro_eval/quantization_analysis/workshop_fixedalloc_refit_20260919/source/dit_fitter.py'
        main=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='main')
        index=next(i for i,n in enumerate(main.body) if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=='farm.CKPT')
        # Execute actual argument parsing and checkpoint binding; stop before model I/O.
        main.body=main.body[:index+1]+[ast.Return(value=ast.parse('farm.CKPT',mode='eval').body)]
        module=ast.fix_missing_locations(ast.Module(body=[main],type_ignores=[]))
        ns=dict(argparse=argparse,Path=Path,PROJ=Path('/synthetic/default'),
                torch=SimpleNamespace(device=lambda x:x),farm=SimpleNamespace(CKPT=None))
        exec(compile(module,'<fitter-binding-test>','exec'),ns)
        with patch.object(sys,'argv',['fitter','--suite','libero_goal','--k','32','--device','cpu']):
            self.assertEqual(ns['main'](checkpoint='/synthetic/selected'),Path('/synthetic/selected'))
            self.assertEqual(ns['main'](),Path('/synthetic/default/models-gr00t-n17-libero/libero_goal'))
        stage=ast.parse((ROOT/'stage.py').read_text())
        calls=[n for n in ast.walk(stage) if isinstance(n,ast.Call) and ast.unparse(n.func)=='m.main' and any(k.arg=='strict_metrics' for k in n.keywords)]
        self.assertEqual(len(calls),1)
        self.assertEqual({k.arg:ast.unparse(k.value) for k in calls[0].keywords},dict(strict_metrics='True',checkpoint='model'))
