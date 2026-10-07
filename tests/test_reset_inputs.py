"""Contract boundaries for reset-only preparation; no simulator imported."""
import importlib.util
import hashlib
import copy
from pathlib import Path
import tempfile
import unittest

path=Path(__file__).resolve().parents[1]/'gr00t_libero/reset_inputs.py'
spec=importlib.util.spec_from_file_location('reset_inputs',path)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class ResetInputsTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        root=Path(self.tmp.name);repo=root/'repo';adapter=root/'adapter'
        def ref(p):
            p.parent.mkdir(parents=True,exist_ok=True);p.write_text('fixture')
            return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
        initial=ref(root/'states')
        self.c=dict(schema='sacq_reset_inputs_v1',repo=str(repo),adapter_directory=str(adapter),output=str(root/'output'),
            settings=dict(environment_seed=17,python_seed=17,numpy_seed=17,n_envs=1,episode_ids=list(range(50)),n_action_steps=8,max_episode_steps=720,terminate_on_success=False,policy_forward_calls=0,environment_steps=0),
            tasks=[dict(suite='suite',task='task',task_id=0,language='do task',initial_file=initial)],inputs=[initial],
            sources=[ref(adapter/'seed17_adapter.py'),ref(repo/'gr00t/eval/sim/LIBERO/libero_env.py'),ref(repo/'gr00t/eval/sim/wrapper/multistep_wrapper.py')])

    def test_explicit_pinned_contract_accepted(self):m.validate(self.c,0)

    def test_changed_protocol_rejected(self):
        for key,value in [('numpy_seed',7),('n_envs',5),('episode_ids',[0]),('environment_steps',1)]:
            c=copy.deepcopy(self.c);c['settings'][key]=value
            with self.assertRaisesRegex(ValueError,'historical'):m.validate(c,0)

    def test_changed_source_rejected(self):
        Path(self.c['sources'][0]['path']).write_text('changed')
        with self.assertRaisesRegex(ValueError,'Changed'):m.validate(self.c,0)

    def test_missing_binding_and_path_traversal_rejected(self):
        c=copy.deepcopy(self.c);c['sources'].pop()
        with self.assertRaisesRegex(ValueError,'hashed source'):m.validate(c,0)
        self.c['tasks'][0]['task']='../escape'
        with self.assertRaisesRegex(ValueError,'path component'):m.validate(self.c,0)

if __name__=='__main__':unittest.main()
