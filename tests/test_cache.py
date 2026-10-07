"""Cache lifetime tests on synthetic CPU tensors, not model accuracy tests."""
import unittest
from unittest.mock import patch
import torch
from sacq.calibration_layers import QuantLinear
from sacq.cache import reconstruction_cache_scope


class CacheTests(unittest.TestCase):
    def layer(self):
        packed=dict(out_features=2,in_features=4,group_size=4,vector_dim=2,
                    codebook=torch.tensor([[1.,2.],[3.,4.]]),
                    scales=torch.ones(2,1),indices=torch.tensor([[[0,1]],[[1,0]]]))
        return QuantLinear(packed,None,'cpu')

    def test_reuses_only_within_scope_and_updates_after_step(self):
        layer=self.layer()
        x=torch.ones(1,4,dtype=torch.bfloat16)
        optimizer=torch.optim.SGD(layer.parameters(),lr=.1)
        with patch.object(layer,'weight_',wraps=layer.weight_) as build:
            with reconstruction_cache_scope(layer):
                first=layer(x)
                second=layer(x)
                self.assertEqual(build.call_count,1)
                (first.float().sum()+second.float().sum()).backward()
            self.assertIsNone(layer._wc)
            optimizer.step()
            with reconstruction_cache_scope(layer):
                current=layer(x)
                self.assertEqual(build.call_count,2)
            self.assertFalse(torch.equal(first,current))

    def test_no_grad_request_does_not_poison_training(self):
        layer=self.layer()
        x=torch.ones(1,4,dtype=torch.bfloat16)
        with torch.no_grad(),reconstruction_cache_scope(layer):
            layer(x)
        with reconstruction_cache_scope(layer):
            loss=layer(x).float().sum()
        loss.backward()
        self.assertIsNotNone(layer.codebook.grad)
        self.assertGreater(layer.codebook.grad.abs().sum().item(),0)

    def test_nested_module_cleanup_on_exception(self):
        layer=self.layer()
        model=torch.nn.Sequential(layer)
        with self.assertRaisesRegex(RuntimeError,'synthetic'):
            with reconstruction_cache_scope(model):
                model(torch.ones(1,4,dtype=torch.bfloat16))
                raise RuntimeError('synthetic')
        self.assertIsNone(layer._wc)
