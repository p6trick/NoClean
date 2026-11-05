"""Check distances, denominators and rectification iteration bookkeeping."""
import unittest
from types import SimpleNamespace

import torch
from torch import nn

from analyze_rectification import rmse, summarize
from run import rectify


class Threshold(nn.Module):
    def forward(self, images):
        value = (images.flatten(1).mean(1)-.5)*10
        return torch.stack([-value,value],1)


class MechanismTests(unittest.TestCase):
    def test_rmse_is_per_image_not_batch_global(self):
        left = torch.tensor([0.,1.]).view(2,1,1,1).expand(2,1,2,2)
        right = torch.zeros_like(left)
        self.assertTrue(torch.equal(rmse(left,right),torch.tensor([0.,1.],dtype=torch.float64)))
        self.assertAlmostEqual(rmse(left,right).mean().item(),.5)

    def test_zero_iteration_recovery_and_failure_are_counted_separately(self):
        model = Threshold().eval().requires_grad_(False)
        clean = torch.tensor([.7,.7,.1]).view(3,1,1,1).expand(3,1,4,4).clone()
        attack = torch.tensor([.7,.4,.9]).view(3,1,1,1).expand(3,1,4,4).clone()
        labels = torch.tensor([1,1,0])
        args = SimpleNamespace(rectify_steps=3,rectify_alpha=.1,rectify_epsilon=.25,
                               rectify_stop='correct',gradient_loss='logits-ce')
        result,iterations = rectify(model,attack,labels,args)
        success = model(result).argmax(1).eq(labels)
        self.assertEqual(iterations.tolist(),[0,2,3])
        self.assertEqual(success.tolist(),[True,True,False])
        self.assertLessEqual((result-attack).abs().max().item(),.250001)
        da,dr = rmse(attack,clean),rmse(result,clean)
        records = [{'success':bool(success[i]),'initially_correct':i==0,
                    'iterations':int(iterations[i]),'adv_clean_rmse':da[i].item(),
                    'rect_clean_rmse':dr[i].item(),'rectify_linf':(result[i]-attack[i]).abs().max().item(),
                    'attack_linf':(attack[i]-clean[i]).abs().max().item(),
                    'failure_at_limit':i==2} for i in range(3)]
        summary = summarize(records,1e-12)
        self.assertAlmostEqual(summary['success_percent'],200/3)
        self.assertAlmostEqual(summary['failure_percent'],100/3)
        self.assertEqual(summary['avg_iterations_success_including_zero'],1)
        self.assertEqual(summary['avg_iterations_success_positive_only'],2)
        self.assertEqual(summary['recovery_among_initially_wrong_percent'],50)
        self.assertAlmostEqual(summary['closer_percent'],200/3)

    def test_no_success_has_no_iteration_mean(self):
        records = [{'success':False,'initially_correct':False,'iterations':40,
                    'adv_clean_rmse':.1,'rect_clean_rmse':.2,'rectify_linf':.03,
                    'attack_linf':.1,'failure_at_limit':True}]
        summary = summarize(records,1e-12)
        self.assertIsNone(summary['avg_iterations_success_including_zero'])
        self.assertEqual(summary['failure_percent'],100)
        self.assertEqual(summary['closer_percent'],0)


if __name__=='__main__':
    unittest.main()
