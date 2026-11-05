"""Check end-to-end input gradients, PGD direction and clean-centered budgets."""
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from evaluate_mnist_adaptive_pgd import DefendedMNIST, pgd_candidates


class Threshold(nn.Module):
    def forward(self, images):
        value = (images.flatten(1).mean(1)-.5)*10
        return torch.stack([-value, value], 1)


class Invert(nn.Module):
    def forward(self, images):
        return 1-images


class AdaptivePGDTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_adaptive_gradient_reaches_input_through_frozen_denoiser(self):
        model = DefendedMNIST(Threshold(), Invert()).eval().requires_grad_(False)
        clean = torch.full((1,1,4,4), .7, requires_grad=True)
        labels = torch.tensor([0])
        loss = F.cross_entropy(model(clean), labels)
        gradient, = torch.autograd.grad(loss, clean)
        step = .0001
        plus, minus = clean.detach().clone(), clean.detach().clone()
        plus[0,0,0,0] += step
        minus[0,0,0,0] -= step
        finite_difference = (F.cross_entropy(model(plus), labels)-F.cross_entropy(model(minus), labels))/(2*step)
        self.assertLess(gradient.max().item(), 0)
        self.assertAlmostEqual(gradient[0,0,0,0].item(), finite_difference.item(), places=3)

    def test_adaptive_attack_succeeds_where_classifier_attack_transfers_poorly(self):
        classifier = Threshold()
        defense = DefendedMNIST(classifier, Invert()).eval().requires_grad_(False)
        clean = torch.full((2,1,4,4), .7)
        labels = torch.zeros(2,dtype=torch.long)
        nonadaptive, _ = pgd_candidates(classifier, clean, labels, [clean], .3, .05, 8)
        adaptive, diag = pgd_candidates(defense, clean, labels, [clean], .3, .05, 8)
        self.assertTrue(defense(nonadaptive).argmax(1).eq(labels).all())
        self.assertTrue(defense(adaptive).argmax(1).ne(labels).all())
        self.assertLessEqual((adaptive-clean).abs().max().item(), .300001)
        self.assertGreaterEqual(adaptive.min().item(), 0)
        self.assertLessEqual(adaptive.max().item(), 1)
        self.assertEqual(diag['gradient_calls'], 8)
        self.assertEqual(diag['zero_gradient_samples'], 0)

    def test_projection_restarts_and_original_input_immutability(self):
        model = Threshold()
        clean = torch.tensor([.05, .95]).view(2,1,1,1).expand(2,1,4,4).clone()
        before = clean.clone()
        labels = torch.tensor([0,1])
        candidate, _ = pgd_candidates(model, clean, labels, [clean-2, clean+2], .2, .1, 3)
        self.assertTrue(torch.equal(clean,before))
        self.assertLessEqual((candidate-clean).abs().max().item(), .200001)
        self.assertGreaterEqual(candidate.min().item(), 0)
        self.assertLessEqual(candidate.max().item(), 1)


if __name__ == '__main__':
    unittest.main()
