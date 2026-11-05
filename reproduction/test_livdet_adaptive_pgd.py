"""Check exact PNG BPDA forward and input gradients through LivDet defense."""
import unittest
import torch
from torch import nn

from adaptive_pgd import DefendedLivDet, png_bpda, pgd_candidates
from evaluate_legacy_livdet import classifier_input


class Threshold(nn.Module):
    def forward(self, images):
        value = 10*(images.flatten(1).mean(1)-.5)
        return torch.stack([-value, value], dim=1)


class LivDetAdaptiveTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_bpda_forward_is_exact_legacy_and_backward_clips_channels(self):
        rgb = torch.zeros(1,3,581,581)
        rgb[:,0] = .431
        rgb[:,1] = 1.2
        rgb[:,2] = -.1
        rgb.requires_grad_(True)
        actual = png_bpda(rgb)
        expected = classifier_input(rgb.detach(), True)
        self.assertTrue(torch.equal(actual, expected))
        actual.sum().backward()
        self.assertTrue(torch.allclose(rgb.grad[:,0], torch.full_like(rgb.grad[:,0], .299)))
        self.assertEqual(rgb.grad[:,1:].abs().sum().item(), 0)

    def test_bpda_attack_has_nonzero_gradient_and_respects_budget(self):
        model = DefendedLivDet(Threshold(), nn.Identity()).eval().requires_grad_(False)
        clean = torch.full((1,1,581,581), .51)
        labels = torch.tensor([1])
        attacked, diagnostics = pgd_candidates(model, clean, labels, [clean], .03, .01, 4)
        self.assertEqual(model(clean).argmax(1).item(), 1)
        self.assertEqual(model(attacked).argmax(1).item(), 0)
        self.assertLessEqual((attacked-clean).abs().max().item(), .030001)
        self.assertEqual(diagnostics['zero_gradient_samples'], 0)


if __name__ == '__main__':
    unittest.main()
