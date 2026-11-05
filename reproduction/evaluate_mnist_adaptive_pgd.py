"""MNIST non-adaptive and adaptive end-to-end PGD evaluation."""
from adaptive_pgd import DefendedMNIST, candidate_score, load_denoiser, pgd_candidates, main

if __name__ == '__main__':
    main('mnist')
