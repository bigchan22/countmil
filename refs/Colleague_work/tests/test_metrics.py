"""Unit tests for pca/metrics.py — see spec §10.4."""
import torch

from pca.metrics import top1_acc_from_log_pmf, mae_from_log_pmf
from pca.metrics import ece_from_log_pmf, reliability_diagram_data


def test_top1_acc_argmax_matches_truth():
    """log_P argmax == bag_y on perfect predictions."""
    log_P = torch.full((3, 5), -10.0)
    log_P[0, 1] = 0.0   # argmax=1
    log_P[1, 4] = 0.0   # argmax=4
    log_P[2, 2] = 0.0   # argmax=2
    bag_y = torch.tensor([1, 4, 2])
    acc = top1_acc_from_log_pmf(log_P, bag_y)
    assert acc == 1.0


def test_top1_acc_argmax_partial():
    """3 of 4 correct → 0.75."""
    log_P = torch.full((4, 3), -10.0)
    log_P[0, 0] = 0.0
    log_P[1, 1] = 0.0
    log_P[2, 2] = 0.0
    log_P[3, 0] = 0.0     # wrong
    bag_y = torch.tensor([0, 1, 2, 1])
    acc = top1_acc_from_log_pmf(log_P, bag_y)
    assert abs(acc - 0.75) < 1e-7


def test_mae_uniform_pmf():
    """For uniform PMF over [0, T-1] and target=0, expected sum = (T-1)/2."""
    T = 11
    log_P = torch.full((1, T), -torch.tensor(float(T)).log().item())   # uniform 1/T
    bag_y = torch.tensor([0])
    mae = mae_from_log_pmf(log_P, bag_y)
    # E[Y_pred] = (T-1)/2 = 5; |E - bag_y| = 5
    assert abs(mae - 5.0) < 1e-5


def test_ece_hand_computed():
    """
    Hand-built case: 4 samples, 2 bins (n_bins=2).
      sample 0: log_P max = log(0.9) at k=0; bag_y=0  → conf=0.9, correct, bin=high
      sample 1: log_P max = log(0.9) at k=0; bag_y=1  → conf=0.9, wrong,   bin=high
      sample 2: log_P max = log(0.4) at k=0; bag_y=0  → conf=0.4, correct, bin=low
      sample 3: log_P max = log(0.4) at k=0; bag_y=1  → conf=0.4, wrong,   bin=low
    Bins (n_bins=2, edges 0,0.5,1.0):
      low bin:  acc=0.5, conf=0.4, |0.5-0.4|=0.1, weight=2/4
      high bin: acc=0.5, conf=0.9, |0.5-0.9|=0.4, weight=2/4
    ECE = 0.5*0.1 + 0.5*0.4 = 0.25
    """
    # Build log_P with controlled max prob.
    def make_log_P(p_max):
        # 2-class support; max prob = p_max at k=0, rest = (1-p_max)/(T-1) at k=1
        T = 2
        return torch.tensor([[p_max, 1 - p_max]]).log()
    log_P = torch.cat([make_log_P(0.9), make_log_P(0.9),
                        make_log_P(0.4), make_log_P(0.4)], dim=0)
    bag_y = torch.tensor([0, 1, 0, 1])
    ece = ece_from_log_pmf(log_P, bag_y, n_bins=2)
    assert abs(ece - 0.25) < 1e-6, f"got {ece}"


def test_ece_perfect_calibration_zero():
    """When confidence == accuracy in every bin, ECE = 0."""
    # All samples: confidence 0.6, half correct → bin-0.6 has acc 0.5, conf 0.5?
    # Use confidence=1.0 with all correct → ECE=0.
    log_P = torch.tensor([[0.0, -1e10]] * 4).float()    # confidence ≈ 1, prediction = 0
    bag_y = torch.tensor([0, 0, 0, 0])
    ece = ece_from_log_pmf(log_P, bag_y, n_bins=15)
    assert ece < 1e-5, f"got {ece}"


def test_reliability_diagram_data_shape():
    """reliability_diagram_data returns (bin_centers, accs, confs, weights) of length n_bins."""
    torch.manual_seed(3)
    log_P = torch.log_softmax(torch.randn(50, 8), dim=1)
    bag_y = torch.randint(0, 8, (50,))
    centers, accs, confs, weights = reliability_diagram_data(log_P, bag_y, n_bins=10)
    assert centers.shape == accs.shape == confs.shape == weights.shape == (10,)
    assert abs(weights.sum() - 1.0) < 1e-6
