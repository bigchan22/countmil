"""Tests for scripts/analyze_a1.py verdict + partial-mode logic.

These tests use hand-built summary dicts (no actual experiment data) to verify
the verdict tier logic and partial-mode handling. See spec §11 / plan Task 24.
"""
import sys
from pathlib import Path

# Make scripts importable from tests
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import pytest
from scripts.analyze_a1 import (
    CellStat, aggregate, evaluate_h1, evaluate_h2, evaluate_h3, evaluate_h4,
    detect_saturation, hard_preset_already_run, hard_preset_clis, verdict,
    HARD_CAP, SELECTORS,
)


def _cell(mean, seeds=None):
    seeds = seeds if seeds is not None else [mean] * 5
    return CellStat(mean=mean, std=0.01, seeds=seeds)


def _make_full_summary(pca_better=True):
    """4 datasets, 5 methods each. PCA wins on NLL if pca_better=True."""
    summary = {}
    for ds in ['mnist_sum', 'svhn_sum', 'ultramnist']:
        pca_nll = 1.0 if pca_better else 1.5
        summary[ds] = {
            'pca':  {'nll': _cell(pca_nll), 'acc': _cell(0.6),
                     'mae': _cell(2.0), 'ece': _cell(0.05)},
            '2b':   {'nll': _cell(1.4),     'acc': _cell(0.5),
                     'mae': _cell(2.5), 'ece': _cell(0.10)},
            '3a':   {'nll': _cell(1.4),     'acc': _cell(0.5),
                     'mae': _cell(2.5), 'ece': _cell(0.12)},
            '3b':   {'nll': _cell(1.5),     'acc': _cell(0.45),
                     'mae': _cell(2.7), 'ece': _cell(0.15)},
        }
    summary['mnist_signed'] = {
        'pca': {'nll': _cell(0.7), 'acc': _cell(0.45),
                'mae': _cell(1.0), 'ece': _cell(0.04)},
        '1a':  {'nll': _cell(2.0), 'acc': _cell(0.10),
                'mae': _cell(3.0), 'ece': _cell(0.10)},
    }
    return summary


def test_partial_mode_h2_skips_absent_datasets(tmp_path):
    """C1 regression: in partial mode, absent datasets must not record pass=False.

    Under H2 relative threshold each present dataset needs at least one baseline
    (so PCA acc top-1 can be checked) — fixture includes 1a baselines.
    """
    summary = {
        'mnist_sum': {
            'pca': {'nll': _cell(1.0), 'acc': _cell(0.6),
                    'mae': _cell(2.0), 'ece': _cell(0.05)},
            '1a':  {'nll': _cell(2.5), 'acc': _cell(0.2),
                    'mae': _cell(4.0), 'ece': _cell(0.10)},
        },
        'mnist_signed': {
            'pca': {'nll': _cell(0.7), 'acc': _cell(0.45),
                    'mae': _cell(1.0), 'ece': _cell(0.04)},
            '1a':  {'nll': _cell(2.0), 'acc': _cell(0.10),
                    'mae': _cell(3.0), 'ece': _cell(0.10)},
        },
    }
    # Need a fake mnist_mil/summary.json for B1 reuse check
    (tmp_path / 'mnist_mil').mkdir()
    import json as _json
    (tmp_path / 'mnist_mil' / 'summary.json').write_text(_json.dumps({
        'summary': {'50': {'nll': {'mean': 0.99}}}
    }))
    h2 = evaluate_h2(summary, tmp_path, partial=True)
    # Should NOT contain entries for svhn_sum, ultramnist (skipped, not failed)
    assert 'multiclass_svhn_sum' not in h2['checks'], "partial mode leaked absent fail"
    assert 'multiclass_ultramnist' not in h2['checks'], "partial mode leaked absent fail"
    assert 'multiclass_mnist_sum' in h2['checks']
    assert 'signed_mnist_signed' in h2['checks']
    assert h2['checks']['multiclass_mnist_sum']['pass']
    assert h2['checks']['signed_mnist_signed']['pass']
    assert h2['checks']['binary_b1_reuse']['pass']
    assert h2['pass']


def test_full_mode_h2_records_absent_as_fail(tmp_path):
    """Default mode: absent datasets recorded as pass=False."""
    summary = {
        'mnist_sum': {
            'pca': {'nll': _cell(1.0), 'acc': _cell(0.6),
                    'mae': _cell(2.0), 'ece': _cell(0.05)},
        },
    }
    (tmp_path / 'mnist_mil').mkdir()
    import json as _json
    (tmp_path / 'mnist_mil' / 'summary.json').write_text(_json.dumps({
        'summary': {'50': {'nll': {'mean': 0.99}}}
    }))
    h2 = evaluate_h2(summary, tmp_path, partial=False)
    assert 'multiclass_svhn_sum' in h2['checks']
    assert not h2['checks']['multiclass_svhn_sum']['pass']
    assert not h2['pass']


def test_hard_preset_already_run_per_dataset(tmp_path):
    """C2 regression: must return per-dataset dict, not global bool."""
    # mnist_sum saturated, has cap=50 hard run. ultramnist saturated, no hard run.
    (tmp_path / 'mnist_sum' / 'N10_cap50_sig0.1_pca').mkdir(parents=True)
    import json as _json
    (tmp_path / 'mnist_sum' / 'N10_cap50_sig0.1_pca' / 'seed0.json').write_text(
        _json.dumps({'config': {'per_class_cap': 50}}))
    (tmp_path / 'ultramnist' / 'N4_capnone_pca').mkdir(parents=True)
    (tmp_path / 'ultramnist' / 'N4_capnone_pca' / 'seed0.json').write_text(
        _json.dumps({'config': {'per_class_cap': None}}))
    saturation = {'mnist_sum': True, 'ultramnist': True, 'svhn_sum': False}
    out = hard_preset_already_run(tmp_path, saturation)
    assert out == {'mnist_sum': True, 'ultramnist': False, 'svhn_sum': False}, \
        f"got {out}"


def test_verdict_run_hard_preset_skips_done_dataset(tmp_path):
    """C2 regression: verdict still RUN_HARD_PRESET when one ds done, another not."""
    summary = _make_full_summary(pca_better=False)  # H1 fails everywhere
    h1 = evaluate_h1(summary, ['mnist_sum', 'svhn_sum', 'ultramnist'])
    (tmp_path / 'mnist_mil').mkdir()
    (tmp_path / 'mnist_mil' / 'summary.json').write_text(json.dumps({
        'summary': {'50': {'nll': {'mean': 0.99}}}
    }))
    h2 = evaluate_h2(summary, tmp_path)
    h3 = evaluate_h3(summary, ['mnist_sum', 'svhn_sum', 'ultramnist'])
    h4 = evaluate_h4(summary, ['mnist_sum', 'svhn_sum', 'ultramnist'])
    saturation = {'mnist_sum': True, 'ultramnist': True}
    # Pre-create mnist_sum hard preset done, ultramnist not done
    (tmp_path / 'mnist_sum' / 'N10_cap50_sig0.1_pca').mkdir(parents=True)
    (tmp_path / 'mnist_sum' / 'N10_cap50_sig0.1_pca' / 'seed0.json').write_text(
        json.dumps({'config': {'per_class_cap': 50}}))
    v = verdict(h1, h2, h3, h4, saturation, tmp_path)
    assert v == 'RUN_HARD_PRESET', f"got {v}; ultramnist still needs its hard preset"


def test_hard_preset_clis_skips_done(tmp_path):
    saturation = {'mnist_sum': True, 'ultramnist': True}
    already = {'mnist_sum': True, 'ultramnist': False}
    clis = hard_preset_clis(saturation, already)
    # mnist_sum CLIs should not appear; ultramnist CLIs should
    assert not any('run_mnist_sum.py' in c for c in clis)
    assert any('run_ultramnist.py' in c for c in clis)


# ---------------------------------------------------------------------------
# Track 3-A additions: H4 (point-prediction supremacy), new verdict tier,
# and per-seed selector functions.
# ---------------------------------------------------------------------------


def test_evaluate_h4_pass_when_pca_wins_acc_and_mae():
    """PCA acc beats max baseline by >0.05 AND mae beats min baseline by >0.05
       on ≥2 of 3 multi-class datasets."""
    summary = _make_full_summary()  # PCA acc=0.6, max baseline acc=0.5; mae 2.0 vs 2.5
    h4 = evaluate_h4(summary, ['mnist_sum', 'svhn_sum', 'ultramnist'])
    assert h4['pass'], h4
    assert h4['pass_count'] == 3
    for ds in ['mnist_sum', 'svhn_sum', 'ultramnist']:
        d = h4['per_dataset'][ds]
        assert d['available']
        assert d['acc_win'] and d['mae_win']
        assert d['pass']


def test_evaluate_h4_fail_when_acc_margin_too_thin():
    """If max baseline acc is within 0.05 of PCA acc, H4 fails (acc gap fails)."""
    summary = {}
    for ds in ['mnist_sum', 'svhn_sum', 'ultramnist']:
        summary[ds] = {
            'pca': {'nll': _cell(1.0), 'acc': _cell(0.6),
                    'mae': _cell(2.0), 'ece': _cell(0.05)},
            '2b':  {'nll': _cell(1.4), 'acc': _cell(0.57),  # 0.03 gap < 0.05
                    'mae': _cell(2.5), 'ece': _cell(0.10)},
        }
    h4 = evaluate_h4(summary, ['mnist_sum', 'svhn_sum', 'ultramnist'])
    assert not h4['pass'], h4
    for ds in ['mnist_sum', 'svhn_sum', 'ultramnist']:
        d = h4['per_dataset'][ds]
        assert d['available']
        assert not d['acc_win']
        assert d['mae_win']  # mae still wins
        assert not d['pass']


def test_verdict_a1_point_prediction_win(tmp_path):
    """New tier: H1 fail + H3 fail + H4 pass → A1_POINT_PREDICTION_WIN."""
    # Build a summary where:
    # - H1 fails (PCA NLL > best baseline)
    # - H3 fails (PCA ECE > 0.7 × min baseline ECE)
    # - H4 passes (PCA wins acc + MAE)
    summary = {}
    for ds in ['mnist_sum', 'svhn_sum', 'ultramnist']:
        summary[ds] = {
            'pca': {'nll': _cell(2.5), 'acc': _cell(0.6),
                    'mae': _cell(1.0), 'ece': _cell(0.30)},  # high ECE
            '2b':  {'nll': _cell(2.0), 'acc': _cell(0.5),   # better NLL
                    'mae': _cell(2.0), 'ece': _cell(0.05)},  # better ECE
            '3a':  {'nll': _cell(2.1), 'acc': _cell(0.4),
                    'mae': _cell(2.5), 'ece': _cell(0.06)},
            '3b':  {'nll': _cell(2.2), 'acc': _cell(0.4),
                    'mae': _cell(2.6), 'ece': _cell(0.06)},
        }
    summary['mnist_signed'] = {
        'pca': {'nll': _cell(0.7), 'acc': _cell(0.45),
                'mae': _cell(1.0), 'ece': _cell(0.04)},
        '1a':  {'nll': _cell(2.0), 'acc': _cell(0.10),
                'mae': _cell(3.0), 'ece': _cell(0.10)},
    }
    h1 = evaluate_h1(summary, ['mnist_sum', 'svhn_sum', 'ultramnist'])
    (tmp_path / 'mnist_mil').mkdir()
    (tmp_path / 'mnist_mil' / 'summary.json').write_text(json.dumps({
        'summary': {'50': {'nll': {'mean': 0.99}}}
    }))
    h2 = evaluate_h2(summary, tmp_path)
    h3 = evaluate_h3(summary, ['mnist_sum', 'svhn_sum', 'ultramnist'])
    h4 = evaluate_h4(summary, ['mnist_sum', 'svhn_sum', 'ultramnist'])
    assert not h1['pass']
    assert h2['pass']
    assert not h3['pass']
    assert h4['pass']
    saturation = {'mnist_sum': False, 'svhn_sum': False, 'ultramnist': False}
    v = verdict(h1, h2, h3, h4, saturation, tmp_path)
    assert v == 'A1_POINT_PREDICTION_WIN', f"got {v}"


def test_verdict_demote_a1_when_h4_also_fails(tmp_path):
    """H2 pass + H1+H3+H4 all fail (and saturation clear) → DEMOTE_A1.

    Under H2 relative threshold, DEMOTE_A1 requires PCA top-1 on acc (so H2
    passes). Fixture: PCA wins acc by 0.01 (just enough for H2 strict >, not
    enough for H4 acc_margin=0.05).
    """
    summary = {}
    for ds in ['mnist_sum', 'svhn_sum', 'ultramnist']:
        summary[ds] = {
            # PCA top-1 acc by 0.01 — H2 passes, H4 acc gap < margin so fails
            'pca': {'nll': _cell(2.5), 'acc': _cell(0.51),
                    'mae': _cell(2.5), 'ece': _cell(0.30)},
            '2b':  {'nll': _cell(2.0), 'acc': _cell(0.50),
                    'mae': _cell(2.0), 'ece': _cell(0.05)},
            '3a':  {'nll': _cell(2.1), 'acc': _cell(0.50),
                    'mae': _cell(2.0), 'ece': _cell(0.06)},
            '3b':  {'nll': _cell(2.2), 'acc': _cell(0.50),
                    'mae': _cell(2.0), 'ece': _cell(0.06)},
        }
    summary['mnist_signed'] = {
        'pca': {'nll': _cell(0.7), 'acc': _cell(0.45),
                'mae': _cell(1.0), 'ece': _cell(0.04)},
        '1a':  {'nll': _cell(2.0), 'acc': _cell(0.10),
                'mae': _cell(3.0), 'ece': _cell(0.10)},
    }
    h1 = evaluate_h1(summary, ['mnist_sum', 'svhn_sum', 'ultramnist'])
    (tmp_path / 'mnist_mil').mkdir()
    (tmp_path / 'mnist_mil' / 'summary.json').write_text(json.dumps({
        'summary': {'50': {'nll': {'mean': 0.99}}}
    }))
    h2 = evaluate_h2(summary, tmp_path)
    h3 = evaluate_h3(summary, ['mnist_sum', 'svhn_sum', 'ultramnist'])
    h4 = evaluate_h4(summary, ['mnist_sum', 'svhn_sum', 'ultramnist'])
    assert h2['pass'], h2
    assert not h1['pass']
    assert not h3['pass']
    assert not h4['pass']
    saturation = {'mnist_sum': False, 'svhn_sum': False, 'ultramnist': False}
    v = verdict(h1, h2, h3, h4, saturation, tmp_path)
    assert v == 'DEMOTE_A1', f"got {v}"


def test_selector_tail5_reads_stored_best_field():
    """tail5 selector returns the JSON's pre-computed best_test_* fields."""
    d = {
        'best_test_nll': 3.290, 'best_test_acc': 0.635,
        'best_test_mae': 1.360, 'best_test_ece': 0.267,
        'epoch_test_nll': [10.0, 1.5, 2.0],   # would-be min is 1.5
        'epoch_train_loss': [3.0, 2.0, 1.0],
        'epoch_test_acc': [0.1, 0.5, 0.6],
        'epoch_test_mae': [3.0, 2.0, 1.4],
        'epoch_test_ece': [0.5, 0.3, 0.27],
    }
    out = SELECTORS['tail5'](d)
    assert out == {'nll': 3.290, 'acc': 0.635, 'mae': 1.360, 'ece': 0.267}


def test_selector_min_test_nll_picks_min_epoch():
    """min_test_nll selector reads the epoch with min(epoch_test_nll)."""
    d = {
        'best_test_nll': 3.290, 'best_test_acc': 0.635,  # ignored
        'best_test_mae': 1.360, 'best_test_ece': 0.267,
        'epoch_test_nll':  [10.0, 1.5, 2.0],   # min at idx=1
        'epoch_test_acc':  [0.1,  0.5, 0.6],
        'epoch_test_mae':  [3.0,  2.0, 1.4],
        'epoch_test_ece':  [0.5,  0.3, 0.27],
        'epoch_train_loss': [3.0, 2.0, 1.0],
    }
    out = SELECTORS['min_test_nll'](d)
    assert out['nll'] == 1.5
    assert out['acc'] == 0.5
    assert out['mae'] == 2.0
    assert out['ece'] == 0.3


def test_selector_argmin_train_loss_uses_train_index():
    """argmin_train_loss selector reads test metrics at the epoch with min train loss."""
    d = {
        'best_test_nll': 3.290, 'best_test_acc': 0.635,
        'best_test_mae': 1.360, 'best_test_ece': 0.267,
        'epoch_test_nll':   [10.0, 1.5, 2.0],
        'epoch_test_acc':   [0.1,  0.5, 0.6],
        'epoch_test_mae':   [3.0,  2.0, 1.4],
        'epoch_test_ece':   [0.5,  0.3, 0.27],
        'epoch_train_loss': [3.0,  2.0, 1.0],   # argmin at idx=2 (last)
    }
    out = SELECTORS['argmin_train_loss'](d)
    assert out['nll'] == 2.0
    assert out['acc'] == 0.6
    assert out['mae'] == 1.4
    assert out['ece'] == 0.27


def test_aggregate_with_selector_returns_different_values(tmp_path):
    """End-to-end: aggregate under tail5 vs min_test_nll on the same fixture
       returns different per-method NLL values."""
    setting_dir = tmp_path / 'mnist_sum' / 'N10_cap100_sig0.0_pca'
    setting_dir.mkdir(parents=True)
    (setting_dir / 'seed0.json').write_text(json.dumps({
        'best_test_nll': 3.0, 'best_test_acc': 0.5,
        'best_test_mae': 2.0, 'best_test_ece': 0.2,
        'epoch_test_nll':  [4.0, 1.0, 5.0],   # min=1.0 at idx=1
        'epoch_test_acc':  [0.3, 0.4, 0.5],
        'epoch_test_mae':  [3.0, 2.5, 2.0],
        'epoch_test_ece':  [0.3, 0.1, 0.2],
        'epoch_train_loss': [5.0, 3.0, 1.0],  # argmin at idx=2
    }))
    s_tail5 = aggregate(tmp_path, ['mnist_sum'], selector='tail5')
    s_min   = aggregate(tmp_path, ['mnist_sum'], selector='min_test_nll')
    s_train = aggregate(tmp_path, ['mnist_sum'], selector='argmin_train_loss')
    assert s_tail5['mnist_sum']['pca']['nll'].mean == 3.0
    assert s_min['mnist_sum']['pca']['nll'].mean == 1.0
    assert s_train['mnist_sum']['pca']['nll'].mean == 5.0  # idx=2 → epoch_test_nll[2]


# ---------------------------------------------------------------------------
# H2 relative-threshold tests (replaces previous absolute acc>0.30 / >0.20)
# ---------------------------------------------------------------------------


def test_h2_pca_top1_passes_below_old_absolute_threshold(tmp_path):
    """SVHN-style: PCA acc=0.245 (below old 0.30 absolute) but top-1 → PASS.

    Captures the threshold-artifact case the relative rewrite is meant to fix:
    SVHN per-class random ≈ 1/91, so 0.245 is ~22× random and best-among-
    methods, but the MNIST-calibrated 0.30 floor rejected it.
    """
    summary = {
        'svhn_sum': {
            'pca': {'nll': _cell(3.65), 'acc': _cell(0.245),
                    'mae': _cell(3.08), 'ece': _cell(0.30)},
            '2b':  {'nll': _cell(4.94), 'acc': _cell(0.116),
                    'mae': _cell(3.67), 'ece': _cell(0.12)},
            '3a':  {'nll': _cell(6.34), 'acc': _cell(0.031),
                    'mae': _cell(9.46), 'ece': _cell(0.31)},
        }
    }
    (tmp_path / 'mnist_mil').mkdir()
    (tmp_path / 'mnist_mil' / 'summary.json').write_text(json.dumps({
        'summary': {'50': {'nll': {'mean': 0.99}}}
    }))
    h2 = evaluate_h2(summary, tmp_path, partial=True)
    assert h2['checks']['multiclass_svhn_sum']['pass'], h2['checks']
    assert '0.245' in h2['checks']['multiclass_svhn_sum']['detail']
    assert '0.116' in h2['checks']['multiclass_svhn_sum']['detail']


def test_h2_pca_below_baseline_fails(tmp_path):
    """If any baseline beats PCA on acc → multiclass H2 FAILS."""
    summary = {
        'mnist_sum': {
            'pca': {'nll': _cell(1.0), 'acc': _cell(0.45),
                    'mae': _cell(2.0), 'ece': _cell(0.05)},
            '2a':  {'nll': _cell(1.5), 'acc': _cell(0.55),  # 2a beats PCA
                    'mae': _cell(2.5), 'ece': _cell(0.10)},
        }
    }
    (tmp_path / 'mnist_mil').mkdir()
    (tmp_path / 'mnist_mil' / 'summary.json').write_text(json.dumps({
        'summary': {'50': {'nll': {'mean': 0.99}}}
    }))
    h2 = evaluate_h2(summary, tmp_path, partial=True)
    assert not h2['checks']['multiclass_mnist_sum']['pass'], h2['checks']


def test_h2_pca_tied_with_baseline_fails(tmp_path):
    """Strict greater: PCA acc tied with max baseline → FAIL (not top-1)."""
    summary = {
        'mnist_sum': {
            'pca': {'nll': _cell(1.0), 'acc': _cell(0.50),
                    'mae': _cell(2.0), 'ece': _cell(0.05)},
            '2a':  {'nll': _cell(1.5), 'acc': _cell(0.50),  # tie
                    'mae': _cell(2.5), 'ece': _cell(0.10)},
        }
    }
    (tmp_path / 'mnist_mil').mkdir()
    (tmp_path / 'mnist_mil' / 'summary.json').write_text(json.dumps({
        'summary': {'50': {'nll': {'mean': 0.99}}}
    }))
    h2 = evaluate_h2(summary, tmp_path, partial=True)
    assert not h2['checks']['multiclass_mnist_sum']['pass']


def test_h2_no_baselines_fails(tmp_path):
    """Edge case: only PCA in dataset → cannot verify top-1 → FAIL."""
    summary = {
        'mnist_sum': {
            'pca': {'nll': _cell(1.0), 'acc': _cell(0.6),
                    'mae': _cell(2.0), 'ece': _cell(0.05)},
        }
    }
    (tmp_path / 'mnist_mil').mkdir()
    (tmp_path / 'mnist_mil' / 'summary.json').write_text(json.dumps({
        'summary': {'50': {'nll': {'mean': 0.99}}}
    }))
    h2 = evaluate_h2(summary, tmp_path, partial=True)
    check = h2['checks']['multiclass_mnist_sum']
    assert not check['pass']
    assert 'no baselines' in check['detail']
