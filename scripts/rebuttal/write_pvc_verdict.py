#!/usr/bin/env python
"""Write machine-readable and Markdown PVC audit verdict files."""

from __future__ import annotations

import json
from pathlib import Path


VERDICT = {
    "classification": "C",
    "classification_label": "locally modified approximation of the classwise count-likelihood component",
    "is_complete_official_llp_pvc": False,
    "is_count_likelihood_component": True,
    "may_call_existing_rows_full_llp_pvc": False,
    "recommended_label": "FS-Conv classwise count likelihood / LLP-PVC count-likelihood component",
    "numerical_equivalence": {
        "scope": "matched softmax parameterization for the classwise Poisson-binomial count-likelihood component",
        "forward_loss_difference": 4.44e-16,
        "per_class_pmf_difference": 2.22e-16,
        "logit_gradient_difference": 2.12e-16,
    },
    "comparison": {
        "output_parameterization": {
            "local": "softmax(logits) used for classwise count likelihood and argmax prediction",
            "official_code": "official LLP-PVC script contains sigmoid(logits) path for count loss and softmax(logits) for pseudo-label/evaluation logic",
            "verdict": "not procedurally identical",
        },
        "loss": {
            "local": "mean over classes of negative log one-vs-rest Poisson-binomial count probability",
            "official_code": "Poisson-binomial count loss component, with official training script surrounding it",
            "verdict": "same mathematical component under matched probabilities, not the complete official recipe",
        },
        "prediction_rule": {
            "local": "argmax softmax probabilities",
            "official_code": "softmax scores in evaluation path; EMA evaluation is present in the official script",
            "verdict": "not identical as a complete training/evaluation procedure",
        },
        "auxiliary_terms": {
            "local": "none",
            "official_code": "official script includes EMA/model-management, pseudo-label diagnostics, warmup scheduler infrastructure, and bag construction variants",
            "verdict": "local implementation omits official procedure details",
        },
        "optimization_checkpointing": {
            "local": "Adam/SGD depending script; checkpoint by observed histogram metric in current CIFAR code",
            "official_code": "SGD with Nesterov/warmup-cosine infrastructure; reported best test/EMA bookkeeping",
            "verdict": "not identical",
        },
        "bag_construction": {
            "local": "on-the-fly random bags in old CIFAR/MNIST scripts; fixed-bag rebuttal script newly added",
            "official_code": "official loaders include random, cluster, and alphafirst bag construction modes",
            "verdict": "not identical",
        },
    },
}


def main() -> None:
    out_dir = Path("results/rebuttal")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pvc_verdict.json").write_text(json.dumps(VERDICT, indent=2, sort_keys=True))
    md = [
        "# PVC Audit Verdict",
        "",
        "**Verdict: C.** The existing method named `PVC` is a locally implemented approximation of the shared classwise Poisson-binomial count-likelihood component. It is not the complete official LLP-PVC method.",
        "",
        "Existing rows should be labeled as `FS-Conv classwise count likelihood` or `LLP-PVC count-likelihood component`, not as full official LLP-PVC.",
        "",
        "## Numerical Verification",
        "",
        "This is a special-case relationship check, not a competitive baseline result.",
        "",
        "| Quantity | Max absolute difference |",
        "| --- | ---: |",
        "| Forward loss | 4.44e-16 |",
        "| Per-class PMF | 2.22e-16 |",
        "| Logit gradient | 2.12e-16 |",
        "",
        "## Key Differences",
        "",
        "- Output parameterization: local code uses `softmax`; official code contains a `sigmoid` path for the count loss and `softmax` for evaluation/pseudo-label logic.",
        "- Complete loss/procedure: local code implements the classwise count likelihood only; official code includes broader training procedure details.",
        "- Prediction/checkpointing: local code predicts by `argmax softmax` and checkpoints by observed histogram metric; official code has its own evaluation/EMA bookkeeping.",
        "- Bag construction: local old runs used on-the-fly random bags; official code supports multiple bag construction strategies.",
    ]
    (out_dir / "final_equivalence.md").write_text("\n".join(md) + "\n")


if __name__ == "__main__":
    main()
