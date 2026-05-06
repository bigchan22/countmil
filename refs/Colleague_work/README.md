# 연구 노트 — PCA 순수 방법론 논문으로의 피벗

## 목적

CellSelection.tex의 application case study를 제거하고
**Probabilistic Convolutional Aggregator (PCA)** 를 중심에 둔 순수 방법론 논문으로
재작성하기 전에, 검증해야 할 연구 방향을 정리한 노트.
바로 본문을 다시 쓰기보다 여기 정리된 axis들을 하나씩 실험으로 점검한 뒤
contribution을 확정하는 것이 목적.

## 파일 인덱스

| 파일 | 내용 |
|---|---|
| `01_pca_strengthening.md` | PCA 자체를 알고리즘적으로 강화하는 axes A1–A4 |
| `02_training_strengthening.md` | 약한 감독 학습 자체를 강화하는 axes B1–B7 |
| `03_evaluation_strategy.md` | Cross-domain 검증 전략 (표준 benchmark + 실 도메인 + roadmap) |

---

## 1. 전략적 함의 — 왜 단순 conv-reform만으로는 부족한가

Shukla 2023 ("A Unified Approach to Count-Based Weakly Supervised Learning",
NeurIPS 2023) 대비 차별점 (2026-05-03 갱신, A1 main 승격):

- (a) **DP → conv 재구성**: 동일 PMF, 동일 점근 복잡도. 단독으로는 cosmetic.
- (a*) **Atomic PMF 일반성 — A1 [Main]**: conv view는 kernel 모양에 무관 →
  multi-class / signed / mixture로 자연스럽게 확장. Shukla DP는 점화식을 매번 다시 짜야 함.
  **이게 reviewer-legible한 conceptual 차별점**.
- (c) **GPU batching**: 엔지니어링.

이전 노트의 "(b) Signed extension"은 (a*)의 special case로 흡수
(application-specific motivation 제거 후 standalone contribution으로는 약함).

**A1을 main으로 두고, B1·B2·A4·cross-domain validation으로 thickness 확보.**
아래 priority 표가 그 후보들의 압축 요약.

\textemdash

## 2. 우선순위 — 임팩트 vs 구현 부담

| 순위 | 항목 | 위치 | 임팩트 | 구현 부담 |
|---|---|---|---|---|
| **Main (verified)** | **A1 Atomic PMF 일반성 (multi-class, signed, mixture)** — `A1_CONFIRMED` (2026-05-06) | algorithm | 매우 큼 | 작음 |
| Low | ~~B1 Count-conditioned EM / posterior matching~~ — week-1 demoted | training | 미미 (verified) | 보통 |
| High | B2 Bag mixup / additive consistency | training | 큼 | 작음 |
| High | A4 Signed counting risk-consistency 정리 (contingent) | theory | 큼 | 큼 |
| High | Cross-domain validation (표준 MIL/LLP/PU) | eval | 큼 | 큼 |
| Mid  | A2 FFT/tree reduction (큰 bag wall-clock) | algorithm | 큼 | 보통 |
| Low  | A3 Hierarchical / cross-bag overlap | algorithm | 보통 | 보통 |
| Low  | B3 Entropy / cardinality (강등) | training | 작음 | 작음 |
| Low  | B5 Curriculum on bag size | training | 작음 | 작음 |
| Low  | B6 Constraint projection at inference | training | 보통 | 작음 |
| Drop | B4 Inert-instance augmentation | training | 미미 | 작음 |
| Eval | B7 PMF calibration → 03 §3로 이전 | eval metric | n/a | n/a |

## 3. 결정 사항

- **Purely-Relative Transformer**: 삭제 — PCA 단일 집중. 기존 Section II 재배치 필요.
- **A1 main 승격 (2026-05-03)**: Atomic PMF 일반성을 main contribution으로.
  "signed extension"은 standalone에서 제거하고 A1의 special case로 흡수.
  Conv view의 진짜 의미를 한 줄로 잡을 수 있게 함.
- **B3/B4/B7 강등 (2026-05-03)**:
  - B3 Mid → Low (NLL이 이미 expectation 매칭, B1과 redundant).
  - B4 Drop (generic augmentation, count-WSL specific 아님).
  - B7은 학습 axis가 아니라 평가 metric — `03_evaluation_strategy.md` §3로 이전.
- **Roadmap 재구조화 (2026-05-03)**: Sequential 9단계 → 4개 병렬 트랙 (`03 §4`).
- **B1 week-1 verdict (2026-05-04)**: **Demoted to Low**. MNIST-MIL N∈{10,50,100}/200ep + LLP Adult N∈{32,128}/200ep, 5 seeds each, paired NLL vs EM(λ=1). Δ@N=50(MNIST)= +0.00 pp (p=0.816); Δ@N=128(LLP)= +0.11 pp (p=0.501) — 양쪽 모두 1pp threshold 미달. §3.5의 "NLL gradient already encodes the posterior" 관찰이 경험적으로 확증됨. Variance-reduction (H3)은 MNIST에서만 PASS (ratio 0.65), LLP에선 FAIL (0.73) — boost-axis로도 약함. 상세: `results/mnist_mil/summary.md`, `results/llp_adult/summary.md`. Track 1 step 3 (B2 bag mixup) 가 다음.
- **A1 final verdict (2026-05-06)**: **`A1_CONFIRMED`** ⭐ — 4 dataset 100 runs across MNIST-sum (cap=100, 5 seeds × 6 method × 80ep) + MNIST-signed (5 seeds × 2 method × 80ep) + SVHN-sum (5 seeds × 6 method × 80ep, lr=1e-4 + aug) + UltraMNIST (5 seeds × 6 method × 60ep). H1 2/3 PASS (svhn p=0.001 Δ=+0.969 nats, ultramnist p=0.005 Δ=+2.070 nats), **H2 5/5 PASS (relative threshold)**, H4 3/3 PASS clean (PCA dominates acc + MAE on every dataset), H3 1/3 PASS (ultramnist first H3 win, ratio 0.62). Atomic PMF generality empirically 확립; atom support change만으로 binary→multi-class→signed 통합 작동. 상세: `docs/notes/2026-05-06-a1-final-verdict.md`, `results/summary_overall.md`. **Future limitations**: H3 calibration tradeoff (post-hoc calibration B3 axis), MNIST-sum H1 p=0.190 (proper validation split 필요), 2b CLT Gaussian variance collapse on synthetic data (baseline robustness 이슈).
- **Target venue**: NeurIPS 2026 — A1 main + B2/A4 + cross-domain의 두꺼운 논문 방향. (B1 제외)
