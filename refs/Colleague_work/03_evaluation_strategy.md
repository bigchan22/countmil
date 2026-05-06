# C. 평가 전략 — Cross-domain validation 채우기

작성일: 2026-05-03

순수 방법론 논문이 되려면 **표준 벤치마크 + 1~2개 실 도메인** 평가가 필수.
단일 응용에 의존하지 않고 generalizable methodology임을 보여야 함.

---

## 1. Shukla 2023과 head-to-head 가능한 표준 벤치마크

Shukla 논문 setup을 그대로 따라 직접 비교 표를 만들 수 있음.

### LLP
- **Adult** (8192 train), **Magic** (6144 train)
- bag size $\{8, 32, 128, 512\}$
- label proportion 분포: $[0, 1/2]$, $[1/2, 1]$, $[0, 1]$
- baseline: PL (Tsai 2020), LMMCM, Shukla CL
- metric: test AUC

### MIL
- **MNIST-MIL** (Ilse 2018): bag-level + instance-level
  - bag size $\sim \mathcal N(10, 2)$, $\mathcal N(50, 10)$, $\mathcal N(100, 20)$
  - #training bags $\in \{50, 100, ..., 500\}$, low-bag $\{10, 20, 30, 40\}$
- **Colon Cancer**: 100 H&E images, instance dependence violated → robustness 검증
- baseline: Attention / Gated-Attention (Ilse 2018), Instance-Max, Shukla CL
- metric: test AUC + instance-level acc/AUC

### PU
- **Binarized MNIST**, **MNIST17**, **Binarized CIFAR**, **CIFAR Cat-vs-Dog**
- baseline: CVIR (Garg 2021), nnPU (Kiryo 2017), uPU (du Plessis 2015), Shukla CL/CL-expect
- metric: test accuracy on unlabeled

### 권장 우선순위 (시간 제한 시)
1. **MNIST-MIL** — 가장 빠름 (~수 시간), Shukla과 직접 비교 가능
2. **MNIST-sum (multi-class, A1 검증)** — A1 main contribution 직접 검증
3. **LLP Adult** — 빠름, table 1개 추가
4. **PU Binarized MNIST** — selectivity로 stretch goal

---

## 2. 실 도메인 후보 (자체 contribution 차별화)

| 도메인 | bag | instance | aggregate label | 데이터 가용성 |
|---|---|---|---|---|
| 의료 이미징 (digital pathology) | WSI 패치 | 세포 핵 | 종양 세포 수 | CAMELYON, TCGA — 공개 |
| 분자 그래프 | 분자 | atom / functional group | functional group 개수 | ZINC, QM9 — 공개 |
| 천체 사진 | sky 패치 | 별 / 은하 | object count | SDSS, Galaxy Zoo — 공개 |
| 입자 물리 | event | particle track | 총 charge / energy bin | ATLAS Open Data — 공개 |
| 단세포 시퀀싱 | tissue | cell type | cell type 비율 (deconvolution) | scRNA-seq atlases — 공개 |

### 권장 1순위: **digital pathology cell counting**
- 데이터 풍부, MIL community에서 친숙 (Colon Cancer가 이미 Shukla/Ilse benchmark)
- 큰 bag size ($N \sim 100$~$1000$ 패치) → A2 (FFT/tree reduction) 효용 입증 가능
- "instance-level discovery"가 임상적으로 의미 있음 → 논문의 dual objective와 부합

### 2순위: **분자 functional group counting**
- multi-class counting (A1 검증에 직결)
- ZINC15 등 공개 데이터 풍부
- molecular property 예측과 연결 가능 (downstream task)

---

## 3. PMF calibration metric — 모든 method 공통 보고 (B7에서 이전 2026-05-03)

PCA가 full PMF를 내는 점을 살리려면 mode/NLL만 보고하는 Shukla 평가를 넘어
**분포 자체의 calibration**을 평가해야 함.

### 추가 보고 항목
- **ECE on count bins**: 예측 PMF의 mode confidence vs 실제 정답률.
- **Reliability diagram**: count별 confidence-accuracy 산점도.
- **Auxiliary loss option**: focal loss / label smoothing 유사 calibration penalty (선택적, 학습 시).

### 의의
"우리는 분포 자체를 잘 학습한다"는 PMF 기반 접근의 차별점을 정량 표로 살림.
Shukla은 mode/NLL만 평가하므로, 이게 PCA의 native 차별점.

### 적용 범위
모든 baseline (Shukla DP, mean pooling, ours)에 동일 metric 적용 → 비교 깔끔.
PU 도메인에서는 Shukla이 보인 binomial 분포 매칭 (`Fig PU Dist`) 시각화와 직접 비교 가능.

---

## 4. 실험 우선순위 — 병렬 트랙 구조 (validation roadmap, 2026-05-03 재구조화)

NeurIPS 2026 deadline까지 5–6주 가정. Sequential이 아닌 4개 트랙 병렬 진행.

### Track 1 — Empirical core (week 1–3, gate)

| 단계 | 실험 | 목적 | 검증 axis |
|---|---|---|---|
| 1a | MNIST-MIL binary baseline (Shukla 비교) | 기준점 | 전체 |
| 1b | MNIST-sum multi-class baseline | A1 main 검증 | A1 |
| 2  | + B1 (count-conditioned EM) | training boost 검증 | B1 |
| 3  | + B2 (bag mixup consistency) | OOD objective-level 검증 | B2 |

**Gate (week 3 끝):** B1·B2 모두 marginal NLL 대비 의미 있는 향상 ≥ 1%p? 안 되면 axis 재조정.

### Track 2 — Theory (week 1–4, 독립 진행)

| 단계 | 작업 | 산출 |
|---|---|---|
| T1 | Cancellation lemma toy 증명 ($N=2,3,4$) | 일반화 추측 |
| T2 | Risk-consistency theorem statement + 가정 명시 | 정리 명세 |
| T3 | Proof 시도 또는 강등 결정 | week 3 끝까지 결정 |

**Gate (week 3 끝):** T1이 깨끗하게 풀리지 않으면 A4를 부록 sketch로 강등.

### Track 3 — Cross-domain (week 3–5)

| 단계 | 실험 | 목적 |
|---|---|---|
| 3a | LLP Adult / Magic | LLP paradigm |
| 3b | PU Binarized MNIST | PU paradigm |
| 3c | Digital pathology Colon Cancer (stretch) | 실 도메인 scientific impact |

### Track 4 — Large-bag enabler (week 4–5, optional)

| 단계 | 실험 | 목적 |
|---|---|---|
| 4a | FFT-tree gradient stability sanity check | A2 viability |
| 4b | A1 multi-class on large bag (pathology 또는 합성) | A1 + A2 결합 |

### 병합 ablation (week 5–6)

- B6 (constraint projection at inference) 모든 method에 균일 적용 → 비교 표.
- B3 (entropy/cardinality) ablation 한 줄 — 강등 후에도 sanity check.
- 모든 method에 PMF calibration metric (§3) 동시 보고.

### 트랙 의존성

각 트랙 독립 → 한 트랙이 막혀도 다른 트랙이 진행 가능.
**Track 1 + Track 3가 minimum viable submission**: A1 main + B1/B2 + cross-paradigm 검증.
Track 2 (A4) 통과 시 thickness 추가, Track 4는 large-bag domain stretch.
