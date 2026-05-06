# B. 약한 감독 학습 강화 axes

작성일: 2026-05-03

aggregate-only supervision의 본질적 어려움: **under-determination**.
같은 $Y$를 만드는 $\{z_u\}$ 구성이 다수 → shortcut 다수, OOD 실패.
이 axis들은 그 약점을 본질적으로 공격.

---

## B1. Count-conditioned EM / posterior matching [High, 가장 추천]

### 아이디어
PCA에서 활용 안 된 점: count constraint 하의 posterior $p(z_u \mid \sum_i z_i = Y)$ 를
**정확히 계산 가능**.

#### 수식
- Forward conv: $P(\sum z_i = Y)$ (이미 갖고 있음)
- Leave-one-out conv: $P_{i \neq u}(\sum z_i = s)$ (인스턴스 $u$ 제외)
- Posterior:
  $$
  p(z_u = 1 \mid \textstyle\sum z_i = Y) = \frac{p_u \cdot P_{i \neq u}(\sum = Y - 1)}{P(\sum = Y)}.
  $$

#### EM 절차
- **E-step**: 현재 모델로 위 posterior 계산 → soft pseudo-label $q_u$
- **M-step**: cross-entropy $-\sum_u q_u \log p_u + (1-q_u)\log(1-p_u)$ 로 모델 업데이트
- 또는 한 번에: NLL + $\lambda \cdot \mathrm{KL}(q \| p)$ joint loss

### 왜 강력한가
- Expectation surrogate (mean pooling) 또는 Shukla의 marginal NLL 보다 framing 측면에서
  더 명시적인 instance-level supervision 합성
- 라벨 없이 인스턴스-level supervision을 합법적으로 합성
- Free-energy bound로 이론 backing 가능 ($\log P(Y) \geq \mathbb E_q [\log p(z, Y)] + H(q)$)

### 구현
Conv 한 번 더, leave-one-out trick. PyTorch 30~50줄.
- 효율: leave-one-out은 $O(N)$번 돌리지 않고
  forward chain 한 번 + backward chain 한 번 (= prefix·suffix product)으로 **$O(N)$ 전체** 가능.
- 즉 추가 비용은 forward와 동일 order.

### 검증 스케치
- **MNIST-MIL** (Ilse 2018, Shukla Table 4): instance-level discovery accuracy
  - baseline: plain conv NLL (= ours 현재)
  - ours+EM: posterior matching 추가
  - Shukla DP NLL과도 직접 비교 (instance-level은 Shukla도 평가)
- **LLP Adult / Magic**: bag size 8, 32, 128. EM이 큰 bag에서 더 큰 이득을 줄 것으로 예상.

### Open question
- EM이 local optimum에 갇힐 수 있음. Warm-start 필요.
  현재 논문의 temperature annealing이 그 역할을 할지, 별도 schedule 필요할지.
- **회의적 관점**: marginal NLL gradient는 chain rule로 같은 posterior를 이미 포함.
  B1이 추가 정보를 주는 게 아니라 gradient 경로를 바꿈. 실제 향상이 있는지는
  ablation 필수 (week 1 첫 실험 우선순위).

---

## B2. Bag mixup / additive consistency [High]

### 아이디어
Disjoint bag $\mathcal B_1, \mathcal B_2$를 합치면 합 분포는 정확히 두 분포의 conv:
$$
P_{\mathcal B_1 \sqcup \mathcal B_2} = P_{\mathcal B_1} * P_{\mathcal B_2}.
$$
이는 모델이 만족해야 할 **정확한 algebraic invariant**.

#### Loss
$$
\mathcal L_{\mathrm{mix}} = \mathrm{KL}(\hat P_{\mathcal B_1 \sqcup \mathcal B_2} \,\|\, \hat P_{\mathcal B_1} * \hat P_{\mathcal B_2}).
$$

### 무엇을 풀어주는가
- 큰 bag과 두 작은 bag의 일관된 예측 → **bag size 일반화**
  (Purely-Relative Transformer가 architecture로 풀려는 OOD 문제를
  **objective 차원에서** 풀어줌)
- 다른 size에 대해 self-supervised signal 생성 (라벨 없이도 추가 학습 신호)
- 작은 bag→큰 bag로 학습 신호를 전달 (curriculum 효과 자동)

### 검증 스케치
- **Bag size extrapolation**: $N_{\mathrm{train}} \in \{10\}$, $N_{\mathrm{test}} \in \{20, 50, 100\}$
- vanilla NLL vs +B2 mixup
- bag-level extrapolation accuracy + instance-level recovery 별도 보고
  (KL이 aggregate distribution에 작용 → bag-level은 향상돼도 instance-level은
  약할 수 있음. 두 차원 모두 모니터링 필수.)
- 핵심 질문: Architecture에 의존하지 않고 objective 차원에서만 OOD bag size 일반화가
  가능한가?

### Open question
- Disjoint bag을 어디서 얻나? 현재 데이터에서 인위적으로 split 가능 (bag을 둘로 나누고 합 label만 관측되었다고 가정).
- Overlap이 있는 경우는? — A3 (cross-bag joint conv)와 직접 연결됨.

---

## B3. Entropy + sparsity / cardinality regularization [Low, ablation only — 강등 2026-05-03]

### 아이디어
이미 entropy regularizer 있음 (`CellSelection.tex:631`, coefficient 0.01). 추가 후보:
- **Cardinality prior**: $\big|\sum_u p_u - \mathbb E[Y]\big|^2$ 또는 KL 페널티
- **Sharpness**: $\sum_u p_u(1-p_u)$ 페널티 → 거의 binary

### 강등 사유
- **Cardinality prior**: bag-level NLL이 이미 분포 전체를 매칭 → expectation 매칭은
  strict subset. 추가 항으로서 실질적 information 0.
- **Sharpness**: 기존 entropy regularizer와 거의 동일.
- B1 (count-conditioned EM)이 작동하면 자연스럽게 sharp해짐 → 중복.

### 처리
독립 contribution 아님. ablation table 한 줄로만 보고 (각 항 on/off, 계수 sweep).
본문 별도 절 불필요.

---

## B4. Inert-instance augmentation / permutation invariance [Drop — 부록 robustness ablation only, 강등 2026-05-03]

### 강등 사유
- "Easy negatives 추가"는 generic data augmentation이지 count-based WSL specific 아님.
- "Permutation noise"는 set 기반 architecture (DeepSets, Transformer)에서 architecture-level로 처리됨 → 중복.
- "Drop a known-zero"는 어떤 인스턴스가 zero인지 모른다는 setup 전제와 모순.

### 처리
본문 contribution에서 제외. 부록 robustness ablation 한 항목으로만 (또는 완전 drop).

### 아이디어 (참고용 보존)
Bag에 명백한 negative 인스턴스를 추가 (count 불변).
모델이 그 인스턴스에 0 할당해야 — 직접 강제.

#### Variants
- **Add easy negatives**: 도메인에 따라 명백히 0인 instance를 inject
- **Permutation noise**: instance 순서 무작위화 (transformer는 자동 invariant이지만 noise+consistency 강제는 별개 효과)
- **Drop a known-zero**: 학습 중 일부 인스턴스를 randomly drop, count는 유지 → robustness

### 검증
augmentation on/off 비교, instance-level accuracy. 값싼 ablation이라 여러 axis와 함께 묶기.

---

## B5. Curriculum on bag size [Low, scheduling]

### 아이디어
작은 bag일수록 supervision 강함 (극단: $N=1$이면 fully supervised).
- 단계적 $N$ 증가
- 또는 self-distillation: 작은 bag에서 학습된 모델로 큰 bag의 pseudo-instance-label 생성

### 검증
fixed $N$ vs curriculum schedule (linear, log, step).
Shukla 결과 (`:912`–`:916`)에서도 작은 #training bags일수록 차이가 큼 — 동일 패턴 기대.

---

## B6. Constraint projection at inference [Low, post-hoc]

### 아이디어
test 시 $Y$가 관측 가능하면 raw $\{p_u\}$를 **count-constrained simplex에 투영**:
- argmax over $\{z_u\}$ s.t. $\sum z_u = Y$ (DP 또는 IP로 exact)
- 또는 conv가 주는 posterior로 MAP 추론

### 효과
학습이 부정확해도 inference에서 constraint 강제 → robustness↑.
**모든 baseline에 일괄 적용 가능 → 도구로서의 가치가 method로서의 가치보다 큼.**

### 검증
post-hoc trick. 모든 method (Shukla DP, mean pooling, ours)에 균일 적용 → 비교 표 깔끔.
ablation 쉬움.

---

## B7. Calibration of predicted PMF — **이전: 03 §3 평가 metric으로 (2026-05-03)**

학습 axis가 아니라 **모든 method 공통 보고 metric**. 자세한 내용은
`03_evaluation_strategy.md` §3 (PMF calibration metric) 참조.

학습 시점에서의 auxiliary calibration loss는 옵션이지만, 그 자체로 contribution은 아님.

---

## 결합 시너지 (어느 axis끼리 묶으면 강한가)

- **B1 + B2**: 둘 다 라벨 없이 더 많은 supervision 합성 — 누적 효과 가능성. 단, 함께 쓰면 학습 불안정 가능.
- **B1 + A1**: multi-class에서 conditional posterior 계산도 길이 $K{+}1$ atoms로 자연 확장.
- **B2 + A2**: tree reduction 자체가 bag을 합치는 algebra와 동형. 큰 bag에서 둘 다 동시에 의미.
- **B6 + 모두**: post-hoc이라 어떤 학습법과도 결합 가능, 깔끔한 추가 ablation.
