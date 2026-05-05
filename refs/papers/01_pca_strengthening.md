# A. PCA 알고리즘 강화 axes

작성일: 2026-05-03

각 axis마다: (1) 아이디어 (2) 무엇이 풀리는가 (3) 차별점 (4) 검증 스케치.

---

## A1. Bernoulli 너머의 atomic PMF — conv view의 진짜 일반성 [Main]

### 아이디어
Shukla의 DP 점화는 $z_i \in \{0,1\}$에 hard-coded.
Conv는 atomic PMF의 **모양에 무관** — kernel 길이만 바꾸면 됨.

### 무엇이 풀리는가
- **Multi-class counting (main pitch)**: $z_i \in \{0,1,\dots,K\}$. 각 $\mathbf d_i$ 길이 $K{+}1$.
  Shukla이 future work로 미룬 multi-class를 자연스럽게 해결 (Shukla `:993`).
- **Signed counting (special case)**: $s_i \in \{-1, +1\}$, atom support $\{-1, 0\}$ 또는 $\{0, +1\}$ (길이 3).
- **Multiplicity counting**: $z_i \in \{0, m_i\}$. Multi-class의 trivial reparametrization — 부록 한 단락만.
- **Mixture atoms (future work)**: $z_i$가 Bernoulli×Gaussian ("selected then noised").
  hybrid PMF/PDF conv. discretization·gradient 안정성 미해결 — 본문 제외, future work로 명시.

### 왜 차별 포인트인가
DP는 점화식을 매번 다시 짜야 하지만 conv view는 **kernel 길이만 바꾸면 끝**.
이게 reviewer에게 가장 명확한 conceptual 차별점.

### 검증 스케치
- **Toy 1: multi-class counting on MNIST (MNIST-sum).**
  bag = MNIST 숫자 10장, label = bag 안 숫자들의 합 (0~90).
  - baseline 1: Shukla DP를 multi-class로 일반화한 변종 (직접 구현)
  - baseline 2: 단순 mean pooling regression
  - ours: $\mathbf d_i$ 길이 10인 categorical conv
  - metric: bag-level top-1 accuracy, NLL, calibration ECE
- **Toy 2: signed multi-class.** 같은 setup이지만 일부 인스턴스에 sign flip.
  signed extension의 일반성을 A1의 special case로 보임.
- **Open question**: multi-class에서 categorical은 ordinal인가 nominal인가?
  $\{0,1,2\}$가 합산되면 ordinal (count) — 적합. nominal multi-class는 conv 적용 안 됨.
  scope 제한 필요.

---

## A2. FFT / tree reduction — 큰 bag에서의 점근 복잡도 개선 [Mid]

### 아이디어
Conv의 **결합법칙** → tree-reduction 또는 single FFT lift.
- Naive sequential conv: $\mathcal O(NS)$ (Shukla DP와 동일)
- Balanced tree reduction: 각 단계 length 두 배, $\log N$ 단계 → $\mathcal O(S \log S \cdot \log N)$ FFT-based, 또는 $\mathcal O(S^2 \log N / \log S)$ 직접
- Single FFT lift: 모든 $\mathbf d_i$를 길이 $S$ Fourier 도메인으로 → pointwise multiply → IFFT once. $\mathcal O(NS \log S)$.

DP는 본질적으로 sequential이라 이 점근 이득 **불가능**.

### 검증 스케치
- 합성 bag with $N \in \{10, 100, 1000, 10000\}$, support $S = N$.
- 비교 대상: naive DP / naive conv chain / tree-reduced conv / FFT chain
- 측정: forward + backward wall-clock, peak memory, gradient quality
- 그림: log-log plot, "큰 bag 영역에서만 의미 있다"는 점을 명확히

### Open question
- Backward (gradient)는 FFT chain에서도 well-conditioned인가?
  Fourier domain에서 작은 magnitude 값이 누적되면 numerical instability 가능 — 실험 필요.
  이 sanity check 통과가 A2 viability의 gate.
- 일반적인 MIL/LLP setting에서 $N$이 그렇게 큰가?
  Shukla bag size 10–500. digital pathology (WSI 패치 100~1000), 입자 물리 event,
  단세포 시퀀싱 등 큰-$N$ 도메인이 진짜 검증대.
- A1 multi-class와의 결합: $N=100$, $K=10$이면 PMF support $\sim 1000$ → FFT가
  의미 생기는 영역. A1 main 검증에 동반되어야 가치가 살아남.

---

## A3. Hierarchical / cross-bag aggregation [Low]

### 아이디어
같은 인스턴스가 여러 bag에 등장하는 자연 설정 (single-cell deconvolution에서
같은 cell type이 여러 tissue에 등장, overlapping astronomy patches, 여러 assay에
공유되는 분자 등):
- 동일 인스턴스의 $p_u$ 일관성 강제 → identifiability 확보
- bag overlap 시 $P(Y_1, Y_2)$는 marginal product 아님 → **2D conv**로 joint distribution 모델링
- DP로는 cross-bag joint를 다루는 일반 알고리즘이 없음

### 검증 스케치
- **Synthetic LLP 변종**: 인스턴스 풀 1000개에서 두 overlapping bag 샘플 (overlap rate 50%).
- baseline: 각 bag을 독립으로 가정하는 Shukla-style
- ours: 2D conv로 joint $P(Y_1, Y_2)$
- metric: instance-level recovery rate, joint NLL

### Note
표준 MIL/LLP/PU benchmark는 instance-bag 1:1 → cross-bag 자연 setup이 아니라
합성 데이터 의존이 약점. **A4 open question (EM이 cross-bag information으로
identifiability를 회복하는가)와 결합되어야 의미가 살아남.** 단독으로는 priority가 낮음.

---

## A4. Signed counting risk-consistency [High, theory — contingent on (4.1) cancellation lemma 통과]

### 아이디어
Shukla은 LLP에 대해서만 risk-consistency 증명 (Kobayashi 2022 reduction 사용,
Shukla `:1006`–`:1071`).
signed counting $Y = \sum s_i z_i$ 케이스로 확장 — 새 정리.

### 다뤄야 할 점
- **Cancellation에 의한 identifiability loss를 정량화하는 lemma**:
  $Y=0$이 "아무것도 안 뽑음"과 "$+$, $-$ offset"으로 동시 설명 가능.
  signed bag에서 risk-consistency가 성립하기 위한 가정 (e.g., bag 내 sign 분포 조건).
- consistency rate ($m \to \infty$에서 $1/\sqrt m$).
- 어떤 가정 하에 unbiased risk estimator가 되는지.

### 검증 스케치
- 이론 결과는 증명. 실험으로는 finite-sample regret 곡선:
  bag 수 $m \in \{10^2, 10^3, 10^4, 10^5\}$ 에 대해 test error → 0 with rate $\sim 1/\sqrt m$.
- 가정이 깨질 때 (예: 한쪽 sign이 dominant) 어떻게 degrade되는지 plot.

### Open question
- 표준 MIL/PU 셋업 (sign 분포가 비균형이거나 instance가 여러 bag에 reuse됨)에서
  identifiability 가정이 자연스럽게 만족되는가?
- Identifiability가 깨질 때 EM (B1)을 결합하면 회복되는가? (이게 사실이면 강한 메시지)
