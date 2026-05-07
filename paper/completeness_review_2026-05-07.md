# CountMIL 논문 완결성 종합 검토

**작성일:** 2026-05-07
**검토 대상:** `paper/main.tex` + `paper/sections/01–05` + `paper/sections/appendix.tex` + `paper/experiment_report_2026-05-06.md`
**제출 목표:** NeurIPS 2026
**검토 범위:** 논문의 자체 완결성(주장–증거 정합성, 누락, 설득력). NeurIPS 양식/체크리스트 가이드라인은 별도 문서에서 다룸.

---

## Executive Summary

골격은 견고하지만, **"약속된 결과의 부재"가 가장 큰 약점**이다. Introduction에서 contribution으로 명시한 항목 중 하나(count-conditioned posterior marginal로 instance recovery)는 본문 experiments 섹션에 결과 표가 전혀 없다. 또한 ICML 2026 reject 사유 중 하나였던 "Attention-MIL 류 베이스라인 부족"은 데이터(`experiment_report` FashionMNIST 결과)는 이미 가지고 있지만 main paper에 반영되지 않았다. Empirical 강점(digit-sum exact vs MSE, CIFAR-10 pretrained이 Ma reference를 상회, FFT-tree 8500× 속도, fixed-instance-budget caveat)은 강하므로, 이미 가진 결과를 페이퍼에 정확히 반영하는 것만으로 상당한 보강이 가능하다.

---

## A. 페이퍼 골격 점검

### 분량 (TeX 소스 기준)

| 파일 | 줄수 | 평가 |
|---|---:|---|
| `main.tex` | 85 | 셸 + abstract + include |
| `sections/01_introduction.tex` | 34 | 5개 contribution, 포지셔닝 규칙 잘 지킴 |
| `sections/02_related_work.tex` | 31 | 28개 인용. count-loss + LLP 두 축 |
| `sections/03_problem_setup.tex` | 100 | Proposition 1 (finite-support convolution law) |
| `sections/03_method.tex` | 255 | 알고리즘 박스 + 4개 special case + backend 표 |
| `sections/04_experiments.tex` | 215 | 5개 영역 |
| `sections/05_discussion.tex` | 23 | **너무 짧음** |
| `sections/appendix.tex` | 44 | **너무 짧음** |

빌드 결과 본문 페이지: 8 페이지 (NeurIPS 2026의 9 페이지 한도 안). References + Appendix 포함 11 페이지.

### 구조 적절성

- Abstract — 6문장, 모든 핵심 영역(binary/signed/sum/posterior/histogram + CIFAR-10 + fixed-budget caveat) 언급. ✓
- Section 1 (Introduction) — 한 페이지짜리 적절한 길이. Contribution 5개 명시. ✓
- Section 2 (Related Work) — count-loss + LLP/PVC 두 축. attention-MIL 대비 부족.
- Section 3 (Problem Setup) — 100줄. Proposition 1과 task-kernel 표.
- Section 3 (Method) — 255줄. 알고리즘 박스, 4개 special case, posterior 수식, backend 표. **포함된 내용이 많아 균형이 좋음.**
- Section 4 (Experiments) — 5 subsection. Posterior 결과 표 부재. Attention MIL 비교 row 부재.
- Section 5 (Discussion) — 23줄. **너무 짧다.** broader impact, future work 부재.
- Appendix — 44줄, 3개 subsection. **C "Additional Tables"는 표 없이 한 문장만**.

---

## B. 핵심 주장 vs 실험 결과 정합성

### Intro contribution 5개 vs 실험 증거 매핑

| Contribution (intro 24–31) | Method 섹션 위치 | Experiments 표/Figure | 강도 |
|---|---|---|:-:|
| (a) finite-support convolution view 자체 | Prop 1, Eq 7, 알고리즘 1 | Backends 표 | 이론 명확 |
| (b) backbone-agnostic likelihood가 signed/digit-sum/histogram에 확장 | Method 3.3 special cases | Tables binary/digit-sum/signed/histogram | 강함 |
| (c) count-conditioned posterior marginal | Method 3.4 + Appendix A | **표 없음** | **누락** |
| (d) 실용적 convolutional backend (grouped + FFT-tree) | Method 3.5 + 3.6 | Runtime 표 (FFT-tree만) | 부분 |
| (e) hidden-instance recovery 실험 | — | binary/signed/sum/CIFAR rows | 강함 |

→ (c)는 contribution 항목으로 명시되어 있는데 본문에 결과가 0개. Appendix C에 한 줄짜리 negative result 만 있음.

### 결과 표별 강도

| Table / Figure | 핵심 결과 | 강도 |
|---|---|:-:|
| Table `binary-countmil` (4 row) | MNIST n=10/50, train 1k/5k, count MAE 0.090–0.686, Hidden acc 0.987–0.991 | 적절한 sanity check |
| Table `digit-sum` (8 + 3 row) | n=10/1000: exact 0.612 vs MSE 2.535 (acc 0.982 vs 0.226). UltraMNIST 5 seeds 0.341±0.038. SVHN pretrained 0.952 instance acc | **매우 강함, 핵심 입증** |
| Figure `ordinal-sum-instance-accuracy` | exact vs MSE 시각화 | 보조 |
| Table `signed` (4 row) | n=10/1000 signed MAE 0.095, Hidden acc 0.990. cancellation 모드 row 없음 | 부분 — cancellation 모드 row 추가 권고 |
| Table `histogram-llp-pvc` (12 row) | MNIST PVC vs MSE (n=50/1000: 98.5% vs 11.6%), CIFAR-10 from-scratch 4 row + pretrained 4 row, fixed-budget 2 row | **강함, 핵심 입증** |
| Table `runtime` (4 row) | n=64–512, CPU DP vs GPU FFT-tree, 8500–40000× speedup, max abs error 1e-7 | **매우 강함** |

### Method가 정의했지만 Experiments에 없는 것

- Posterior marginal 결과 (Method Eq 17–18 정의됨, 결과 0)
- Grouped signed conv1d 단독 검증 결과 (백엔드만 method 3.6 paragraph에서 언급)
- Multiplicity counts (problem-setup 표 last row에 정의되었지만 실험 없음)

---

## C. ICML 2026 리뷰 약점이 반복되는가

ICML 2026 reject (2026-05-01) 핵심 사유:

1. (a) Shukla 등 기존 count-loss와의 직접 비교 부족
2. (b) Attention-MIL 등 베이스라인 부족
3. (c) 일반성/표준 벤치마크 부족
4. (d) 코드/재현성

현재 NeurIPS 페이퍼 상태:

| 사유 | 현재 상태 |
|:-:|---|
| (c) | **해결됨.** MNIST/FashionMNIST/SVHN/UltraMNIST/CIFAR-10/CIFAR-100 + `src/` 모듈 구조 |
| (d) | **해결됨.** `src/` + `scripts/` + `tests/` 트리 |
| (a) | **부분 해결.** 본문이 "Bernoulli count likelihood = Shukla DP"임을 수식으로 인정. 하지만 결과 표에 별도 "Shukla DP" row가 없어, 표만 보는 reviewer는 베이스라인이 빠진 것으로 인식 가능. **동일 setting에서 DP-구현과 conv-구현을 별도 row로 보여주는 미니 표 1개로 해결됨.** |
| (b) | **여전히 미흡.** `experiment_report` FashionMNIST 섹션(L235–245)에 binary CountMIL conv vs attention vs gated-attention 비교 데이터 존재(Instance AUC: conv 0.9973, attention 0.7034, gated 0.7221). 이것이 main paper main table에 들어가지 않은 것이 가장 큰 손해. |

---

## D. Paper에 빠진 "이미 보유한" 강한 결과들

`experiment_report_2026-05-06.md`에는 있지만 `paper/sections/`의 어디에도 들어가지 않은 결과:

### D.1 FashionMNIST conv vs attention/gated-attention (binary)

| Bag size | Method | Bag AUC | Bag acc. | Count MAE | Instance AUC |
|---:|---|---:|---:|---:|---:|
| 10 | conv | 0.9988 | 0.9605 | 0.183 | **0.9973** |
| 10 | attention | 0.9980 | 0.9725 | – | 0.7034 |
| 10 | gated attention | 0.9979 | 0.9775 | – | 0.7221 |
| 50 | conv | 0.9987 | 0.7720 | 0.886 | **0.9980** |
| 50 | attention | 0.9983 | 0.9880 | – | 0.7434 |
| 50 | gated attention | 0.9984 | 0.9850 | – | 0.7146 |

→ Bag-level metric은 attention이 더 나을 수 있지만 instance-level recovery는 conv가 압도적. ICML reviewer가 정확히 지적했던 attention 비교 약점을 직접 닫는 결과인데, paper에는 부재.

### D.2 MNIST 히스토그램 CE/KL row (현재 paper는 MSE만 표시)

| Bag/train | Objective | Count MAE | Prop. MAE | Digit acc. | Hist exact |
|---|---|---:|---:|---:|---:|
| 10/1000 | CE | 0.0804 | 0.00807 | 0.9725 | – |
| 10/1000 | KL | 0.0804 | 0.00807 | 0.9725 | – |
| 10/1000 | MSE | 0.1814 | 0.01846 | 0.9548 | – |
| 10/1000 | PVC | 0.0320 | 0.00316 | 0.9827 | 0.838 |
| 50/1000 | CE | 0.3756 | 0.00768 | 0.9640 | – |
| 50/1000 | KL | 0.3756 | 0.00768 | 0.9640 | – |
| 50/1000 | MSE | 1.6825 | 0.03453 | 0.1156 | – |
| 50/1000 | PVC | 0.1370 | 0.00276 | 0.9847 | 0.436 |

→ 본문은 MSE만 비교 대상으로 두고 "CE/KL은 experiment report에 기록"이라고만 처리. **CE/KL이 강한 베이스라인이지만 PVC가 그 위에서도 더 좋음**을 직접 보여주는 것이 정직성과 임팩트 모두를 높인다.

### D.3 Digit-sum MAE/Huber 베이스라인 (현재 paper는 MSE만)

| Bag/train | Loss | Expected MAE | Rounded acc. | Rounded MAE | Digit acc. |
|---|---|---:|---:|---:|---:|
| 10/1000 | MSE | 2.535 | 0.133 | 2.527 | 0.226 |
| 10/1000 | MAE | 2.337 | 0.141 | 2.323 | 0.286 |
| 10/1000 | Huber | 2.330 | 0.136 | 2.323 | 0.278 |
| 10/5000 | MSE | 1.022 | 0.386 | 0.981 | 0.730 |
| 10/5000 | MAE | 0.430 | 0.835 | 0.412 | 0.989 |
| 10/5000 | Huber | 0.548 | 0.784 | 0.474 | 0.988 |

→ MSE만 보여주면 "MSE의 한계"로 보이지만, MAE/Huber까지 보여주면 **expected-value 매칭 자체의 한계**로 강화된다. n=10/5000에서 MAE/Huber가 따라오지만 hidden digit acc는 여전히 우리(0.992)가 더 나음.

### D.4 Signed cancellation-heavy 모드 (현재 paper는 random sign만)

| Bag/train | Signs | Instance AUC | Signed MAE | Zero-count accuracy |
|---|---|---:|---:|---:|
| 10/1000 | random | 0.995 | 0.095 | 0.933 |
| 10/1000 | cancellation | 0.998 | 0.071 | 0.965 |
| 50/1000 | random | 0.999 | 0.243 | 0.772 |
| 50/1000 | cancellation | 0.969 | 1.689 | 0.420 |

→ Method 3.3에서 "many-to-one because positive and negative contributions can cancel"이라 적었지만 main table에는 random만 있어 "진짜 ambiguous한 setting"의 결과가 없다. Cancellation-heavy를 별도 row로 두면 **signed가 진짜 novel setting이라는 주장이 데이터로 입증**된다.

### D.5 FashionMNIST robustness 추가 결과

`experiment_report` L246–262: digit/category sum, expected-sum MSE, histogram LLP, signed CountMIL random/cancellation 등 약 12 row의 robustness 결과. → atomic-support generality 주장을 강화할 보조 자료.

### D.6 Posterior marginal instance recovery

`experiment_report`에는 명시적 결과 표가 없으나, posterior 코드는 `src/countmil/posteriors.py`에 있고 prefix/suffix DP가 구현되어 있다. **간단한 mini-experiment 한 번 돌려서 "binary count training 후 posterior marginal로 hidden instance 복원" 결과 한 표를 만들면 contribution (c)를 채울 수 있다.** 만약 시간이 없다면 contribution 항목에서 "posterior marginal"을 빼거나, "an inference/diagnostic tool that complements likelihood training"으로 약화시켜야 한다.

---

## E. Novelty / 설득력 분석 (가상 reviewer 시점)

### E.1 Reviewer가 동의할 만한 부분

- **Proposition 1**은 standard probability identity지만, "atomic PMF 모양만 바꾸면 binary/signed/sum/OVR이 한 framework에 들어간다"는 unifying view 자체는 명확.
- **digit-sum exact vs MSE**의 수치 차이가 매우 크다 (n=10/1000에서 acc 0.226 vs 0.982). 단순히 "조금 좋다"가 아니라 **expected-value 매칭이 거의 무용하다**는 강한 주장이 가능.
- **CIFAR-10 pretrained이 Ma reference를 상회**한다는 결과는 strong claim. n=16 0.845 → 0.902, n=64 0.789 → 0.914.
- **Fixed-instance-budget diagnostic**으로 자기 결과를 caveat 거는 정직성은 reviewer 신뢰도를 높인다 (n16/train1k 0.902 vs n64/train250 0.803, 둘 다 16k instances/epoch).
- **FFT-tree 8500× 속도 + max abs error 1e-7**는 명확한 실용적 contribution.

### E.2 Reviewer가 의문을 가질 만한 부분

| 의문 | 현재 paper의 대응 | 충분한가 |
|---|---|:-:|
| "Atomic PMF generality라는 unifying view의 실질 이득은?" | "backbone 변경 없음, 같은 backend로 처리" 주장 | 약함 — 정량적 ablation 부재 |
| "binary는 Shukla, OVR은 LLP-PVC 일부와 동일이라 새 게 결국 signed + digit-sum 두 task인가?" | 본문이 그렇게 인정 | 정직하지만 novelty 주장이 좁아짐 |
| "CIFAR-10 n=128 pretrained가 빠진 이유는?" | `experiment_report` L402: 시간상 미완 | reviewer는 모름 — 본문에 해명 필요 |
| "Posterior marginals contribution은 결과가 어디?" | Appendix C 한 문장 | **부족** |
| "Discussion에 broader impact, ethical considerations, future work?" | 거의 없음 | 부족 |
| "Ablation (lr, batch size, augmentation 영향)?" | 없음 | 부족 |
| "FashionMNIST/UltraMNIST/SVHN이 robustness인가 cherry-pick인가?" | digit-sum 표에 통합되어 있지만 row가 적음 | 부분적 |

### E.3 Novelty 주장의 강도 (포지셔닝 규칙 vs 실험 증거)

`memory/feedback_paper_positioning.md`가 정한 4–5개 novelty:

1. **convolutional finite-support view** — Proposition 1과 backend 표로 명시. 이론 ✓, 실험적 직접 검증은 약함.
2. **signed / cancellation aggregate** — Section 3.3 + Table `signed`. **결과가 강하지만 cancellation 모드 row 부재로 약함.**
3. **ordinal finite-support sum (digit-sum)** — Section 3.3 + Table `digit-sum` + Figure. **강함.**
4. **count-conditioned posterior marginal** — Section 3.4 + Appendix A 수식. **실험 증거 부재.**
5. **GPU FFT-tree backend의 실용성** — Section 3.6 + Table `runtime`. **강함.**

→ 5개 중 (1)·(4)가 약하고 (2)는 자료가 있지만 본문 미반영, (3)·(5)만 강한 상태.

---

## F. 페이퍼가 이미 잘 하고 있는 점 (계속 유지)

- **포지셔닝 규칙 충실히 따름**: binary novelty 부정 명시, LLP-PVC 관계 명시, "intrinsic large-bag superiority" 주장 부정.
- **Fixed-instance-budget diagnostic**을 자체 caveat으로 둠 → reviewer 신뢰도 매우 ↑.
- **"Hidden-instance label은 학습에 사용되지 않음"** 반복 명시 (intro, problem setup, experiments, appendix) → setup이 명확.
- Proposition + Algorithm box + Backend table → method 섹션이 self-contained.
- abstract가 6문장 안에 모든 핵심 영역과 caveat 포함 — 한 화면 읽기에 적절.
- `experiment_report`와 paper main text 간 일관성: 적힌 수치는 모두 report로 추적 가능.
- Reproducibility: `src/countmil/`, `scripts/`, `tests/` 트리가 reviewer가 따라갈 수 있는 형태.

---

## G. 완결성 보강 우선순위

### ★★★ 시급 (페이퍼 수락 가능성에 직결)

| # | 작업 | 비용 | 영향 | 근거 |
|:-:|---|:-:|:-:|---|
| G1 | **Posterior marginal instance-recovery 표 1개** 본문 또는 appendix에 추가 — 약속한 contribution을 채워야 함. 만약 mini-experiment를 돌릴 시간이 없으면 contribution (c)를 약화·삭제 | 중간 (실험 1회) 또는 낮음 (텍스트만) | 큼 | B 표에서 (c)가 빈 자리, E.2의 첫째 의문에 직결 |
| G2 | **Attention/gated-attention MIL 비교 row** binary count main table에 통합 — D.1 데이터 그대로 사용 | 낮음 (typesetting만) | 매우 큼 | ICML reject 사유 (b) 직접 닫음 |
| G3 | **Shukla DP vs conv 미니 비교 표** — 동일 setting에서 두 row, 수치 동치 확인 | 낮음 | 큼 | ICML reject 사유 (a)에 대한 visual proof |

### ★★ 높음 (정직성·완결성)

| # | 작업 | 비용 | 영향 |
|:-:|---|:-:|:-:|
| G4 | **MNIST histogram CE/KL row** main table에 통합 — D.2 데이터 사용 | 낮음 | 중간 |
| G5 | **Digit-sum MAE/Huber row** main table에 통합 — D.3 데이터 사용 | 낮음 | 중간 |
| G6 | **Signed cancellation-heavy mode** main table에 row 추가 — D.4 데이터 사용 | 낮음 | 중간 |
| G7 | **Discussion 확장** — broader impact, future work, limitations 정리 (현재 23줄 → 50–70줄) | 낮음 | 중간 |
| G8 | **Appendix C "Additional Tables" 채우기** — FashionMNIST 풀 표(D.5), signed cancellation 풀 표, posterior 표 | 중간 | 큼 |

### ★ 보강 (시간 허락 시)

| # | 작업 | 비용 | 영향 |
|:-:|---|:-:|:-:|
| G9 | **CIFAR-10 n=128 pretrained 실험** — Ma table 직접 비교 완성 | 높음 (수일) | 중간 |
| G10 | **Ablation (lr, augmentation, bag-size variance)** | 중간 | 작음–중간 |
| G11 | **CIFAR-100 결과 위치 재검토** — 현재 본문 한 문장은 약함. appendix exploratory로 옮기거나 본문 표에 1 row 추가 (PVC vs KL vs MSE) | 낮음 | 작음 |
| G12 | **References.bib 확장** — attention-MIL 평가 논문 보강, posterior matching/EM 류 인용 강화 | 낮음 | 작음 |

---

## H. 가장 시급한 3가지 (제 추천)

만약 한정된 시간 안에 페이퍼 완결성을 가장 크게 끌어올리고 싶다면 다음 순서:

### 1. Posterior marginal contribution을 결정 (G1)

선택지:
- (a) **Mini-experiment 한 번 돌려서 표 추가**: 이미 학습된 binary count 모델에 `src/countmil/posteriors.py`의 prefix/suffix DP를 적용해 hidden instance label을 복원, instance accuracy/AUC를 측정. 1–2시간 작업.
- (b) **Contribution 항목에서 약화/삭제**: intro list (c)를 "we derive count-conditioned posterior marginals as an auxiliary inference and diagnostic tool"로 약화. 결과가 없는 것이 약속이 깨진 것보다 낫다.

→ 둘 중 어느 쪽이든 결정해야 함. 현재 상태(약속하고 결과 없음)가 가장 나쁨.

### 2. Attention MIL row 추가 (G2)

`experiment_report` D.1 데이터를 binary count main table에 4 row 추가. Instance AUC 0.997 vs 0.703 차이는 **단일 row만으로도 ICML reviewer가 지적했던 비교 약점을 직접 닫는다.** 데이터는 이미 있으므로 typesetting 작업뿐.

### 3. Discussion 확장 + Appendix C 채우기 (G7 + G8)

현재 23줄 Discussion + 한 문장 Appendix C는 reviewer에게 "마무리가 약하다"는 인상을 준다. 이미 가지고 있는 결과를 정리해서 넣기만 해도 페이퍼 완결성이 눈에 띄게 올라간다.

---

## I. 검토에 사용한 자료 목록

- `paper/main.tex`
- `paper/sections/01_introduction.tex` ~ `05_discussion.tex`
- `paper/sections/03_problem_setup.tex`
- `paper/sections/appendix.tex`
- `paper/experiment_report_2026-05-06.md`
- `AGENTS.md`
- `memory/project_overview.md`
- `memory/project_icml_rejection.md`
- `memory/feedback_paper_positioning.md`

추가 확인 권고 (이번 검토에서 깊이 보지 않은 자료):

- `paper/references.bib` — 인용 누락 점검
- `docs/PROJECT_BRIEF.md`, `docs/REMOTE_EXPERIMENTS.md` — 실험 상태 동기화
- `results/` 하위 — 본문 수치와 raw JSON/CSV 일치 여부
- `refs/notes/icml2026rebuttal.md` — ICML reviewer 코멘트 직접 매핑
