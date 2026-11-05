# MNIST adaptive end-to-end PGD

기존 PGD용 N2N 모델을 고정하고, 같은 clean 평가 이미지에서 non-adaptive/adaptive 두 종류의 PGD를 생성한다. 모델을 재학습하지 않는다.

## 서버 실행

GitHub 코드를 pull하고 기존 n2n 환경과 `.env`를 사용한다. 재학습은 수행하지 않는다. 기본 Ours는 Drive 가중치 묶음의 `checkpoints/mnist/pgd/denoiser.pt`다. 과거 재평가 방어 정확도는 91.54%, 논문값은 94.26%지만, 새 공격의 측정값과 같다고 가정하지 않는다.

```bash
conda activate n2n
source .env

# 파일·평가 분할·기본 가중치 해시 검사
python reproduction/evaluate_mnist_adaptive_pgd.py --check-only

# 실제 모델의 입력 gradient와 공격 생성 경로 소량 확인
python -u reproduction/evaluate_mnist_adaptive_pgd.py \
  --device cuda --eval-size 16 --batch-size 8

# 최종 평가 5000장: Ours부터 실행
python -u reproduction/evaluate_mnist_adaptive_pgd.py \
  --device cuda --epsilon 0.4 --alpha 0.01 --steps 40 \
  --random-start --restarts 1 --seed 2026 --eval-size 5000 --batch-size 32
```

## 과거 구현 가설로 두 공격을 함께 확인

LivDet과 같은 `legacy-reference` 프로필을 MNIST에서도 사용할 수 있다.

```bash
python -u reproduction/evaluate_mnist_adaptive_pgd.py \
  --device cuda --attack-profile legacy-reference \
  --eval-size 5000 --batch-size 1
```

epsilon.4,alpha2/255,40회,no random start,probability CE,마지막 반복으로 두 공격을 고정한다. 저장 MNIST PGD의 최대 변화량.3137255는40×2/255와 일치하지만, 이 사실만으로 과거 generator를 확정하지는 않는다. 기존standard 프로필의ε.4/α.01/random-start 결과와 구분한다.

이 프로필은 저장 공격도 같은 모델·ID로 다시 평가한다. `summary.csv`의 `archived_classifier_percent`와 `archived_defended_percent`를 통해 기존8.96%·91.54%를 확인하고, 새 `nonadaptive_pgd_percent`와 비교한다. 새 공격은float로 유지하므로 저장PNG와 정확히 동일한 입력이라는 보장은 없다. 저장 공격 결과는 adaptive combined 집계에 추가하지 않는다.

Clean-target도 함께 보려면 다음처럼 첫 번째 제안 실험에서 학습한 모델을 지정한다.

```bash
python -u reproduction/evaluate_mnist_adaptive_pgd.py \
  --clean-checkpoint ./mnist_pgd_clean_target.pt \
  --device cuda --attack-profile legacy-reference \
  --eval-size 5000 --batch-size 1
```

이 프로필의 결과가 높아져도 기존standard 공격에 대한 취약성이 개선된 것은 아니다. 모든 결과는 직접 측정값으로 보존하며 기존 성능과 같아질 것을 보장하거나 최종 평가값에 맞춰 설정을 반복 선택하지 않는다.

## Clean target도 평가할 때

MNIST clean-target 학습 결과가 서버에 있으면 해당 PGD 모델을 직접 지정한다. LivDet용 모델이나 FGSM용 모델을 지정하지 않는다.

```bash
python -u reproduction/evaluate_mnist_adaptive_pgd.py \
  --clean-checkpoint /path/to/mnist_target_ablation_RUN/pgd/seed_2026/clean/denoiser.pt \
  --device cuda --epsilon 0.4 --alpha 0.01 --steps 40 \
  --random-start --restarts 1 --eval-size 5000
```

서버에 clean-target MNIST 모델이 없다면 처음에는 Ours 1행만 평가한다. 기본 Ours는 Drive의 수집 모델이므로 추가 파일 이동이 필요 없다. `--rectified-checkpoint`로 별도 Ours 모델을 지정할 수 있으며 두 열은 항상 같은 모델을 평가한다.

Ours를 이미 평가했다면 위 명령에 `--clean-only`를 추가해 clean-target 모델만 평가한다. 기본 Drive 묶음에는 새 clean-target 가중치가 포함되지 않으므로, 기존 MNIST target 비교 실행에서 얻은 PGD clean 모델의 `denoiser.pt`를 서버로 옮겨 지정해야 한다. 두 실험을 한 표에 합칠 때는 epsilon·alpha·steps·random start·restarts·seed·batch size·평가 manifest가 같은지 확인한다.

## 공격·평가 정의

- Non-adaptive PGD: classifier의 logits CE를 최대화해 생성한 입력을 denoiser→classifier로 평가.
- Adaptive PGD: denoiser→가중 회색조 변환·clamp→classifier의 logits CE를 최대화. 입력까지 실제 gradient를 역전파.
- Untargeted L∞, ε .4, α .01, 40회, uniform random start 1회가 기본값. 설정값은 요청된 평가 조건이며 저장 공격의 역사적 생성 설정과 완전히 같다고 검증된 값은 아님.
- 공격 시작점과 projection 중심은 항상 clean 이미지. 저장된 공격 이미지에 추가 공격하지 않음. 새 공격은 float로 유지하며 저장 PNG를 거치지 않음.
- MNIST는 미분 가능한 float 후처리를 그대로 사용하므로 BPDA를 사용하지 않음. 두 모델은 eval 모드로 고정하고 가중치 gradient를 끄지만 입력 gradient는 유지.
- 각 batch에서 같은 seed로 만든 시작점을 모든 목적함수·모델에 공유. 각 공격의 시작점·모든 step·restart 중 공격에 성공한 후보를 유지하며, 같은 성공 여부이면 CE가 큰 후보 선택. 배치 크기도 재현 설정에 포함.
- 기본 평가 ID는 수집된 PGD manifest의 final evaluation 5000장. 학습 분할과 중복되지 않는지 검사.
- `--no-random-start` 또는 `--restarts 5`는 별도 공격 조건. 설정을 구분해 보고.

## 결과 읽기

`RESULTS` 폴더의 `summary.csv`, `metrics.json`, `config.json`을 공유하면 된다.

| 출력 열 | 의미 |
|---|---|
| clean_defended_percent | 방어를 통과한 clean 정확도 |
| nonadaptive_pgd_percent | Classifier만 공격한 후보의 방어 후 정확도 |
| adaptive_pgd_only_percent | End-to-end PGD 자체 후보의 정확도 |
| adaptive_pgd_combined_percent | Clean·non-adaptive·adaptive 중 하나라도 방어를 오분류시키면 실패로 세는 정확도 |
| classifier_nonadaptive_percent | Non-adaptive 공격 후 방어 없는 classifier 정확도 |

요청한 두 열을 그대로 비교할 때는 `nonadaptive_pgd_percent`와 `adaptive_pgd_only_percent`를 사용한다. 추가로 combined 열도 보고하면 기존 공격이 성공한 샘플을 adaptive 최적화가 놓쳐 강건성이 높아 보이는 현상을 막을 수 있다. Combined는 자체 adaptive 공격만의 수치와 구분해 표기한다.

`metrics.json`에는 gradient 유한성·zero-gradient 샘플 횟수·최대 perturbation을 기록한다. Zero-gradient 횟수는 이미지 수가 아니라 전체 gradient 호출에서 누적한 sample 수다. 손실·입력 gradient가 비유한 값이면 중단한다. 모든 gradient가 0이면 높은 정확도를 방어 성공으로 해석하기 전에 gradient 경로를 조사해야 한다.

`predictions.json`은 이미지별 결과이며 입력·체크포인트의 SHA256과 manifest를 보관한다. 평가에서 모델이나 공격 파라미터를 선택하지 않는다. 소량 평가는 동작 확인이며 논문과 비교할 최종 성능은 아니다. 단일 CE-PGD 설정의 결과를 검증된 robustness 보장으로 표현하지 않는다.
