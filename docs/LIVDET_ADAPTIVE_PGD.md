# LivDet adaptive BPDA-PGD

저장된 clean 최종 평가 1,250장에서 non-adaptive PGD와 전체 방어에 대한 adaptive BPDA-PGD를 새로 생성한다. 모델은 재학습하지 않는다.

## Ours 먼저 평가

```bash
git pull origin main
conda activate n2n
source .env

python reproduction/evaluate_livdet_adaptive_pgd.py --check-only

# 먼저 실제 모델·장치에서 소량 검사
python -u reproduction/evaluate_livdet_adaptive_pgd.py \
  --device cuda --eval-size 2 --batch-size 1

# 전체 평가
python -u reproduction/evaluate_livdet_adaptive_pgd.py \
  --device cuda --epsilon 0.03 --alpha 0.01 --steps 40 \
  --random-start --restarts 1 --seed 2026 --eval-size 1250 --batch-size 1
```

Ours 기본 모델은 원래 모아 둔 `N2N_assets/checkpoints/livdet/pgd/denoiser.pt`다. 이는 회수한 PGD-labelled 후보이며 실제 학습 공격은 미확인이다. 기존 저장 공격에서 방어 후89.76%, 논문91.36%를 참고값으로 기록하지만 새 PGD 결과와 동일하다고 가정하지 않는다.

## 첫 번째 제안 실험의 clean-target 모델 평가

교수님 첫 번째 clean/rectified 비교 실험에서 **새로 학습한 LivDet PGD clean-target 가중치**를 사용한다. 원래 수집한 가중치 묶음에는 이 파일이 포함되지 않는다. 서버에 있는 실제 실행 경로를 지정한다.

```bash
python -u reproduction/evaluate_livdet_adaptive_pgd.py \
  --clean-only \
  --clean-checkpoint ./reproduction/runs/livdet_target_ablation_20261104_161606_395725/pgd/seed_2026/clean/denoiser.pt \
  --device cuda --epsilon 0.03 --alpha 0.01 --steps 40 \
  --random-start --restarts 1 --seed 2026 --eval-size 1250 --batch-size 1
```

`--clean-only`를 빼면 두 모델을 한 번에 평가한다. 별도 실행 결과를 합칠 때는 공격 설정·분류기 해시·평가 이미지 해시·ID 순서가 동일한지 확인한다.

## Forward와 BPDA

실제 LivDet 방어는 denoiser RGB 출력의 clamp→8bit 절사→PIL grayscale을 classifier에 넣는다. 이 양자화·PIL 변환은 입력까지 직접 역전파할 수 없다. 따라서:

- Forward: 기존 PNG 평가와 동일한 연산 결과를 그대로 사용. 실제 파일 저장은 필요하지 않음.
- Backward: clamp된 RGB의 가중 회색조 합산으로 근사하고 양자화의 미분을 우회. RGB clamp 밖의 gradient는 0.
- PGD의 모든 후보 성공 여부와 최종 정확도는 **실제 forward**로 계산.

공격은 clean grayscale 입력을 중심으로 L∞ .03 안에서 생성하고 [0,1]로 제한한다. denoiser 내부에서는 3채널로 반복한다. 기존 아카이브 RGB 공격과 같은 입력 분포라고 단정하지 않는다. Non-adaptive는 classifier만 공격하고, adaptive는 denoiser와 classifier 모두를 통과해 gradient를 계산한다. 두 모델은 eval 모드로 고정한다.

기본값은 α .01, 40step, uniform random start 1회다. 모든 공격에 같은 초기점을 제공하고 모든 step·restart 중 성공한 후보를 유지한다. 여러 restart는 `--restarts 5`로 지정하는 별도 조건이다. 단일 BPDA 근사는 최강 공격임을 보장하지 않으며 zero-gradient 진단을 함께 확인한다.

## 결과

`RESULTS` 폴더의 `summary.csv`, `metrics.json`, `config.json`을 공유한다. `nonadaptive_pgd_percent`와 `adaptive_pgd_only_percent`로 요청한 표를 작성한다. `adaptive_pgd_combined_percent`는 clean·non-adaptive·adaptive에서 하나라도 오분류된 경우 실패로 집계하는 별도 결과다.

공통 공격 코드는 `adaptive_pgd.py`이며, 기존 MNIST 명령과 float 공격 경로는 유지한다. LivDet에서는 gradient가 BPDA 근사라는 점을 결과에 명시해야 한다. 파일·라벨·공격 예산 검증은 유지하며, 모델 학습 출처와 과거 공격 생성 조건의 불확실성도 함께 기록한다.
