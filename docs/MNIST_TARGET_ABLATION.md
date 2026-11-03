# MNIST clean-target vs. rectified-target

저장된 공격 이미지를 입력으로 사용하고 target만 clean/rectified로 바꿔 두 U-Net을 새로 학습한다. 과거 Ours 가중치는 이 실험의 비교 모델로 사용하지 않는다. EA는 우선 아카이브 AA에 대응시킨다.

## 실행

저장소 루트에서 환경을 활성화하고 가중치·데이터 경로를 읽는다.

```bash
conda activate n2n
source .env

# 학습 전 파일·분할 검사
python reproduction/mnist_target_ablation.py --check-only

# 6개 공격 × 2개 target: 총 12회 학습
python -u reproduction/mnist_target_ablation.py \
  --attacks fgsm pgd cw ba bim aa --seeds 2026 --device cuda \
  --checkpoint-selection validation-mse
```

우선 FGSM만 확인하려면 `--attacks fgsm`으로 실행한다. 반복 실험은 `--seeds 2026 2027 2028`로 지정한다. 여러 seed에서도 데이터 분할은 바꾸지 않는다.

## 고정 조건

- 기존 선택 모델의 manifest에 기록된 4,500 train / 500 validation / 5,000 final evaluation.
- 두 조건에 동일한 공격 이미지, U-Net first_pool=1, 같은 초기 state_dict와 미니배치 순서.
- L2(MSE), Adam lr .001, betas (.9,.99), eps 1e-8, 30 epoch, batch32, augmentation 없음.
- 기본값은 각 모델에서 자기 target에 대한 validation MSE가 최소인 epoch를 선택. 이전 MNIST 재현과 같은 선택 규칙이며, final evaluation 정확도는 선택에 사용하지 않음.
- 최종 평가에는 정답 라벨을 denoiser 입력으로 사용하지 않으며, 공격 입력과 clean 입력 정확도를 모두 측정.
- 회색조 입력을 RGB 3채널로 반복해 U-Net에 입력하고, float 출력의 가중 회색조 변환 후 [0,1] 범위로 제한.

명령에서 학습 설정을 바꾸면 두 target 조건에 함께 적용된다. 학습 seed와 checkpoint 초기 상태 해시, 입력 이미지 해시, ID·라벨 manifest를 실행 폴더에 기록한다. 원래 rectified 이미지 생성 조건과 공격 버전은 완전히 확인되지 않은 상태이며, 이 비교가 논문의 전체 학습 조건을 재현한다는 뜻은 아니다.

## 출력

실행 종료 시 `RESULTS` 경로가 출력된다.

- `attack_accuracy.csv`: 요청한 2행 × 6공격 표, 공격 입력 방어 후 정확도.
- `clean_input_accuracy.csv`: 같은 모델들의 clean 입력 정확도.
- `per_seed.csv`: 조건·공격·seed별 정확도와 선택 epoch.
- 각 조건의 `denoiser.pt`, `training_history.json`, `metrics.json`, `pairing.json`.
- 각 조건의 `best_validation_denoiser.pt`와 `last_denoiser.pt`도 별도 보관. `denoiser.pt`는 지정한 선택 규칙에 해당하는 모델.

여러 seed이면 셀에 평균±표본표준편차를 기록한다. Mean 열은 공격별 평균들의 산술평균이다. AA 열은 `EA (archive AA)`로 표시하며 표준 AutoAttack으로 단정하지 않는다.

초기 실행 `mnist_target_ablation_20261102_192534_733135`는 마지막 epoch를 선택했다. 이후 기본값을 이전 MNIST 재현과 같은 `validation-mse`로 수정했다. 초기 실행을 다시 재현하려면 `--checkpoint-selection last`를 지정한다. 기존 실행 폴더는 유지하며 새 실행은 별도 폴더에 저장한다.

Validation MSE의 값 자체는 clean/rectified 사이에서 직접 비교하지 않는다. 각 조건의 목표에 대한 MSE로 그 조건 안에서만 모델을 선택하고, 최종 비교 지표는 동일 평가 분할의 분류 정확도다. 이 규칙은 마지막 epoch 고정과 별도 실험 조건이므로 구분해서 보고한다. 30epoch·batch32는 이전 실행을 맞추기 위한 설정이며 논문의 전체 학습 설정이 확정됐다는 의미는 아니다.
