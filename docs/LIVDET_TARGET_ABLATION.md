# LivDet clean-target vs. rectified-target

두 조건 모두 새로 학습하며 target만 clean/저장 rectified로 바꾼다. 과거 체크포인트는 비교군으로 사용하지 않는다. 입력 공격·평가 분할·초기 state_dict·배치 순서·crop 좌표·학습 설정은 두 조건에서 같다.

## 실행

```bash
conda activate n2n
source .env

# 6개 공격의 저장 이미지와 분할 검사
python reproduction/livdet_target_ablation.py \
  --attacks fgsm pgd cw ba bim aa --check-only

# 먼저 FGSM 두 조건
python -u reproduction/livdet_target_ablation.py \
  --attacks fgsm --seeds 2026 --device cuda \
  --epochs 100 --batch-size 1 --crop-size 581
```

나머지를 실행하려면 `--attacks pgd cw ba bim aa`를 지정한다. 한 번에 모두 실행하려면 `--attacks fgsm pgd cw ba bim aa`를 지정한다. 전체 이미지를 학습하므로 MNIST보다 시간이 오래 걸린다.

## 설정과 근거

| 항목 | 적용 설정 | 근거·범위 |
|---|---|---|
| Epoch / batch / lr | 100 / 1 / .001 | 최신 원고에 명시된 값 |
| Optimizer / loss | Adam, betas (.9,.99), eps 1e-8 / MSE | Adam은 원고, 세부 값은 기존 구현 |
| Scheduler | 없음 | 두 조건에 같은 고정 lr 적용 |
| Architecture | Denoiser first_pool=2 | 회수된 LivDet 모델과 같은 구조; 논문 최종 구조 확정은 아님 |
| 입력 / crop | RGB, 581×581 전체 이미지 | 기존 3채널 구현 유지; 원고는 grayscale로 기술. full-image 학습은 명시적 구현 선택 |
| 분할 | 1125 train / 125 validation / 1250 final evaluation | 기존 재현의 seed2026 고정 분할; 학습 seed로 바꾸지 않음 |
| 선택 | 각 조건의 validation MSE 최소 epoch | Final evaluation은 모델 선택에 사용하지 않음 |
| 출력 | RGB clamp→8bit 절사→PIL grayscale | 기존 LivDet PNG 평가 경로; 실제 PNG 파일로 저장·재로딩하지 않아도 같은 연산 적용 |
| Clean 평가 | PIL grayscale를 3채널 반복 | 이전 clean 평가와 같은 입력 경로 |

원고의 1250장 fitting pool 중 125장을 validation으로 분리하므로 실제 학습 장수는 원고와 다르다. 원래 공격·rectified 생성 조건 역시 완전히 검증되지 않았다. 따라서 이 실험을 논문 설정 전체의 완전 재현으로 표현하지 않는다. 두 target의 validation MSE 값 자체를 서로 비교하지 않고, 각 모델 내부의 epoch 선택에만 사용한다.

`--crop-size 128`은 두 조건 모두 128patch 학습·중앙 validation patch로 바꾸는 별도 조건이다. 전체 이미지를 사용하는 실행과 구분해 보고한다. `--output-processing float`도 별도 평가 조건이며 기본 PNG와 혼용하지 않는다. EA는 아카이브 AA에 대응하는 것으로 표기한다.

## 출력

`RESULTS`가 가리키는 폴더에 아래를 저장한다.

- `attack_accuracy.csv`: target별 공격 입력 방어 정확도.
- `clean_input_accuracy.csv`: target별 clean 입력 정확도.
- `per_seed.csv`: 선택 epoch·초기 상태 해시·정확도·후처리.
- `config.json`, 공격별 `manifest.json`·`input_hashes.json`.
- 조건별 `denoiser.pt`, `best_validation_denoiser.pt`, `last_denoiser.pt`, 학습 이력과 평가 결과.

기존 recoverable PNG 오류는 기존 로더와 같은 방식으로 읽고 `image_decode_recoveries.json`에 기록한다. 이미지를 조용히 제외하거나 원본 파일을 수정하지 않는다. 여러 seed를 실행하려면 `--seeds 2026 2027 2028`로 지정한다. 표의 셀에는 평균±표본표준편차를 기록한다.
