# MNIST adaptive PGD epsilon sensitivity

## 기준 실행 조사와 설정 결정

기준 결과는 `mnist_adaptive_pgd_20261105_172406_795278`의 직접 측정값이다.

| 모델 | Non-adaptive PGD | Adaptive PGD | checkpoint |
|---|---:|---:|---|
| Clean target | 96.24% | 19.90% | 첫 번째 교수님 제안 실험에서 새로 학습한 PGD clean 모델, epoch16 |
| Rectified target | 91.64% | 8.76% | 수집 가중치의 MNIST PGD 모델, epoch16 |

실제 `config.json`은 **alpha=2/255≈0.007843137254902, 40회, random start 없음, probability CE, 마지막 반복, batch1, seed2026**이다. 요청서의 alpha0.01은 이 수치의 설정과 다르다. 이번에는 사용자가 선택한 대로 **기록된 기준 설정을 유지하고 epsilon만 변경**한다. alpha0.01을 적용하면서 같은 실험이라고 표현하지 않는다.

- 기존 진입점: `reproduction/evaluate_mnist_adaptive_pgd.py`
- 실제 공격·모델 wrapper: `reproduction/adaptive_pgd.py`의 `pgd_candidates`, `DefendedMNIST`
- Clean 모델 원본: `reproduction/runs/mnist_target_ablation_20261102_202650_774330/pgd/seed_2026/clean/denoiser.pt`; 서버 복사본 이름은 `mnist_pgd_clean_target.pt`.
- Rectified 모델: `${N2N_ASSET_ROOT}/checkpoints/mnist/pgd/denoiser.pt`
- Classifier: `${N2N_ASSET_ROOT}/checkpoints/mnist/classifier.pth`
- 평가 분할: `${N2N_ASSET_ROOT}/checkpoints/mnist/pgd/source_data_manifest.json`의 evaluation5000장. 기준 실행과 ID·순서·라벨·clean 파일SHA256을 일치시킨다.
- 기준 결과 파일: 위 실행 폴더의 `config.json`, `metrics.json`, `summary.csv`, `predictions.json`, `manifest.json`, `clean_image_hashes.json`.

기존 `--attack-profile legacy-reference` 진입점은 epsilon을0.4로 다시 설정한다. 따라서 shell에서 epsilon 옵션만 바꿔 반복 호출하면 올바른 sweep이 되지 않는다. 새 runner는 **기존 공격 함수를 직접 호출**하며 공격 알고리즘·전처리·모델 정의를 수정하지 않는다.

## 작업 범위

이번 코드 작성 중에는 설치, 다운로드, 모델 로딩·추론, 실험 실행, 테스트, Git push를 수행하지 않았다. 기존 코드와 결과 파일의 읽기 및 소스 수정만 수행했다. 아래 명령은 사용자가 직접 수행한다. 기존 `n2n` 환경과 데이터·가중치·기준 실행 폴더가 서버에 필요하다.

## 현재 PC에서 사용자가 push

아래 `/path/to/NoClean`은 실제 로컬 저장소 폴더로 바꾼다.

```bash
cd /path/to/NoClean
git add README.md docs/MNIST_EPSILON_SENSITIVITY.md \
  reproduction/epsilon_sweep_io.py \
  reproduction/mnist_epsilon_sweep.py \
  reproduction/analyze_mnist_epsilon_sweep.py \
  reproduction/requirements_analysis.txt
git diff --cached --stat
git commit -m "Add reference-matched MNIST epsilon sensitivity sweep"
git push origin main
```

## 서버에서 사용자가 pull·경로 검사

```bash
cd ~/data-vol1/NoClean
git pull origin main
conda activate n2n
source .env

python reproduction/mnist_epsilon_sweep.py \
  --reference-run ./reproduction/runs/mnist_adaptive_pgd_20261105_172406_795278 \
  --clean-checkpoint ./mnist_pgd_clean_target.pt \
  --check-only
```

`--check-only`는 기준 설정·모델 파일SHA256·평가 ID와 순서·라벨·clean 파일SHA256을 검사하고 모델 추론 전에 종료한다. 테스트를 실행하는 명령은 아니다. 기준 폴더가 다른 위치에 있으면 `--reference-run`을 바꾼다. 해당 폴더 전체를 복사했다면 서버의 옛 절대 경로는 평가 데이터 접근에 사용하지 않으며, 현재 `.env`와 체크포인트 manifest로 경로를 해결한다.

## 전체 평가

```bash
python -u reproduction/mnist_epsilon_sweep.py \
  --reference-run ./reproduction/runs/mnist_adaptive_pgd_20261105_172406_795278 \
  --clean-checkpoint ./mnist_pgd_clean_target.pt \
  --device cuda \
  --output ./reproduction/runs/mnist_epsilon_sensitivity_reference
```

epsilon0.05/0.10/0.15/0.20/0.30/0.40에서 두 모델을 평가한다. 전체 실행은5000장×6epsilon×2모델의40-step gradient 공격이므로 시간이 오래 걸린다. 정확한 이전 설정을 유지하려고 batch1도 유지한다. runtime에는 GPU·PyTorch·CUDA·TF32 플래그·코드 해시를 기록한다.

0.4의 clean/rectified 조건을 먼저 실행하고 기존19.90%·8.76%와 비교한다. 전체 조건이 끝나면 표는 epsilon 오름차순으로 기록한다. 0.4의 정확도가 다르면 결과를 보존하고 나머지 epsilon 진행을 중단한다. `reference_reproduction.csv`에는 정답 수 차이와 이미지별 예측 불일치 수도 기록한다. Random start가 꺼져 있으므로 차이를 random start 때문이라고 설명하지 않는다. 먼저 해시, torch/GPU 환경, 수치 계산 설정을 확인한다.

부분 동작 확인을 사용자가 직접 하려면 새 output 경로에 `--eval-size 16`을 추가할 수 있다. 이때 기준값은 과거 실행의 같은16개 ID 예측이며, 전체19.90%·8.76%와 비교하지 않는다. 이 결과는 partial로 표시한다. 작은 subset을 전체 실행에 이어 붙이지 않는다.

## 중단 후 재개

```bash
python -u reproduction/mnist_epsilon_sweep.py \
  --reference-run ./reproduction/runs/mnist_adaptive_pgd_20261105_172406_795278 \
  --clean-checkpoint ./mnist_pgd_clean_target.pt \
  --device cuda \
  --resume ./reproduction/runs/mnist_epsilon_sensitivity_reference
```

완료된 epsilon×모델 조건은 이미지 cache와CSV 해시를 확인해 재사용한다. 미완료 `.partial` 조건은 해당 조건의 처음부터 다시 계산한다. Resume은 데이터·가중치·공격 설정·코드·실행 환경이 같은 경우에만 허용한다. 실행 도중 코드나 환경을 바꾸지 않는다.

0.4 불일치의 원인을 확인한 뒤 차이를 명시하고 나머지를 계속할 필요가 있다면 위 재개 명령에 `--allow-reference-mismatch`를 추가한다. 이 옵션은 결과를 강제로 맞추거나 공격을 바꾸지 않는다. 불일치 상태는 결과 표와 요약에 남는다.

## 결과 분석·그래프·이미지

평가 runner 자체에는 새 패키지가 필요하지 않는다. 별도 분석 스크립트만 matplotlib를 사용한다. 설치가 필요하면 사용자가 직접 수행한다.

```bash
python -m pip install -r reproduction/requirements_analysis.txt

python reproduction/analyze_mnist_epsilon_sweep.py \
  --run ./reproduction/runs/mnist_epsilon_sensitivity_reference \
  --visual-epsilons 0.05 0.10 0.20 0.40 \
  --visualization-count 8
```

이 분석 명령은 GPU나 모델을 불러오지 않고 저장된 측정값·float 이미지만 읽는다. 기존에 matplotlib가 있으면 설치 명령은 생략할 수 있다. 완성된 요청 조건만 렌더링하며 완료 전에는 재개를 안내한다.

자동 시각화는 처음 두 고정 샘플과 epsilon별 공격 성공 변화·모델 간 차이가 있는 예시를 선택한다. 모든 epsilon에 같은 ID를 사용하며 선택 정책과 ID를 기록한다. 이는 정성 분석을 위한 사후 선택이며 전체 평가 통계에는 영향을 주지 않는다. 수동 지정은 `--sample-ids image_3052.png image_4444.png`처럼 사용한다.

각 샘플 PNG는 epsilon별 행과 아래5개 열로 구성된다.

1. 원본 clean — classifier 예측과 정답
2. Clean-target용 adaptive 공격 — 분류기 단독 예측과 방어 공격 성공 여부
3. Clean-target denoised — 전체 방어 예측
4. Rectified-target용 adaptive 공격 — 분류기 단독 예측과 방어 공격 성공 여부
5. Rectified-target denoised — 전체 방어 예측

모든 영상은 같은 [0,1] 표시 범위로 그린다. 실제 공격·평가에 시각화 PNG를 다시 입력하지 않는다. 큰epsilon에서 의미가 훼손됐는지는 시각적 분석이며, classifier 예측 변화만으로 입증하지 않는다.

## 출력 파일

- `accuracy_by_epsilon.csv`: 요청한2행×6epsilon 정확도 표.
- `accuracy_long.csv`: epsilon·supervision·정확도·샘플수·seed·공격 설정·모델 해시·ASR.
- `perturbation_stats.csv`: 평균/최대L∞와L2, denoised-clean RMSE, ASR 분자·분모, gradient 진단.
- `reference_reproduction.csv`: epsilon0.4와 기준 실행의 재현 비교.
- `robust_accuracy_curve.png`: 실제 측정값을 표시한 두 모델의 곡선.
- `visualizations/sample_*.png`, `visualization_manifest.json`: 동일 샘플 비교와 선택 기록.
- `summary.md`: 설정·결과·기준 재현·그림 링크를 담은 요약.
- `config.json`, `manifest.json`, `input_hashes.json`, `inputs.npz`: 실행·가중치·데이터 식별 및 원본 cache.
- `conditions/epsilon_*/{clean,rectified}/`: 조건별 `metrics.json`, `per_image.csv`, `images.npz`.

공격 이미지와 denoised 출력은0~1float32로 보존한다. 모든5000장×12조건에서 두 영상 cache의 비압축 크기는 약376MB이며 npz로 압축해 저장한다. 기존 데이터·원고·체크포인트·과거 결과는 수정하지 않는다.

## 지표와 해석

주 정확도는 과거 실행과 같은 마지막 adaptive 후보의 전체 평가 정확도다. ASR은 **해당 defended model이 clean에서 맞힌 샘플 중 공격 후 틀린 비율**이며 분자·분모를 함께 기록한다. 전체 오분류율100−accuracy와 구분한다. 처음부터 오분류인 clean까지 실패로 처리한 정확도는 별도 열이다.

L∞·L2는 clean과 공격 입력 간의 정규화된 픽셀 차이에서 계산한다. L2는 픽셀 수로 나누지 않는다. RMSE는 실제 classifier에 들어가는 가중 회색조·clamp 후 denoised 영상과 clean 사이의 샘플별RMSE 평균이다. 모든gradient의 유한성, L∞≤epsilon+1e-6, 유효 픽셀 범위를 평가 중 검사한다.

이번 설정에서는40×2/255≈0.313725이므로 epsilon0.4라도 초기점이clean이면0.313725보다 멀리 이동할 수 없다. 따라서 곡선을epsilon0.4 범위 전체의 최강 공격 결과로 표현하지 않는다. 마지막 반복과 고정step에 따른 비단조 현상은 그대로 보고하고 곡선을 인위적으로 단조화하지 않는다. 기존standard/logits-CE 공격 결과도 유지한다.
