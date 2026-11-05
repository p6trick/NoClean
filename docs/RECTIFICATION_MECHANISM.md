# Rectification mechanism 분석

교수님 세 번째 제안 실험이다. 저장된 공격 이미지를 고정하고 final-evaluation 샘플에서 rectified 이미지를 새로 생성하며, 분류 복구와 clean 픽셀 거리의 관계를 분석한다. N2N 가중치를 로드하거나 denoiser를 학습하지 않는다.

## 서버 실행

```bash
git pull origin main
conda activate n2n
source .env

# 파일·평가 분할·분류기 해시 검사
python reproduction/analyze_rectification.py --check-only

# MNIST 먼저: 6공격 각각5000장
python -u reproduction/analyze_rectification.py --datasets mnist --device cuda

# LivDet: 6공격 각각1250장
python -u reproduction/analyze_rectification.py --datasets livdet --device cuda
```

두 데이터셋을 한 번에 실행하려면 `--datasets mnist livdet`을 지정한다. 특정 공격만 수행하려면 `--attacks fgsm pgd`처럼 지정한다. `--mnist-eval-size 16 --livdet-eval-size 2`는 소량 동작 확인이며 결과에 partial로 기록한다.

## 고정 설정

- 입력 공격은 모델별 `source_data_manifest.json`의 final evaluation에서 가져온다. 공격을 새로 생성하거나 기존 저장 target을 읽지 않는다.
- 기본 rectification: MNIST epsilon.3, LivDet epsilon.03, step size2/255, 최대40회, random start 없음. 최신 원고의 보정 값을 따른 조건이며 과거 저장 target의 정확한 생성 버전이라는 뜻은 아니다.
- 정답 라벨의 logits CE를 낮추도록 입력을 업데이트한다. Projection 중심은 clean이 아니라 **공격 이미지**다. [0,1] 범위로 제한하고 정답 분류에 도달하면 해당 샘플 업데이트를 중단한다.
- 처음부터 정답이면0회로 유지한다. 실패하면 마지막 iterate를 사용한다. 전체 배치의 반복이 아니라 이미지별 실제 업데이트 횟수를 기록한다.
- 분류기는 eval 모드로 고정하며 가중치 학습은 없다. 기본값은 MNIST batch32, LivDet batch1이다.
- 세 영상 모두 같은 [0,1] grayscale 픽셀 공간에서 비교한다. 새 rectified는float 상태로 측정하고 PNG 양자화를 거치지 않는다.

다른 조건은 명시적으로 지정해 별도 보고한다. 예를 들어 `--mnist-rectify-epsilon 0.4`는 기본.3과 다른 조건이며, `--gradient-loss legacy-probability-ce`는 logits CE와 다른 진단이다. 최종 평가 결과를 보고 설정을 반복 선택하지 않는다.

## 지표

| 표의 열 | summary.csv 열 | 정의 |
|---|---|---|
| Success (%) | success_percent | 최종 rectified가 정답인 전체 비율;0회 성공 포함 |
| Failure (%) | failure_percent | 최대 반복 후에도 오분류인 전체 비율 |
| Avg. iter. | avg_iterations_success_including_zero | 성공 샘플의 평균 업데이트 수;0회 포함 |
| Adv.-Clean RMSE | adv_clean_rmse | 각 샘플RMSE(x_adv,x)의 산술평균 |
| Rect.-Clean RMSE | rect_clean_rmse | 각 샘플RMSE(x_rec,x)의 산술평균 |
| Closer (%) | closer_percent | d_rec < d_adv−1e-12인 전체 비율 |

0회 샘플이 효과를 부풀리지 않도록 `initially_correct_percent`, 처음 오분류된 샘플만의 `recovery_among_initially_wrong_percent`, 실제 복구 성공 샘플의 `avg_iterations_success_positive_only`도 함께 기록한다. 처음 오분류된 샘플과 복구된 샘플에 대한 Closer 비율·복구된 샘플의 RMSE도 제공한다. 분모가 0인 지표는 null/빈 칸으로 표시한다.

Success와Closer는 다른 개념이다. 정답 분류에 성공하면서도 clean과의 RMSE는 커질 수 있다. 이것만으로 classifier-specific이라고 확정할 수는 없고, 그 주장은 다른 분류기에 대한 전이 검사로 추가 확인해야 한다.

## 출력과 공유

`RESULTS` 폴더에 전체 `summary.csv`, `config.json`, `completion.json`이 생긴다. 데이터셋별 `mnist/summary.csv`, `livdet/summary.csv`도 저장한다. 공격별 폴더에는 `per_image.csv`, `metrics.json`, `manifest.json`, `input_hashes.json`이 있다. 각 공격이 완료될 때마다 집계 파일을 갱신한다.

먼저 `summary.csv`와 `config.json`을 공유하면 표를 작성할 수 있다. 상세 원인 분석은 해당 공격의 `per_image.csv`를 추가로 확인한다. EA는 archive AA에 대응하는 것으로 표기한다.

기본적으로 새 이미지 전체는 디스크에 저장하지 않는다. 분석용 이미지를 보관하려면 `--save-rectified`를 추가한다. 공격별 `analysis_only_rectified/`에0~1float 텐서를 저장하고 `analysis_only`·`not_for_training` 태그를 붙인다. 특히 LivDet은 저장 용량이 크다. 어떤 경우에도 이 코드에서 새 이미지를 학습 target으로 사용하지 않는다.

True-label rectification 성공률은 분석용 참조이며 정답을 모르는 배포 방어의 정확도로 보고하면 안 된다. LivDet의 기존 손상 PNG는 이전과 같은 tolerant loader로 읽으며, 제외하지 않고 복구 기록을 보관한다.
