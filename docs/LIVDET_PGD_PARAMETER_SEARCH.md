# LivDet PGD 생성 설정 탐색

## 목적과 범위

현재 저장 PGD의 undefended classifier 정확도3.76%와 새 과거 구현 가설의11.92%가 다르다. N2N 방어 후 수치가 비슷하다는 사실만으로 공격 생성이 재현됐다고 판단하지 않는다. 이 탐색은 노트북에서 확인한 설정과 저장 이미지의 픽셀 범위에서 얻은 가설을 **학습 또는 validation 측의 동일 이미지**에 적용해, 저장 공격의 생성 과정에 대한 단서를 찾는다.

전체 평가셋 비교는 `--split evaluation --sample-size 0`으로 실행한다. final-evaluation 1,250장의 원래 클래스 비율과 순서를 유지하며, 같은 샘플의 저장 공격과 각 생성 설정의 분류기 정확도를 비교한다. 이 결과는 평가셋에서 수행한 설정 비교이며, 과거 생성 설정을 확정하는 증거로 간주하지 않는다. N2N 모델은 로드하지 않는다.

## 후보 설정

| 항목 | 기본 후보 | 근거 |
|---|---|---|
| Epsilon | .03,8/255 | 저장 공격 범위 및8bit 저장에서 얻은 가설; exact generator 미확인 |
| Step size | 2/255,.01 | 남은 과거 코드 및 현재 실험의 비교 조건 |
| 반복 | 7,40 | GreenBit용 PGD-Copy1 및 PGD-CMFK |
| Random start | 없음,있음 | 과거 코드와 현재 구현 대조 |
| Loss | probability CE,logits CE | 과거 Softmax 포함 모델 사용 및 현재 logits 공격 |
| 전처리 | PIL L→tensor,RGB tensor→torchvision grayscale | 현재 loader와 후보 노트북의 ToTensor→Grayscale 순서 |
| 후보 선택 | 마지막 반복 | 과거 구현. 추가 best-iterate 비교는 명시적 옵션 |
| 출력 인코딩 | float,save_image 방식,truncate8 | 저장 전후 입력 차이 검사 |

기본64개 생성 설정을 검사하고 각 설정의3개 인코딩, 총192개 행을 보고한다. `--include-source-budget-controls`를 추가하면 노트북에 남은epsilon.3도 검사해96개 생성 설정·288개 행이 된다. .3은 현재 저장 공격과 다른 예산의 대조군으로 표시하고, 저장 공격 재현 후보 순위에서 제외한다. `PGD.ipynb`의400회나 음수 방향 업데이트는 보정 코드이므로 PGD 공격 후보에 자동 포함하지 않는다.

출력 `save_image` 인코딩은 multiply255→add.5→clamp→uint8에 대응하는 내림 연산이다. 단순 `torch.round`의 반올림 규칙과 구분한다. `tensor-gray`는 RGB bilinear resize 이후ToTensor된 영상에 torchvision의 회색조 가중치를 적용한다. 두 전처리의 clean 입력 차이와 인코딩 후 실제perturbation 범위도 기록한다.

동일sample-ID·seed의 random perturbation은 같은epsilon 안에서 공유한다. 원래random seed는 미확인이다. 모든 공격은untargeted,clean 중심projection,[0,1] clipping,batch1이다.

## 사용자 실행

코드 작성 중 설치·추론·테스트·push는 수행하지 않았다. 기존 `n2n` 환경을 사용하며 추가 패키지는 필요하지 않다. 원래 평가·adaptive 설정을 이 탐색 결과로 자동 교체하지 않는다.

먼저 현재 PC에서 수정한 코드를 직접 commit/push한다. 아래 경로는 실제 저장소 위치로 바꾼다.

```bash
cd /path/to/NoClean
git add reproduction/audit_livdet_pgd_generation.py \
  reproduction/search_livdet_pgd_parameters.py \
  docs/LIVDET_PGD_PARAMETER_SEARCH.md README.md
git commit -m "Add source-grounded LivDet PGD parameter search"
git push origin main
```

실행할PC에서:

```bash
conda activate n2n
source .env

python reproduction/search_livdet_pgd_parameters.py \
  --sample-size 8 --include-source-budget-controls --check-only

python -u reproduction/search_livdet_pgd_parameters.py \
  --device cuda --sample-size 8 --include-source-budget-controls \
  --output ./reproduction/runs/livdet_pgd_parameter_search_train8
```

학습 측Fake4장·Alive4장을 고정해 먼저 탐색한다. 모든조합을 검사하므로 단일PGD 실행보다 오래 걸린다. `--check-only`는 검색 계획·데이터·모델 해시까지만 확인하고 추론 전에 종료한다. 기존 `audit_livdet_pgd_generation.generate`를 재사용하되epsilon을 인자로 넘길 수 있게 했으며 기존 호출의기본값.03은 유지한다.

중단 후 같은 환경·코드·설정으로 재개:

```bash
python -u reproduction/search_livdet_pgd_parameters.py \
  --device cuda --sample-size 8 --include-source-budget-controls \
  --resume ./reproduction/runs/livdet_pgd_parameter_search_train8
```

완료된 생성 설정은 해시 검사 후 재사용하고 미완료 설정은 처음부터 다시 계산한다. 탐색 중 코드를 변경하지 않는다. 표본을 32장으로 늘리면 새로운 output 폴더에서 별도 실행한다.

## 결과 해석과 공유

- `summary.csv`: 모든 설정과 인코딩의 분류기 정확도·예측 일치·저장 영상과의 거리.
- `ranking.csv`: .03·8/255 가설만, 저장 영상과의 평균RMSE가 작은 순서. 이후 exact 픽셀일치·예측일치를 사용해 정렬.
- `archived_baseline.json`: 해당 탐색 표본에서의 저장 공격 정확도. 이 값을 final1250장의3.76%와 직접 비교하지 않는다.
- `config.json`, `manifest.json`, `input_hashes.json`: 탐색근거·실행설정·ID·원본파일·분류기 식별 정보.
- `cases/case_*/per_image.csv`: 각샘플의 픽셀거리·실제변화범위·예측일치.

먼저 `ranking.csv`, `summary.csv`, `config.json`을 공유한다. 상위 후보가 픽셀·예측 양쪽에서 유사한지 확인하고, 학습 측 추가 표본이나 validation에서 검토한다. 전체 평가 비교가 필요하면 아래 실행 방법을 사용한다. 정확도만 같거나 픽셀 일치가 불충분하면 원래 generator를 복구했다고 표현하지 않는다.

저장 공격 이후의 별도 이미지 변환, 실행 당시 사용한 모델 파일의 차이, 현재 남지 않은 함수 재정의가 원인일 수 있다. 이 유한 그리드로 결과가 맞지 않으면 이 항목들도 추적해야 하며, 추가 가설을 검사할 때에는 설정과 탐색 범위를 명시한다.


## 전체 final-evaluation 정확도 비교

기존 8장 실행과 다른 폴더를 사용한다. 추가 설치는 필요하지 않다.

```bash
conda activate n2n
source .env
python reproduction/search_livdet_pgd_parameters.py \
  --split evaluation --sample-size 0 --include-source-budget-controls --check-only &&
python -u reproduction/search_livdet_pgd_parameters.py \
  --device cuda --split evaluation --sample-size 0 \
  --include-source-budget-controls \
  --output ./reproduction/runs/livdet_pgd_parameter_search_eval1250
```

96개 생성 설정 각각에서 1,250장을 공격하고, 각 출력에 대해 float/PNG 반올림/PNG 내림 세 가지를 평가한다. GPU batch는 1이다. 전체 이미지 캐시는 CPU RAM에 보관한다. 100장마다 진행 상황을 표시하고 완료된 설정마다 결과를 저장한다. PNG 파일 자체나 N2N 출력은 생성하지 않는다.

중단 후 재개:

```bash
python -u reproduction/search_livdet_pgd_parameters.py \
  --device cuda --split evaluation --sample-size 0 \
  --include-source-budget-controls \
  --resume ./reproduction/runs/livdet_pgd_parameter_search_eval1250
```

- `accuracy_comparison.csv`: 같은 1,250장의 저장 공격 정확도와 차이가 작은 순서. `classifier_accuracy_percent`가 생성 공격 정확도, `delta_to_archived_pp`가 동일 샘플 저장 공격과의 차이, `delta_to_historical_3p76_pp`가 이전 기록 3.76%와의 차이다.
- `archived_baseline.json`: 이번에 실제 사용한 저장 공격의 정확도. 과거 3.76%와 다르면 이 차이도 함께 보고한다.
- `summary.csv`: 모든 설정의 원래 순서 결과. `ranking.csv`는 기존의 이미지 RMSE 순위를 유지한다.
- epsilon .3 대조군도 정확도 비교 파일에 포함하지만 `budget_role`로 구분한다. epsilon .03이나 8/255와 다른 예산이므로 혼동하지 않는다.

`accuracy_comparison.csv`, `archived_baseline.json`, `config.json`을 공유하면 정확도 중심으로 비교할 수 있다. 정확도가 비슷한 설정은 후속 non-adaptive/adaptive 비교에서 동일하게 고정할 수 있지만, 과거 설정을 정확히 복원했다고 표현하지 않는다.
