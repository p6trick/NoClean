# 현재까지 확인한 결과

기준은 `N2N_Journal.pdf`다. 정확도 단위는 %, 차이는 재평가−논문의 percentage points다. 아래 결과는 기존 실행의 집계 파일을 복사한 것이며, 배포 준비 중 새로 전체 실험을 돌린 결과는 아니다.

## 동일 공격 방어: Table 1·2

| 공격 | MNIST 논문 | MNIST 재평가 | 차이 | LivDet 논문 | LivDet 재평가 | 차이 |
|---|---:|---:|---:|---:|---:|---:|
| FGSM | 98.18 | 98.02 | -0.16 | 91.11 | 91.28 | +0.17 |
| PGD | 94.26 | 91.54 | -2.72 | 91.36 | 89.76 | -1.60 |
| CW | 97.79 | 96.82 | -0.97 | 89.27 | 88.00 | -1.27 |
| BA | 97.89 | 98.24 | +0.35 | 80.55 | 80.64 | +0.09 |
| BIM | 98.46 | 97.44 | -1.02 | 90.16 | 90.16 | 0.00 |
| AA | 95.80 | 95.08 | -0.72 | 88.95 | 89.04 | +0.09 |

MNIST는 새로 학습한 N2N 모델의 5,000장 평가다. LivDet은 회수한 기존 가중치의 1,250장 평가다. LivDet PGD 후보의 실제 학습 공격은 확인되지 않았다. 논문 수치와 가까운 추론 결과가 곧 학습 과정 재현을 뜻하지 않는다.

원본 집계: [MNIST](../results/mnist_matched/summary.csv), [LivDet](../results/livdet_matched.csv).

## Cross-attack: Table 3·4, Appendix A.7/A.8

N2N의 대각선을 제외한 30개 셀 평균은 LivDet **78.09**, MNIST **93.18**이다. 논문 Table 3·4의 열평균으로 계산한 전체 평균은 각각 76.43, 93.66이다. 행렬의 행은 학습 공격, 열은 평가 공격이다.

- [LivDet N2N 행렬](../results/cross_attack/livdet/matrix.csv), [논문 비교](../results/cross_attack/livdet/comparison.csv)
- [MNIST N2N 행렬](../results/cross_attack/mnist/matrix.csv), [논문 비교](../results/cross_attack/mnist/comparison.csv)
- [LivDet AT 행렬](../results/at_recovered/livdet/matrix.csv), [논문 비교](../results/at_recovered/livdet/cross_comparison.csv)

LivDet AT는 PGD v1과 나머지 확인된 모델을 조합한 복구 결과다. 30개 셀 평균은 71.96으로 논문 71.95와 가깝다. 두 실행의 집계 결과를 합쳤으며, 이 파일 자체가 별도의 새 추론 실행을 뜻하지 않는다. AT 후보의 선택은 논문 결과를 확인한 이후에 이루어졌다.

## Table 5·6

[Table 5 비교](../results/cross_dataset_table5/comparison.csv)는 LivDet 가중치에 MNIST용 pool1 구조를 적용한 전이 결과다. 당시 논문에서 실제로 사용한 구조·전처리·가중치 조합은 미확인이다.

[Table 6 비교](../results/clean_table6/comparison.csv)는 clean 입력 보존 결과다. 특히 LivDet CW·BA는 float/PNG 경로에 민감하며 논문값이 완전히 재현되지 않았다. Float와 PNG 경로는 양자화뿐 아니라 채널별 clipping 순서도 다르다.

## DIP와 미완료 비교 방법

[저장 DIP 결과](../results/dip_saved/summary.csv)는 FGSM 40.00, PGD 39.92, CW 40.00, BA 40.00이다. 대부분 Alive 예측으로 쏠린 결과이므로 성공적인 복원으로 해석하지 않는다.

[BIM·AA 새 DIP 결과](../results/dip_bim_aa_pilot/summary.csv)는 **각 2장만** 생성한 부분 평가다. 두 공격 모두 50%이며 전체 1,250장 결과나 논문값과 직접 비교할 수 없다. 학습된 모델 체크포인트 대신 이미지별 최적화를 수행한다.

DIPDefend는 일반 DIP와 다른 방법이다. 해당 방법의 기존 로컬 코드·결과는 확인되지 않았고, TVM·JPEG도 이 저장소의 완료된 비교 실험에 포함되지 않는다.

각 결과의 원래 실행 ID는 [PROVENANCE.json](../results/PROVENANCE.json), 해석의 제한은 [출처 안내](PROVENANCE.md)에 있다.
