# 코드·가중치·결과 출처

## 공개용 사본의 범위

기존 연구 작업 폴더에서 실행 코드와 집계 결과만 복사하고, 개인 절대 경로를 환경변수와 텍스트 자리표시자로 바꿨다. 원본 작업 파일과 가중치는 이동하거나 덮어쓰지 않았다. PDF, 이미지, 개별 예측, 전체 실행 로그는 공개용 코드 폴더에 포함하지 않는다.

배포용 `reproduction/paths.py`가 경로 변환을 담당한다. `cross_attack.py --matched-only`는 같은 가중치와 전처리를 사용해 Table 1·2의 대각선만 계산하기 위해 추가한 옵션이다. 그 밖의 연구용 진단 스크립트는 선택된 가중치 외에 원래 아카이브 파일을 요구할 수 있다.

## N2N

기존 `Self2Self/noise2noise` 계열 코드와 실험 노트북을 참고했다. 모델 정의와 회색조 변환은 `reproduction/models.py`에 있다. MNIST의 first_pool=1과 LivDet의 first_pool=2는 별도 설정이다. 모델별 출처, epoch, SHA256, 평가 설정은 Drive의 `checkpoints/manifest.json` 및 각 `model_config.json`에 기록했다.

MNIST 모델은 `mnist_suite_20261031_003330_504345`에서 새로 학습했다. LivDet 모델은 기존 학습 가중치를 회수했다. 회수된 LivDet 모델의 최종 학습 코드·데이터 버전·모든 하이퍼파라미터가 확인된 상태는 아니다. PGD로 표시한 후보는 학습 공격 자체도 미확인이다.

## 데이터와 평가

저장된 공격 이미지와 보정 이미지를 활용한다. 기존 공격의 생성 파라미터 및 일부 train/target 버전 대응은 완전히 확인되지 않았다. 평가 분할과 clean/attack ID 대응은 보존된 manifest로 검사한다. 회수된 가중치의 원래 학습 분할이 이 평가 분할과 완전히 독립적이었는지는 확정할 수 없다.

True-label rectification은 학습용 목표 생성 또는 진단용 참조다. 정답 라벨을 모르는 배포 시 방어는 공격 이미지→N2N→분류기 경로를 사용한다. 전체 방어를 겨냥한 adaptive attack 실험은 수행한 것으로 주장하지 않는다.

## AT

기존 `PGD` 폴더의 Keras H5 파일을 평가했다. 확인된 LivDet 조합은 PGD v1, CW/BA v2, FGSM/BIM/AA 기본 버전이다. 원래 파일명을 보존하고 SHA256을 기록했다. 이 조합의 선택은 논문 결과를 본 뒤의 복구 작업이며 독립적인 사전 선택 실험이 아니다.

## DIP

`deep-image-prior/denoising_finger_FGSM.ipynb`, `denoising_finger_PGD.ipynb`, `denoising_finger_CW.ipynb`, `denoising_finger_BD_Alive.ipynb`, `denoising_finger_BD_Fake.ipynb`에서 설정을 확인했다. 모델 코드 사본 8개의 해시는 `reproduction/legacy_dip/SOURCE.json`에 있다. 저장된 결과는 `Self2Self/noise2noise/temporary/data/ECCV/DIP`에서 확인했다.

BIM·AA에 대한 검증된 기존 평가 이미지가 없어 새 생성 코드를 작성했다. 원래 실행 seed는 미확인이므로 새 실행은 명시적인 seed를 사용한다. 801회 반복은 기존 노트북을 따른 값이며 논문에 적힌 800회와 구분한다. 새 부분 평가를 기존 전체 결과와 섞지 않는다.

외부 코드 라이선스와 수정 내역은 [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)에 있다. 데이터·가중치의 공개 권한은 코드 라이선스와 별도로 확인해야 한다.
