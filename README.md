# N2N reproduction

`N2N_Journal.pdf`의 MNIST/LivDet 실험을 재실행하고, 회수한 기존 가중치와 저장 이미지를 검증하기 위한 코드다. 전체 논문을 완전히 재현했다고 주장하는 저장소는 아니다.

**체크포인트와 데이터는 Git에 포함하지 않는다.** 별도 Drive 폴더의 가중치와, 실험에 사용한 clean/공격 이미지가 모두 필요하다. 체크포인트만 다운로드해서는 표 전체를 실행할 수 없다.

- **가중치:** [Google Drive 다운로드](https://drive.google.com/file/d/1JMJETrqqISeedka8RQw6mB3FGkspyV-A/view?usp=sharing) · [구성 안내](docs/ARTIFACTS.md)
- **데이터:** 요청을 통해 별도 제공한다. 이 저장소의 **Issues**에 필요한 데이터셋(MNIST/LivDet)과 재현할 실험을 적어 요청해 주세요. [데이터 요청·구성 안내](docs/DATA.md)

## 구성

```text
reproduction/       학습·평가·진단 코드
reproduction/legacy_dip/  출처·해시를 보존한 DIP 모델 코드
docs/               실행 방법, 데이터·가중치 구성, 해석의 한계
results/            현재까지 확인한 집계 결과 (이미지·개별 예측 제외)
scripts/            배포 파일, 가중치, 데이터 경로 검사
LICENSES/           외부 코드의 원본 라이선스 문구
```

기준 원고에서 Table 1은 LivDet, Table 2는 MNIST의 동일 공격 방어, Table 3·4는 cross-attack, Table 5는 LivDet→MNIST 전이, Table 6은 clean 성능 보존이다. Appendix A.7/A.8은 cross-attack 상세 행렬이다. 최신 원고의 표 번호와 혼용하지 않는다. 원고 PDF는 포함하지 않았다.

## 설치와 경로

Linux와 Conda 기준이다. N2N·DIP에는 `n2n`을 사용하며, AT 평가를 수행할 때만 별도의 `n2n_tf` 환경이 필요하다. 환경 생성은 처음 한 번만 수행한다.

```bash
# N2N / DIP
conda env create -f reproduction/environment.yml
conda activate n2n

# 기존 .env가 없을 때만 예제에서 생성
test -f .env || cp .env.example .env
# .env의 N2N_ASSET_ROOT와 N2N_PM_ROOT를 실제 위치로 수정
source .env
python scripts/check_assets.py --verify-hashes
python scripts/check_data.py
```

```bash
# AT 평가용 환경: 필요할 때 추가 생성
conda env create -f reproduction/environment_tf.yml
conda activate n2n_tf
source .env
```

같은 이름의 환경이 이미 있으면 생성 명령을 반복하지 않고 `conda activate n2n` 또는 `conda activate n2n_tf`로 활성화한다. 버전·GPU 확인 명령은 [가상환경 안내](docs/ENVIRONMENT.md)에 있다.

- `n2n`: PyTorch N2N, DIP, 저장 이미지 평가.
- `n2n_tf`: 기존 Keras AT 모델 평가. TensorFlow 2.15.1/Keras 2.15.0.
- `N2N_ASSET_ROOT`: Drive에서 내려받은 폴더. 그 안에 `checkpoints/`, `at_models/`가 있어야 한다.
- `N2N_PM_ROOT`: 기존 아카이브와 같은 상대 경로 구조의 데이터 폴더.

이미지를 따로 모은 `N2N_data` 묶음을 사용할 때는 `N2N_PM_ROOT=/path/to/N2N_data/PM`으로 설정한다. 이 데이터 폴더도 GitHub에는 포함하지 않는다. 복사 범위와 재수집 명령은 [데이터 안내](docs/DATA.md)에 있다.

가중치를 다운로드한 뒤 [가중치 안내](docs/ARTIFACTS.md)의 구조를 유지한다. 데이터는 별도 요청 후 [데이터 안내](docs/DATA.md)에 따라 연결한다.

## 확인된 가중치로 평가

```bash
conda activate n2n
source .env

# Table 1·2: 선택된 가중치의 동일 공격 방어
python -u reproduction/cross_attack.py --datasets mnist livdet --matched-only

# Table 3·4 및 Appendix A.7/A.8: 6×6 교차 평가
python -u reproduction/cross_attack.py --datasets mnist livdet

# Table 5: 기존 MNIST용 pool1 구조에 LivDet 가중치 적용
python -u reproduction/table5_transfer.py --eval-size 5000 --modes pool1

# Table 6: 같은 데이터셋의 clean 입력 평가
python -u reproduction/table6_clean.py --datasets mnist livdet
```

평가 결과는 실행별 `reproduction/runs/`에 저장된다. 기본 평가 장수는 MNIST 5,000장, LivDet 1,250장이다. **Table 5의 pool1은 기록된 LivDet pool2와 다른 구조 설정이며, 당시 Table 5에 실제 사용됐는지는 미확인이다.**

## 비교 방법

전체 방어에 대한 새 공격 평가는 [MNIST adaptive PGD](docs/MNIST_ADAPTIVE_PGD.md)를 참고한다. `reproduction/evaluate_mnist_adaptive_pgd.py`는 같은 clean 입력에서 classifier-only PGD와 end-to-end PGD를 새로 생성하고, 고정된 PGD denoiser를 평가한다.

LivDet은 [LivDet adaptive BPDA-PGD](docs/LIVDET_ADAPTIVE_PGD.md)의 `reproduction/evaluate_livdet_adaptive_pgd.py`를 사용한다. 실제 PNG 후처리를 forward에 유지하고 backward에 미분 가능한 근사를 적용한다.

```bash
conda activate n2n_tf
source .env
# 확인된 조합: PGD v1, CW/BA v2
python -u reproduction/evaluate_at.py --datasets livdet --profile recovered

conda activate n2n
source .env
# 저장 결과가 확인된 DIP 4개 공격
python -u reproduction/evaluate_saved_dip.py --attacks fgsm pgd cw ba

# BIM·AA DIP 새 생성: 이미지마다 801회, 장시간 작업
python -u reproduction/generate_dip_livdet.py --attacks bim aa --eval-size 2
```

DIP BIM·AA 전체 2,500장은 측정된 장당 약 2분 기준 약 88시간 규모다. 중단 후 재개·장수 확장 방법은 [실험 안내](docs/EXPERIMENTS.md)에 있다. DIPDefend·TVM·JPEG는 현재 구현/평가가 완료되지 않았다.

## 재학습과 연구용 진단

교수님 세 번째 제안 실험은 [Rectification mechanism 분석](docs/RECTIFICATION_MECHANISM.md)을 참고한다. `reproduction/analyze_rectification.py`가 평가 분할에서 보정을 새로 수행하고 성공률·실패율·반복 수·clean RMSE·Closer 비율을 기록한다. Denoiser를 학습하지 않는다.

Clean target과 rectified target의 통제 비교는 [MNIST target 비교 실험](docs/MNIST_TARGET_ABLATION.md)을 참고한다. `reproduction/mnist_target_ablation.py`가 같은 초기 상태와 데이터 순서로 두 모델을 새로 학습한다.

LivDet은 [LivDet target 비교 실험](docs/LIVDET_TARGET_ABLATION.md)의 `reproduction/livdet_target_ablation.py`를 사용한다. 기본 설정은 100epoch·batch1·전체 이미지 학습이며, 먼저 FGSM 두 조건을 확인할 수 있다.

```bash
conda activate n2n
source .env
python -u reproduction/suite.py --attacks fgsm pgd cw ba bim aa
python -u reproduction/livdet.py --attacks fgsm
```

MNIST 재학습 모델과 회수한 LivDet 모델의 출처는 다르다. 위 기본 재학습 설정이 논문 최종 학습 절차와 같다는 의미는 아니다. `screen_pgd_checkpoints.py`, `evaluate_legacy_livdet.py` 등은 선택된 Drive 파일 외에 원래 전체 아카이브를 요구하는 연구용 코드다.

## 결과와 검증

[결과 요약](docs/RESULTS.md) · [실험 설정/명령](docs/EXPERIMENTS.md) · [가중치·데이터 출처와 한계](docs/PROVENANCE.md)

배포용 사본에서 수행한 검사와 소량 추론 범위는 [검증 기록](docs/VALIDATION.md)에 있다.

```bash
python scripts/check_release.py
python -m unittest discover -s tests
# n2n 환경에서 원래 파이프라인 테스트
python -m unittest discover -s reproduction -p 'test_*.py'
```

공격은 원래 분류기에 대해 생성된 저장 이미지다. 전체 방어를 겨냥한 adaptive robustness 검증은 아니다. PGD 라벨의 LivDet N2N 모델은 학습 공격이 미확인인 후보이며, 일부 모델 버전은 논문 결과를 확인한 뒤 찾아낸 복구 결과다.

외부 코드의 출처·라이선스는 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)를 참고한다. 프로젝트 자체의 공개 코드 라이선스는 아직 지정하지 않았다.

실제 `.env`, 이미지 데이터와 가중치는 커밋하지 않는다. `.env.example`과 문서의 경로 예제를 사용해 각자의 다운로드 위치를 지정한다.
