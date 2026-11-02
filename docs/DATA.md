# 데이터 구성

## 데이터 요청

**이미지 데이터는 요청을 통해 별도 제공한다 (data available upon request).** 이 저장소의 **Issues**에 아래 내용을 적어 요청해 주세요.

- 필요한 데이터셋: MNIST, LivDet 또는 둘 다
- 재현할 표·실험 또는 필요한 공격 종류

가중치용 Google Drive 링크에는 이미지가 포함되지 않는다. 제공받은 `N2N_data` 폴더는 코드·가중치와 별도로 보관하고, 아래 안내에 따라 경로를 지정한다.

## 제공받은 데이터 연결

가중치 배포 폴더에는 clean·공격·rectified·저장 DIP 이미지가 포함되지 않는다. 정확한 표 재실행에는 실험 당시의 이미지가 필요하다. 일반 MNIST를 새로 받거나 공격을 다시 생성하는 것만으로 기존 저장 공격과 같아지지는 않는다.

기존 아카이브의 상대 경로를 유지한 폴더를 지정한다:

```bash
export N2N_PM_ROOT=/path/to/PM
```

별도 복사본 `N2N_data`를 사용하는 경우 `N2N_PM_ROOT=/path/to/N2N_data/PM`으로 설정한다. 최상위 `N2N_data`가 아닌 그 아래 `PM`을 지정해야 한다. 가중치는 계속 별도의 `N2N_assets`에서 읽는다.

## 데이터 복사본 만들기

기존 아카이브에서 학습·평가에 사용하는 MNIST/LivDet clean·공격·보정 이미지와, 확인된 4개 공격의 저장 DIP 결과를 모은다. 원래 PM 상대 경로를 유지하며 가중치·노트북·불필요한 아카이브 변형은 복사하지 않는다.

```bash
python scripts/collect_data.py --source-root /path/to/original/PM \
  --destination /path/to/N2N_data --plan-only
python scripts/collect_data.py --source-root /path/to/original/PM \
  --destination /path/to/N2N_data
```

복사 중 파일별 SHA256을 확인하고 `N2N_data/files.csv`에 상대 경로·용량·해시·사용 구간을 기록한다. 원본을 삭제하지 않는다. 재실행하면 같은 복사본은 해시를 검사해 재사용하고, 다른 내용의 기존 파일은 덮어쓰지 않고 오류로 알린다. 이 목록은 NAS 전체 이미지 목록이 아니라 현재 재현 코드가 사용하는 범위다.

2026-11-02에 모은 이미지 묶음은 총 **167,500장**, 약 **9.18GB**다. 별도로 MNIST 순서 파일 1개와 관리용 manifest를 포함한다.

| 데이터셋 | Clean | 공격 | Rectified | 저장 DIP | 이미지 합계 |
|---|---:|---:|---:|---:|---:|
| MNIST | 10,000 | 60,000 | 60,000 | 0 | 130,000 |
| LivDet | 2,500 | 15,000 | 15,000 | 5,000 | 37,500 |

공격·rectified는 각각 6개 공격을 포함하며, 저장 DIP는 FGSM·PGD·CW·BA 4개 공격이다. 학습·검증·평가 대응은 `N2N_data/manifests/`에 보존했다. LivDet의 학습 분할은 현재 재현 코드의 규칙이며 회수한 과거 모델의 원래 학습 분할을 확정한 기록은 아니다.

대략적인 루트 구조:

```text
PM/
  PGD/MNIST/data/                canonical clean MNIST
  FGSM/image_names.txt          original canonical order
  PGD/data/2015/GreenBit/...    clean LivDet Alive
  PGD/CWAttack/...
  PGD/BoundaryAttack/...
  BIM/Fingerprint/...
  AA/Finger/...
  Self2Self/noise2noise/temporary/data/
    MNIST/real_{FGSM,PGD,CW,BA,BIM,AA}_valid/
    ECCV/FGSM_0.03/...
    ECCV/PGD_0.03/...
    ECCV/Fake_all/...
    ECCV/DIP/...               located saved DIP outputs
```

정확한 파일 경로·ID·라벨은 Drive의 모델별 `source_data_manifest.json`에 기록돼 있다. 자리표시자를 환경변수로 치환해서 읽으며 기존 평가 분할을 보존한다. 파일이 없으면 명시적으로 실패한다.

```bash
python scripts/check_data.py
python scripts/check_data.py --datasets mnist --include-training --include-targets
```

`check_data`는 파일 존재를 확인하며 원본 이미지 전체의 바이트 동일성을 증명하지는 않는다. LivDet 묶음은 평가 manifest만 보존하므로 학습 데이터는 `reproduction/livdet_experiments.json`의 경로 규칙으로 연결한다. MNIST 학습 경로는 `saved_experiments.json`에 있다.

체크포인트는 Drive 가중치 폴더에서 읽으므로 PM 아래에 같은 모델을 중복 배치할 필요는 없다. 단, 전체 아카이브 조사용 스크립트는 추가 후보 모델을 요구할 수 있다. 학습·평가·공격·보정 구간을 구분해서 사용한다.
