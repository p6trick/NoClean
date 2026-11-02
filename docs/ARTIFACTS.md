# Google Drive 가중치 배포

**가중치 다운로드: [Google Drive](https://drive.google.com/file/d/1JMJETrqqISeedka8RQw6mB3FGkspyV-A/view?usp=sharing)**

다운로드한 파일이 압축 파일이면 먼저 해제한다. `N2N_assets` 안에 아래의 `checkpoints/`와 `at_models/`가 있도록 구조를 유지한다. 가중치는 GitHub 코드에 포함하지 않는다. 이미지 데이터는 이 다운로드에 포함되지 않으며, [별도 요청](DATA.md)을 통해 제공한다.

```text
N2N_assets/
  checkpoints/
    manifest.json
    summary.csv
    mnist/
      classifier.pth
      {fgsm,pgd,cw,ba,bim,aa}/
        denoiser.pt
        model_config.json
        source_data_manifest.json
        metrics.json
        source_run_config.json
    livdet/
      classifier.pth
      {fgsm,pgd,cw,ba,bim,aa}/...
    code_snapshot/       최초 가중치 수집 시점의 참고 코드 사본
  at_models/
    manifest.json
    GB_*.h5             LivDet AT 후보 9개 (확인된 모델과 비교 버전)
    MNIST_*.h5          MNIST AT 모델 6개
```

다운로드 후 다음처럼 설정한다:

```bash
export N2N_ASSET_ROOT=/path/to/N2N_assets
python scripts/check_assets.py --verify-hashes
```

가중치는 총 29개다: N2N 12개, PyTorch 분류기 2개, AT 15개. 확인된 LivDet AT 조합은 FGSM/BIM/AA의 기본 파일, PGD는 `GB_PGD.h5`, CW는 `GB_CW2.h5`, BA는 `GB_BA2.h5`다. `--profile recovered`가 이 조합을 사용한다. v1/v2는 버전 비교용이다.

MNIST 가중치는 `state_dict`, epoch, config를 담은 dict이고, LivDet은 raw state_dict다. 동일 공격 평가에서는 MNIST first_pool=1, LivDet first_pool=2를 사용한다. Table 5에서만 LivDet 가중치에 pool1을 명시적으로 적용한다. 모델 정의는 공개 저장소의 `reproduction/models.py`를 사용한다.

원본 가중치를 이동하지 않고 복사했으며 SHA256을 확인했다. 메타데이터 경로는 `${N2N_PM_ROOT}` 같은 자리표시자로 바꿨지만 가중치 바이트는 변경하지 않았다. `checkpoints/manifest.json`의 `archive_sha256`은 경로 변환 전 보조 파일의 해시이며, 현재 복사본의 해시는 `sha256`이다.

`checkpoints/` 전체가 필요하다. `denoiser.pt`만 받으면 평가 ID·경로·설정이 빠진다. `code_snapshot/`은 참고 자료이며, 실제 실행은 GitHub 코드로 한다. 모델·데이터의 배포 조건은 소스 코드 라이선스와 별개다.
