# LivDet 저장 PGD와 새 PGD 생성·전처리 진단

조사일: 2026-11-05. 기존 저장 공격의 N2N 정확도89.76%와 새 non-adaptive PGD의58.08% 차이를 조사했다. Adaptive의0%와는 별도로 non-adaptive 입력 경로만 검사했다. 모델·데이터를 수정하거나 평가 결과로 가중치를 선택하지 않았다.

## 확인한 사실

1. 기존 재평가와 새 실행의 classifier SHA256 및 PGD 후보 denoiser SHA256은 같다. 과거 PGD 후보의 실제 학습 공격은 여전히 미확인이다.
2. 두 평가 모두 denoiser 출력에 RGB clamp→8bit 절사→PIL grayscale을 적용한다. 새 코드의 BPDA는 forward 값을 그대로 유지하며, backward에서만 양자화를 우회한다. 따라서 이 output 경로를 단순히 float로 바꿨기 때문에 성능이 낮아진 것은 아니다.
3. 저장 공격은 이미 8bit PNG로 양자화돼 있다. 새 공격은 float를 denoiser에 직접 입력한다. 이는 **denoiser 입력 양자화**의 차이이며 **denoiser 출력 후처리**와 구분한다.
4. Train/evaluation 각각8 Fake+8 Alive, 총32장의 픽셀 검사에서 저장 공격은 RGB 파일이지만 모두 R=G=B였다. RGB 그대로 입력한 값과 grayscale 3채널 반복값이 같았다. 이 표본에서는 색상 채널 차이가 원인이 아니다.
5. 검사한 저장 공격의 최대 clean 대비 L∞는8/255≈.0313726이었다. 새 공격은.03으로 제한된다. PNG round/truncate가 이 정도의 차이를 만들 수 있지만, 이 관찰만으로 과거 encoder 또는 정확한 epsilon을 확정할 수는 없다.

## 학습 측 동일8장의 단일 조건·조합 진단

최종 평가를 제외하고 고정된 training ID8장(클래스별4장)에서 비교했다. 모든 생성 공격의 epsilon은.03, 반복40회다. per-ID seed로 시작점을 고정했고 각 이미지를 batch1로 검사했다. 아래는 진단용 소량 결과이며 전체 정확도가 아니다.

| 공격 조건 | N2N 방어 후 정확도 |
|---|---:|
| 저장 PGD 입력 | 100.00% |
| 저장 PGD를 grayscale 3채널로 다시 입력 | 100.00% |
| 현재 방식: logits CE, random start, alpha.01, best iterate | 50.00% |
| 현재 방식 입력을8bit round 또는 truncate | 각각50.00% |
| 현재 방식에서 last iterate만 사용 | 50.00% |
| last + random start 해제, logits CE 유지 | 50.00% |
| last + alpha2/255, random start·logits CE 유지 | 50.00% |
| last + probability CE, random start 유지 | 50.00% |
| last + random start 해제 + probability CE, alpha.01 | 100.00% |
| last + random start 해제 + probability CE + alpha2/255 | 100.00% |

전체 조합과 분류기 단독 정확도·입력 MSE는 [집계 CSV](../results/livdet_pgd_generation_audit/training8_summary.csv)에 있다. 픽셀 요약은 [pixel_summary.json](../results/livdet_pgd_generation_audit/pixel_summary.json)에 있다.

소량 결과에서는 8bit 양자화만 적용해도 방어 성능이 복구되지 않았다. 반면 probability CE와 random start 해제를 함께 적용하면 같은 표본의 방어 정확도가 달라졌다. 따라서 단순 전처리보다 공격 생성 경로의 조합 차이가 주요 조사 단서다. 단일 설정 하나가 전체31.68%p 격차를 만들었다고 확정하지 않는다.

## 과거 코드에서 확인한 내용과 한계

`PGD-Copy1.ipynb`, `PGD-Copy2.ipynb`, `PGD-CMFK.ipynb`에는 random start 없이 마지막 반복을 반환하는 공격 코드가 있고 alpha2/255를 사용한다. 모델의 Softmax 출력을 제거하지 않고 CrossEntropyLoss에 넣는 방식이다. 현재 코드는 terminal Softmax를 제거한 logits CE, random start, 성공한 중간 후보 보존을 사용한다.

그러나 이 노트북들의 센서·입력 폴더·epsilon·iteration 값은 서로 다르며 일부는 CrossMatch용이다. 현재 남은 코드가 `PGD_0.03/total_valid`의 정확한 생성 버전이라는 연결 증거는 찾지 못했다. 위의 probability CE + no-random 설정은 **과거 방식의 진단 가설**이지 복구가 완료된 원래 generator가 아니다.

이진분류에서 두 loss의 이상적인 gradient 방향은 일치할 수 있지만, 실제 float 계산과 반복 최적화의 결과까지 동일하다고 가정하지 않는다. 초기8장의 gradient 검사에서는 loss별 gradient 크기가 달랐고 sign 차이는 매우 작았다. 그 사실만으로 최종 공격 경로의 차이를 한 원인에 귀속시키지는 않는다.

## 다음 확인과 실행

학습 또는 validation 측 표본 수를 늘려 같은 고정 진단을 수행한다. 논문 최종 평가 정확도를 맞추려고 옵션을 고르지 않는다. 저장 공격을 정확히 만든 코드·실행 로그가 없으면 non-adaptive reference와 fresh attack 결과를 별도 조건으로 보고한다. Adaptive 평가는 이번 명시된 threat model의 결과로 그대로 보존한다.

```bash
conda activate n2n
source .env
python -u reproduction/audit_livdet_pgd_generation.py \
  --device cuda --sample-size 32 --steps 40
```

이 코드는 최종 평가를 공격하지 않고, 학습 측의 정해진 표본에서 생성 옵션과 입력 양자화를 비교한다. `legacy_hypothesis`를 논문 최종 설정으로 단정하거나 adaptive 평가에 더 약한 loss를 자동 적용하지 않는다. Batch64와1의 차이를 분리하려면 동일 per-ID 시작점을 고정한 별도 비교가 필요하다.
