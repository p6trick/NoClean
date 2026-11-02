# 실험 명령과 설정

모든 명령은 저장소 루트에서 실행하고 `.env`를 source한다. 공개 코드는 환경변수로 경로를 읽는다.

| 목적 | 환경 | 코드 | 대표설정 |
|---|---|---|---|
| Table1/2 동일공격 | n2n | cross_attack.py --matched-only | LivDet pool2/PNG, MNIST pool1/float |
| Table3/4, A.7/A.8 상세 | n2n | cross_attack.py | 6×6; 대각선제외열평균 |
| Table5 전이 | n2n | table5_transfer.py --modes pool1 | LivDet가중치→AA MNIST; 구조가정미확인 |
| Table6 clean | n2n | table6_clean.py | 같은데이터셋모델, clean입력 |
| AT 기존모델 | n2n_tf | evaluate_at.py --profile recovered | N2N없이AT분류기를직접적용 |
| 저장DIP | n2n | evaluate_saved_dip.py | 4공격의기존PNG |
| DIP 새생성 | n2n | generate_dip_livdet.py | BIM/AA, 이미지별801회 |

## AT

```bash
conda activate n2n_tf
source .env
python -u reproduction/evaluate_at.py --datasets livdet --profile recovered
# 추가 비교용 후보 한 모델만
python -u reproduction/evaluate_at.py --datasets livdet --profile v1 --models pgd
```

`matrix.csv`, `cell_comparison.csv`, `matched_comparison.csv`, `cross_comparison.csv`가 생성된다. 일부 모델만 평가하면 전체 6모델 평균은 생성하지 않는다. MNIST 옵션도 구현돼 있지만 현재 결과 집계에는 LivDet AT만 포함했다.

## DIP 새 생성과 재개

```bash
conda activate n2n
source .env
python -u reproduction/generate_dip_livdet.py --attacks bim aa --eval-size 2
# 위 실행이 출력한 경로 사용. 동일조건 재개:
python -u reproduction/generate_dip_livdet.py --attacks bim aa --eval-size 2 \
  --resume reproduction/runs/dip_regenerated_RUN_ID
# 완료된 이미지 재사용 + 전체 장수로 명시적 확장:
python -u reproduction/generate_dip_livdet.py --attacks bim aa --eval-size 1250 \
  --resume reproduction/runs/dip_regenerated_RUN_ID --extend
```

기존 노트북을 따른 설정은 skip128×5/skip4/input32, reflection/bilinear, Adam lr .01, uniform[0,.1] 초기 입력, Gaussian 입력 노이즈 std 1/30, 801회다. 논문에는 800회라고 기술돼 있어 차이를 config에 기록했다. `--iterations 800`은 별도 대조 조건이다. 원래 seed는 미확인이므로 공격·ID별 seed를 명시적으로 고정한다.

최적화 목표는 공격 이미지 자체이며 clean이나 정답 라벨은 사용하지 않는다. 기존 PSNR backtracking을 유지하고 마지막 noisy 입력의 train-mode 출력을 저장한다. EMA 출력은 사용하지 않는다. 분류는 최적화 완료 후에만 수행한다. config·입력·출력 해시가 맞는 완료 이미지만 재사용한다.

## 후처리

Float는 raw RGB의 가중 회색조 합산 후 [0,1] 제한이다. PNG는 RGB 채널별 제한 후 8bit 절사 및 PIL 회색조 변환이다. 두 경로는 양자화뿐 아니라 제한 순서도 다르다. PNG 압축 자체는 무손실이다. 기존 보고 조건은 MNIST float, LivDet PNG다.

## 재학습

기존 기본 MNIST 실험은 4500학습/500검증/5000평가, 30epoch, batch32다. LivDet 초기 기본은 1125/125/1250, 30epoch, batch4, 128patch다. 이는 원고의 100epoch/batch1/crop 정책과 완전히 동일하다고 확인되지 않았다. 새 목표 생성 옵션도 기존 공격을 유지하고 true-label 보정만 별도로 수행하는 비교 조건이다.
