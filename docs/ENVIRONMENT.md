# Conda 가상환경

명령은 저장소 루트에서 실행한다. 검증에 사용한 구성은 Linux, Python 3.11, NVIDIA GPU 환경이다. N2N/DIP와 기존 Keras AT 모델은 서로 다른 환경에서 실행한다.

## N2N·DIP: n2n

```bash
conda env create -f reproduction/environment.yml
conda activate n2n
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda); print('GPU available:', torch.cuda.is_available())"
```

핵심 버전은 PyTorch 2.7.1, torchvision 0.22.1, onnx2torch 1.5.3, NumPy 2.4.6이다. 설치 목록은 `environment.yml`, 기존 환경의 패키지 기록은 `requirements.lock.txt`에 있다.

GPU 확인에서 `True`가 나오는지 확인한다. 평가 스크립트의 기본 장치는 CUDA다. GPU 없이 소량 동작을 검사하려면 평가 명령에 `--device cpu`를 지정한다.

## 기존 AT 모델: n2n_tf

AT 평가를 실행할 때만 생성한다.

```bash
conda env create -f reproduction/environment_tf.yml
conda activate n2n_tf
python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__); print('GPUs:', tf.config.list_physical_devices('GPU'))"
```

TensorFlow 2.15.1, Keras 2.15.0, NumPy 1.26.4를 사용한다. `tensorflow[and-cuda]`를 통해 GPU 런타임 의존성을 설치하며 NVIDIA 드라이버는 호스트에 필요하다. 이 환경에 PyTorch용 requirements를 함께 설치하지 않는다.

가중치와 데이터 경로를 연결한 뒤 실제 모델 로딩을 확인할 수 있다:

```bash
source .env
python reproduction/verify_n2n_tf.py --device gpu --output /tmp/n2n_tf_verification.json
```

## 데이터·가중치 연결

가중치는 [Google Drive 안내](ARTIFACTS.md)를 통해 다운로드하고, 이미지는 [데이터 요청 안내](DATA.md)를 통해 제공받는다. 처음 설치하며 `.env`가 없을 때만 예제를 복사한다.

```bash
test -f .env || cp .env.example .env
```

`.env`를 각자의 위치에 맞게 수정한다:

```bash
export N2N_ASSET_ROOT=/path/to/N2N_assets
export N2N_PM_ROOT=/path/to/N2N_data/PM
```

이후 실행할 터미널에서 활성화하고 읽는다:

```bash
conda activate n2n
source .env
python scripts/check_assets.py --verify-hashes
python scripts/check_data.py
```

환경이 이미 생성돼 있으면 `conda env create`를 다시 실행하지 않는다. `conda env list`로 확인한 뒤 사용할 환경을 활성화한다. 가중치·이미지·실제 `.env`는 Git에 포함하지 않는다.
