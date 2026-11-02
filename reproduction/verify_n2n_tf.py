"""Check n2n_tf imports and inference on two trusted archived Keras AT models."""
from paths import CHECKPOINT_ROOT, AT_MODEL_ROOT, loads_json
import argparse
import gc
import json
import os
import platform
from pathlib import Path


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['auto', 'cpu', 'gpu'], default='auto')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--output', type=Path, default=here / 'n2n_tf_verification.json')
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('threads must be positive')
    if args.device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    import numpy as np
    import h5py
    import keras
    import tensorflow as tf
    from PIL import Image

    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if args.device == 'gpu' and not gpus:
        raise RuntimeError('TensorFlow GPU is not visible in this process')
    report = {'python': platform.python_version(), 'tensorflow': tf.__version__,
              'keras': keras.__version__, 'numpy': np.__version__, 'h5py': h5py.__version__,
              'device_requested': args.device, 'visible_gpus': [str(g) for g in gpus],
              'build_info': tf.sysconfig.get_build_info(), 'model_checks': []}
    for dataset, filename in [('livdet', 'GB_FGSM.h5'), ('mnist', 'MNIST_FGSM.h5')]:
        path = AT_MODEL_ROOT / filename
        with h5py.File(path, 'r') as archive:
            original_keras = archive.attrs.get('keras_version', '')
            if isinstance(original_keras, bytes):
                original_keras = original_keras.decode()
        manifest = loads_json((CHECKPOINT_ROOT / dataset / 'fgsm' /
                               'source_data_manifest.json').read_text())
        entry = (manifest['evaluation'] if isinstance(manifest, dict) else manifest)[0]
        model = tf.keras.models.load_model(path, compile=False)
        _, height, width, channels = model.input_shape
        with Image.open(entry['attack_path']) as img:
            img = img.convert('L').resize((width, height), Image.Resampling.BILINEAR)
            array = np.asarray(img, dtype=np.float32)[None, :, :, None] / 255
        if channels != 1:
            raise ValueError(f'Unexpected input channel count: {model.input_shape}')
        output_tensor = model(array, training=False)
        result = np.asarray(output_tensor)
        expected = 2 if dataset == 'livdet' else 10
        if result.shape != (1, expected) or not np.isfinite(result).all():
            raise ValueError(f'Invalid output: {path}, {result.shape}')
        if args.device == 'gpu' and 'GPU:' not in output_tensor.device.upper():
            raise RuntimeError(f'Output was not placed on GPU: {output_tensor.device}')
        record = {'dataset': dataset, 'path': str(path), 'saved_keras_version': str(original_keras),
                  'input_shape': list(model.input_shape), 'output_shape': list(result.shape),
                  'parameter_count': model.count_params(), 'sample_id': entry['id'],
                  'sample_label': entry['label'], 'predicted_label': int(result.argmax(1)[0]),
                  'output_sum': float(result.sum()), 'inference_device': output_tensor.device,
                  'weights_device': model.weights[0].device,
                  'compile': False, 'status': 'loaded_and_inferred'}
        report['model_checks'].append(record)
        print(json.dumps(record), flush=True)
        tf.keras.backend.clear_session()
        del model
        gc.collect()
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + '\n')
    print('VERIFIED', args.output, flush=True)


if __name__ == '__main__':
    main()
