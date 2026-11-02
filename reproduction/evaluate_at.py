"""Evaluate trusted legacy Keras AT classifiers; n2n_tf only, no PyTorch imports."""
from paths import CHECKPOINT_ROOT, AT_MODEL_ROOT, loads_json
import argparse
import csv
import gc
import hashlib
import json
import os
import time
from datetime import datetime
from pathlib import Path


ATTACKS = ['fgsm', 'pgd', 'cw', 'ba', 'bim', 'aa']
MATCHED = {'livdet': [99.76, 40.00, 60.47, 90.39, 78.24, 71.03],
           'mnist': [93.23, 94.55, 95.34, 97.46, 93.58, 90.20]}
CROSS = {'livdet': [68.56, 81.14, 62.08, 70.54, 72.49, 76.91],
         'mnist': [88.10, 48.80, 93.18, 95.95, 88.26, 95.18]}
APPENDIX = {
    'livdet': [[None, 88.48, 74.80, 73.60, 97.60, 73.68],
               [40.16, None, 40.00, 59.60, 44.72, 67.59],
               [60.00, 70.63, None, 74.00, 60.79, 78.96],
               [89.92, 89.60, 89.67, None, 90.64, 89.92],
               [86.23, 88.63, 40.95, 74.32, None, 74.40],
               [66.47, 68.40, 64.96, 71.20, 68.72, None]],
    'mnist': [[None, 59.57, 92.19, 95.24, 93.30, 94.49],
              [93.26, None, 91.07, 95.67, 93.33, 93.55],
              [88.34, 41.31, None, 96.52, 88.59, 95.77],
              [88.30, 37.25, 94.81, None, 88.59, 95.80],
              [93.45, 61.95, 94.66, 96.38, None, 96.31],
              [77.16, 43.92, 77.48, 82.94, 77.49, None]],
}


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + '\n')


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def model_filename(dataset, attack, profile):
    if dataset == 'mnist':
        return 'MNIST_BA2_adv.h5' if attack == 'ba' else f'MNIST_{attack.upper()}.h5'
    suffix = '2' if ((profile == 'v2' and attack in ['pgd', 'cw', 'ba']) or
                     (profile == 'recovered' and attack in ['cw', 'ba'])) else ''
    return f'GB_{attack.upper()}{suffix}.h5'


def write_tables(folder, dataset, matrix):
    write_json(folder / 'matrix.json', matrix)
    with (folder / 'matrix.csv').open('w') as stream:
        writer = csv.writer(stream)
        writer.writerow(['AT model / target attack', *[a.upper() for a in ATTACKS]])
        for source in ATTACKS:
            writer.writerow([source.upper(), *[matrix[source].get(t, '') for t in ATTACKS]])
    cells = []
    for i, source in enumerate(ATTACKS):
        for j, target in enumerate(ATTACKS):
            if target not in matrix[source]:
                continue
            paper = MATCHED[dataset][i] if source == target else APPENDIX[dataset][i][j]
            cells.append({'model_label': source.upper(), 'target_attack': target.upper(),
                          'accuracy_percent': matrix[source][target], 'paper_accuracy_percent': paper,
                          'delta_pp': matrix[source][target] - paper, 'diagonal': source == target})
    if cells:
        with (folder / 'cell_comparison.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(cells[0]))
            writer.writeheader()
            writer.writerows(cells)
    if not all(len(row) == 6 for row in matrix.values()):
        return
    matched, cross, row_means = [], [], []
    for i, attack in enumerate(ATTACKS):
        value = matrix[attack][attack]
        matched.append({'attack': attack.upper(), 'accuracy_percent': value,
                        'paper_accuracy_percent': MATCHED[dataset][i], 'delta_pp': value - MATCHED[dataset][i]})
        value = sum(matrix[s][attack] for s in ATTACKS if s != attack) / 5
        cross.append({'target_attack': attack.upper(), 'accuracy_percent': value,
                      'paper_accuracy_percent': CROSS[dataset][i], 'delta_pp': value - CROSS[dataset][i]})
        value = sum(matrix[attack][t] for t in ATTACKS if t != attack) / 5
        paper = sum(x for x in APPENDIX[dataset][i] if x is not None) / 5
        row_means.append({'model_label': attack.upper(), 'mean_accuracy_percent': value,
                          'paper_mean_from_cells_percent': paper, 'delta_pp': value - paper})
    for name, rows in [('matched_comparison', matched), ('cross_comparison', cross),
                       ('appendix_row_means', row_means)]:
        write_json(folder / f'{name}.json', rows)
        with (folder / f'{name}.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', choices=['livdet', 'mnist'], default=['livdet'])
    parser.add_argument('--profile', choices=['v1', 'v2', 'recovered'], default='v2')
    parser.add_argument('--models', nargs='+', choices=ATTACKS, default=ATTACKS,
                        help='Evaluate selected source AT classifiers; all six target attacks are retained')
    parser.add_argument('--device', choices=['gpu', 'cpu'], default='gpu')
    parser.add_argument('--livdet-eval-size', type=int, default=1250)
    parser.add_argument('--mnist-eval-size', type=int, default=5000)
    parser.add_argument('--livdet-batch-size', type=int, default=2)
    parser.add_argument('--mnist-batch-size', type=int, default=32)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--checkpoint-root', type=Path, default=CHECKPOINT_ROOT)
    parser.add_argument('--model-root', type=Path, default=AT_MODEL_ROOT)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not 1 <= args.livdet_eval_size <= 1250 or not 1 <= args.mnist_eval_size <= 5000:
        parser.error('Invalid evaluation sizes')
    if min(args.livdet_batch_size, args.mnist_batch_size, args.threads) < 1:
        parser.error('Batch sizes and threads must be positive')
    if len(set(args.datasets)) != len(args.datasets):
        parser.error('Duplicate dataset')
    if len(set(args.models)) != len(args.models):
        parser.error('Duplicate source model')
    if args.device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    import numpy as np
    import tensorflow as tf
    import keras
    import h5py
    from PIL import Image, ImageFile

    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if args.device == 'gpu' and not gpus:
        raise RuntimeError('TensorFlow GPU unavailable; activate n2n_tf and check GPU visibility')
    output = args.output or here / 'runs' / datetime.now().strftime('at_eval_%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True, exist_ok=False)
    recoveries = {}
    write_json(output / 'config.json', {
        'paper': str(here.parent / 'N2N_Journal.pdf'), 'method': 'legacy AT Keras classifier inference',
        'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'tensorflow': tf.__version__, 'keras': keras.__version__, 'visible_gpus': [str(g) for g in gpus],
        'preprocessing': 'PIL grayscale /255; TF bilinear resize only if LivDet dimensions differ',
        'profile_selection': ('recovered: PGD v1, CW/BA v2; recovered after inspecting paper-score mismatch'
                              if args.profile == 'recovered' else
                              'v1/v2 are distinct filename candidates; v2 follows surviving code references'),
        'limitations': ['Attack labels are inferred from filenames and surviving notebook references.',
                        'Exact paper-final model version is unconfirmed; v1/v2 are distinct candidates.',
                        'Archived attacks target the original classifier, not these AT classifiers.',
                        'No retraining or adaptive attack generation is performed.']})

    def read_image(path, dataset):
        try:
            with Image.open(path) as img:
                array = np.asarray(img.convert('L'), dtype=np.float32)[:, :, None]
        except OSError as exc:
            previous = ImageFile.LOAD_TRUNCATED_IMAGES
            ImageFile.LOAD_TRUNCATED_IMAGES = True
            try:
                with Image.open(path) as img:
                    array = np.asarray(img.convert('L'), dtype=np.float32)[:, :, None]
            finally:
                ImageFile.LOAD_TRUNCATED_IMAGES = previous
            recoveries[path] = str(exc)
            write_json(output / 'image_load_recoveries.json', recoveries)
        expected = 581 if dataset == 'livdet' else 28
        if array.shape != (expected, expected, 1):
            if dataset == 'mnist':
                raise ValueError(f'Invalid MNIST dimensions: {path}, {array.shape}')
            array = tf.image.resize(array, (expected, expected), method='bilinear').numpy()
        return array / 255

    for dataset in args.datasets:
        folder = output / dataset
        folder.mkdir()
        size = args.livdet_eval_size if dataset == 'livdet' else args.mnist_eval_size
        batch_size = args.livdet_batch_size if dataset == 'livdet' else args.mnist_batch_size
        groups = {}
        reference = None
        for target in ATTACKS:
            data = loads_json((args.checkpoint_root / dataset / target / 'source_data_manifest.json').read_text())
            entries = data['evaluation'] if isinstance(data, dict) else data
            ids = {e['id']: (e['label'], e['path']) for e in entries}
            if len(ids) != len(entries) or len(entries) < size:
                raise ValueError('Duplicate/missing evaluation IDs')
            if reference is not None and ids != reference:
                raise ValueError('Attack evaluation IDs/labels differ')
            reference = ids
            groups[target] = entries[:size]
            write_json(folder / f'{target}_manifest.json', groups[target])
        model_paths = {s: args.model_root / model_filename(dataset, s, args.profile) for s in args.models}
        for path in model_paths.values():
            if not path.is_file():
                raise FileNotFoundError(path)
        write_json(folder / 'evaluation_info.json', {'count_per_target': size,
                   'partial_evaluation': size != (1250 if dataset == 'livdet' else 5000),
                   'evaluated_source_models': args.models, 'complete_model_matrix': len(args.models) == 6,
                   'rows': 'AT classifier nominal training attack', 'columns': 'input target attack'})
        matrix = {s: {} for s in ATTACKS}
        model_records = {}
        for source, path in model_paths.items():
            started = time.monotonic()
            model = tf.keras.models.load_model(path, compile=False)
            expected_size = 581 if dataset == 'livdet' else 28
            expected_classes = 2 if dataset == 'livdet' else 10
            if model.input_shape[1:] != (expected_size, expected_size, 1) or model.output_shape[-1] != expected_classes:
                raise ValueError(f'Unexpected model shape: {path}, {model.input_shape}, {model.output_shape}')
            with h5py.File(path, 'r') as archive:
                saved_version = archive.attrs.get('keras_version', '')
                if isinstance(saved_version, bytes):
                    saved_version = saved_version.decode()
            model_records[source] = {'path': str(path), 'sha256': sha256(path),
                                     'input_shape': model.input_shape, 'output_shape': model.output_shape,
                                     'saved_keras_version': str(saved_version), 'parameter_count': model.count_params()}
            write_json(folder / 'models.json', model_records)
            for target, entries in groups.items():
                correct, count, predictions = 0, 0, []
                confusion = np.zeros((expected_classes, expected_classes), dtype=np.int64)
                for start in range(0, len(entries), batch_size):
                    chunk = entries[start:start + batch_size]
                    images = np.stack([read_image(e['attack_path'], dataset) for e in chunk])
                    tensor = model(images, training=False)
                    if args.device == 'gpu' and 'GPU:' not in tensor.device.upper():
                        raise RuntimeError('Classifier output is not placed on GPU')
                    probabilities = np.asarray(tensor)
                    if probabilities.shape != (len(chunk), expected_classes) or not np.isfinite(probabilities).all():
                        raise ValueError(f'Invalid output: {path}')
                    preds = probabilities.argmax(1)
                    for e, pred in zip(chunk, preds):
                        correct += int(pred == e['label'])
                        confusion[e['label'], pred] += 1
                        predictions.append({'id': e['id'], 'label': e['label'], 'predicted': int(pred)})
                    count += len(chunk)
                    if count % (batch_size * 100) == 0 or count == size:
                        print(dataset, source, '->', target, f'{count}/{size}', flush=True)
                prefix = f'{source}_on_{target}'
                write_json(folder / f'{prefix}_predictions.json', predictions)
                write_json(folder / f'{prefix}_metrics.json', {
                    'count': count, 'correct': correct, 'accuracy_percent': 100 * correct / count,
                    'confusion_matrix': confusion.tolist(),
                    'class_order': ['Fake', 'Alive'] if dataset == 'livdet' else list(range(10))})
                matrix[source][target] = 100 * correct / count
                write_tables(folder, dataset, matrix)
            print('MODEL_DONE', dataset, source, f'{time.monotonic() - started:.1f}s', matrix[source], flush=True)
            tf.keras.backend.clear_session()
            del model
            gc.collect()
    print('RESULTS', output, flush=True)


if __name__ == '__main__':
    main()
