"""Read-only diagnostic of archived MNIST FGSM inputs and target candidates."""
from paths import CHECKPOINT_ROOT
import argparse
import random
from pathlib import Path

import torch

from run import PM, image_batches, index_images, load_classifier, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--sample-size', type=int, default=256)
    args = parser.parse_args()
    torch.set_num_threads(4)
    entries = index_images(PM / 'PGD/MNIST/data', PM / 'FGSM/image_names.txt')
    model = load_classifier(CHECKPOINT_ROOT / 'mnist/classifier.pth', torch.device('cpu'))
    base = PM / 'Self2Self/noise2noise/temporary/data/MNIST'
    report = {'note': 'Target candidates inspected on training IDs only. No training performed.'}
    training = entries[5000:].copy()
    random.Random(2026).shuffle(training)
    training = training[:args.sample_size]
    report['training_ids'] = [e['id'] for e in training]
    variants = {'attack': base / 'FGSM_train(attack)', 'train_dep': base / 'FGSM_train(dep)',
                **{v: PM / 'PGD/MNIST/FGSM' / v
                   for v in ['FGSM_DEP_PGD_003', 'FGSM_DEP_PGD_015', 'FGSM_DEP_PGD']}}
    report['training_sample'] = {}
    for name, directory in variants.items():
        selected = [dict(e, path=str(directory / e['id'])) for e in training]
        count = correct = 0
        for images, labels in image_batches(selected, 64, torch.device('cpu')):
            with torch.no_grad():
                correct += model(images).argmax(1).eq(labels).sum().item()
            count += len(labels)
        report['training_sample'][name] = {'directory': str(directory), 'count': count,
                                           'correct': correct, 'accuracy_percent': 100 * correct / count}
        print(name, report['training_sample'][name], flush=True)
    report['evaluation'] = {}
    for name, directory in [('clean', None), ('saved_attack', base / 'real_FGSM_valid')]:
        selected = entries[:5000] if directory is None else [
            dict(e, path=str(directory / e['id'])) for e in entries[:5000]]
        count = correct = 0
        for images, labels in image_batches(selected, 64, torch.device('cpu')):
            with torch.no_grad():
                correct += model(images).argmax(1).eq(labels).sum().item()
            count += len(labels)
        report['evaluation'][name] = {'count': count, 'correct': correct,
                                      'accuracy_percent': 100 * correct / count}
        print(name, report['evaluation'][name], flush=True)
    write_json(Path(args.output), report)


if __name__ == '__main__':
    main()
