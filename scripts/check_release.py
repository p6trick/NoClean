"""Keep generated binaries, datasets and workstation paths out of the source release."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = {'.pt', '.pth', '.h5', '.hdf5', '.onnx', '.ckpt', '.pdf', '.png', '.jpg', '.zip', '.npz', '.npy'}


def main():
    errors = []
    for path in ROOT.rglob('*'):
        if not path.is_file() or any(part in {'.git', '__pycache__', 'artifacts', 'data', '.venv'} for part in path.relative_to(ROOT).parts):
            continue
        relative = path.relative_to(ROOT)
        if relative.parts[:2] == ('reproduction', 'runs'):
            continue  # Local experiment outputs are excluded from Git and the source ZIP.
        if path.suffix.lower() in FORBIDDEN or path.stat().st_size > 10 * 1024 * 1024:
            errors.append(f'Large/private artifact: {relative}')
        if path.suffix in {'.py', '.json', '.md', '.csv', '.sh', '.txt', '.yml', '.example'}:
            text = path.read_text()
            if re.search(r'/(?:home/[^/\s]+/workspace|mnt/aislabNAS|hdd1/PM)', text):
                errors.append(f'Workstation path: {relative}')
            if ('-----BEGIN ' + 'PRIVATE KEY-----') in text:
                errors.append(f'Private key: {relative}')
    if errors:
        raise SystemExit('\n'.join(errors))
    print('PASS source release contains no model/data binaries, large files, or workstation paths')


if __name__ == '__main__':
    main()
