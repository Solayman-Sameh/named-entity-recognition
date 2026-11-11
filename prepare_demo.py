"""Prepare the selected model from local cache, delivery ZIP, or Hugging Face."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parent


def prepare_model():
    target = ROOT / 'artifacts' / 'selected_model'
    marker = target / '.bundle_sha256'
    required = ['export.json', 'vocab.json', 'model/config.json',
                'model/model.safetensors', 'model/tokenizer.json']
    if marker.exists() and all((target / name).is_file() for name in required):
        return target

    archive = ROOT / 'codex-space' / 'ner_gradio_app.zip'
    if archive.is_file():
        expected = json.loads((archive.parent / 'archive_manifest.json').read_text(encoding='utf-8'))[archive.name]['sha256']
        with archive.open('rb') as source:
            digest = hashlib.file_digest(source, 'sha256').hexdigest()
        if digest != expected:
            raise SystemExit('The app ZIP does not match archive_manifest.json; restore the matching saved outputs.')
        print('Preparing the existing trained model locally from archive...', flush=True)
        target.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as bundle:
            for item in bundle.infolist():
                if not item.filename.startswith('selected_model/') or item.is_dir():
                    continue
                relative = item.filename.removeprefix('selected_model/')
                destination = (target / relative).resolve()
                if not destination.is_relative_to(target.resolve()):
                    raise ValueError('Unsafe model archive path')
                destination.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(item) as source, destination.open('wb') as output:
                    shutil.copyfileobj(source, output)
        if not all((target / name).is_file() for name in required):
            raise SystemExit('The app ZIP is missing required model assets.')
        marker.write_text(digest, encoding='ascii')
        return target

    # Automatically download from Hugging Face Hub if weights are not present
    print('Selected model not found locally. Downloading from Hugging Face Hub...', flush=True)
    scripts_dir = ROOT / 'scripts'
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from download_model import download_selected_model
    return download_selected_model(target)


if __name__ == '__main__':
    print(prepare_model())
