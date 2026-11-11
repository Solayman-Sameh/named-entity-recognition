"""Refresh the single README tree from Git's shareable files, without downloads."""
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    files = sorted(set(filter(None, subprocess.check_output(
        ['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z'],
        cwd=ROOT).decode('utf-8').split('\0'))))
    tree = {}
    for filename in files:
        node = tree
        for part in filename.split('/'):
            node = node.setdefault(part, {})
    descriptions = {
        '.gitattributes': 'Line endings and Git attribute rules',
        '.github': 'GitHub configuration and automated workflows',
        '.github/workflows': 'Continuous integration workflows',
        '.github/workflows/checks.yml': 'Offline checks and verified tagged releases',
        '.gitignore': 'Excludes weights, caches, secrets and drafts',
        'LICENSE': 'MIT license',
        'README.md': 'Overview, results, usage, and links',
        'app.py': 'Root demo launcher',
        'prepare_demo.py': 'Model preparation and auto-download',
        'requirements-ci.txt': 'Lightweight automated-check dependency',
        'requirements-demo.txt': 'Local demo dependencies',
        'docs': 'Planning, verification and release notes',
        'docs/Cloud_Work_Prompt.md': 'Kaggle GPU training handoff prompt',
        'docs/Demo_Setup_2026-11-10.json': 'Demo bundle checksum manifest',
        'docs/Inference_Update_2026-11-10.json': 'Raw-text tokenizer and edge-case fixes',
        'docs/Project_Proposal.md': 'Original project proposal and architecture plan',
        'docs/Release_Readiness_2026-11-10.json': 'Audit checklist and verification record',
        'docs/Release_Readiness_2026-11-11.json': 'Final release readiness checklist',
        'docs/UI_Update_2026-11-10.json': 'Gradio interface and styling adjustments',
        'docs/releases': 'Versioned release notes',
        'docs/releases/v1.0.0.md': 'v1.0.0 release notes',
        'media': 'App preview and finished demo video',
        'media/README.md': 'Media assets overview and notes',
        'media/app-preview.png': 'Interface preview screenshot',
        'media/demo.mp4': 'Finished 25-second demo video',
        'notebooks': 'Current training workflow and original executed run',
        'notebooks/named_entity_recognition_all_in_one.ipynb': 'Self-contained Kaggle training workflow',
        'notebooks/named_entity_recognition_executed.ipynb': 'Historical executed training record with outputs',
        'reports': 'Small results and figures',
        'reports/REPORT.md': 'Model evaluation and comparison report',
        'reports/comparison.json': 'Strict seqeval F1 metrics for all models',
        'reports/crf_test_comparison.json': 'BiLSTM vs BiLSTM-CRF boundary transition diffs',
        'reports/eda.png': 'Entity distribution exploratory analysis chart',
        'reports/learning_curves.png': 'Loss and validation F1 curves across epochs',
        'scripts': 'Repository maintenance tools',
        'scripts/check_repository.py': 'Model-free integrity and syntax validator',
        'scripts/download_model.py': 'Hugging Face weights downloader and validator',
        'scripts/update_readme_tree.py': 'Single README tree generator',
        'src': 'Training pipeline and inference implementation',
        'src/app.py': 'Gradio application definition and styling',
        'src/bootstrap.py': 'Kaggle environment setup and dependencies',
        'src/common.py': 'IOB conversion, seqeval metrics and chunking helpers',
        'src/finalize.py': 'Metric recalculation, artifacts bundle and export',
        'src/inference.py': 'Offline CPU inference engine and span extractor',
        'src/models.py': 'PyTorch LSTM, BiLSTM, BiLSTM-CRF and DistilBERT architectures',
        'src/pipeline.py': 'Orchestration for dataset, training, evaluation and export',
        'src/prepare.py': 'CoNLL-2003 download, GloVe extraction and vocabulary build',
        'src/reference.json': 'Baseline metrics and reproducibility fingerprints',
        'src/requirements.txt': 'Pinned Kaggle GPU training dependencies',
        'src/scheduler.py': 'Multi-GPU / single-GPU training job scheduler',
        'src/source_lock.json': 'Pinned external dataset and model URLs',
        'src/verify.py': 'Post-training checkpoint and prediction auditor',
        'src/worker.py': 'Subprocess runner for isolated model training jobs',
        'tests': 'Regression tests',
        'tests/test_text_inference.py': 'Regression tests for tokenization, offsets and boundaries',
    }
    lines = [ROOT.name + '/']

    def walk(node, prefix='', parent=''):
        for index, (name, children) in enumerate(sorted(node.items())):
            last = index == len(node) - 1
            key = (parent + name).rstrip('/')
            description = descriptions.get(key, '')
            lines.append(prefix + ('└── ' if last else '├── ') + name
                         + ('/' if children else '') + ('  # ' + description if description else ''))
            if children:
                walk(children, prefix + ('    ' if last else '│   '), parent + name + '/')

    walk(tree)
    path = ROOT / 'README.md'
    text = path.read_text(encoding='utf-8')
    assert text.count('```text') == 1, 'Expected one project tree'
    start = text.index('```text')
    end = text.index('```', start + 7)
    path.write_text(text[:start] + '```text\n' + '\n'.join(lines) + '\n' + text[end:], encoding='utf-8')
    print(f'Updated README tree with {len(files)} shareable files.')


if __name__ == '__main__':
    main()
