"""Model-free release checks. No downloads, inference or notebook execution."""
import ast
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote, urlsplit

import nbformat

ROOT = Path(__file__).resolve().parents[1]


def repository_files():
    output = subprocess.check_output(
        ['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z'],
        cwd=ROOT,
    ).decode('utf-8')
    return sorted(set(filter(None, output.split('\0'))))


def main():
    files = repository_files()
    required = ['README.md', 'LICENSE', '.gitignore', 'app.py', 'prepare_demo.py',
                'requirements-demo.txt', 'media/app-preview.png', 'media/demo.mp4',
                '.github/workflows/checks.yml', 'docs/releases/v1.0.0.md']
    for name in required:
        assert name in files, f'Missing shareable release file: {name}'
    forbidden = {'.pt', '.pth', '.ckpt', '.safetensors', '.onnx', '.zip'}
    for name in files:
        path = ROOT / name
        assert path.is_file(), f'Missing file: {name}'
        assert not name.startswith(('artifacts/', 'codex-space/', 'review/', '.venv/')), name
        assert path.suffix.lower() not in forbidden, f'Large model/archive included: {name}'
        assert not (path.name == '.env' or path.name.startswith('.env.') and path.name != '.env.example'), name
        assert path.stat().st_size < 10 * 1024 * 1024, f'Unexpected file over 10 MB: {name}'
        if path.suffix == '.py':
            ast.parse(path.read_text(encoding='utf-8'), filename=name)
        elif path.suffix == '.json':
            json.loads(path.read_text(encoding='utf-8'))
        elif path.suffix == '.ipynb':
            notebook = nbformat.read(path, as_version=4)
            nbformat.validate(notebook)
            for cell in notebook.cells:
                if cell.cell_type == 'code':
                    ast.parse(cell.source, filename=name)
        if path.suffix == '.md':
            for target in re.findall(r'\]\(([^)]+)\)', path.read_text(encoding='utf-8')):
                target = target.strip('<>')
                parsed = urlsplit(target)
                if parsed.scheme or not parsed.path:
                    continue
                resolved = (path.parent / unquote(parsed.path)).resolve()
                assert resolved.is_relative_to(ROOT), f'Nonportable link: {name}: {target}'
                assert resolved.exists(), f'Broken local link: {name}: {target}'

    notebook = nbformat.read(ROOT / 'notebooks/named_entity_recognition_all_in_one.ipynb', as_version=4)
    assignment = ast.parse(notebook.cells[3].source).body[0]
    embedded = ast.literal_eval(assignment.value)
    sources = {p.name: p.read_text(encoding='utf-8') for p in (ROOT / 'src').iterdir() if p.is_file()}
    assert embedded == sources, 'Current training notebook and src/ differ'

    readme = (ROOT / 'README.md').read_text(encoding='utf-8')
    trees = re.findall(r'```text\n(.*?)```', readme, flags=re.S)
    assert len(trees) == 1, 'README must have exactly one project tree'
    stack = []
    listed = set()
    for line in trees[0].splitlines():
        match = re.match(r'^([ │]*)(?:├──|└──) (.*?)(?:\s+#.*)?$', line)
        if not match:
            continue
        depth = len(match[1]) // 4
        name = match[2].strip()
        stack = stack[:depth]
        if name.endswith('/'):
            stack.append(name[:-1])
        else:
            listed.add('/'.join(stack + [name]))
    assert listed == set(files), f'README tree mismatch: missing={set(files)-listed}, stale={listed-set(files)}'
    print(f'PASS: {len(files)} shareable files; syntax, JSON, notebooks, source sync, local links and README tree.')


if __name__ == '__main__':
    main()
