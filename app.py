"""Launch the trained demo from the project root: python app.py."""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from prepare_demo import prepare_model


def main():
    model_dir = prepare_model()
    from app import build_app, CUSTOM_CSS
    demo = build_app(model_dir)
    demo.launch(server_name='127.0.0.1', server_port=int(os.environ.get('PORT', '7860')), css=CUSTOM_CSS)


if __name__ == '__main__':
    main()
