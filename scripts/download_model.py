"""Download and verify the winning DistilBERT model weights from Hugging Face Hub.

GitHub rejects files larger than 100 MB, so the 260.8 MB DistilBERT safetensors
weights are excluded from Git. This utility downloads the verified model weights
and assets from Hugging Face Hub (Solayman-Sameh/named-entity-recognition)
and verifies their SHA-256 checksums.
"""
from pathlib import Path
import argparse
import hashlib
import os
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGET_DIR = ROOT / "artifacts" / "selected_model"

HF_REPO_ID = "Solayman-Sameh/named-entity-recognition"
HF_BASE_URL = os.environ.get(
    "MODEL_SOURCE_URL",
    f"https://huggingface.co/{HF_REPO_ID}/resolve/main",
)

# Relative target file path -> Hugging Face repository filename
MODEL_FILES = {
    "export.json": "export.json",
    "vocab.json": "vocab.json",
    "model/config.json": "config.json",
    "model/model.safetensors": "model.safetensors",
    "model/special_tokens_map.json": "special_tokens_map.json",
    "model/tokenizer.json": "tokenizer.json",
    "model/tokenizer_config.json": "tokenizer_config.json",
    "model/vocab.txt": "vocab.txt",
}

# Verified SHA-256 checksums
EXPECTED_SHA256 = {
    "export.json": "d9a3016c1ec89a400ce1c0cd97ffa6aad836103ea8acdd49e623d8afbcdbbacf",
    "vocab.json": "a77ca0d7ca205c73b8f439c8c7fef81777b3318c79a0c9ccd45b3bf39d534123",
    "model/config.json": "28a252ba852188c30d08f6f764a802ccef1d29ae8d740dfd6eb5896cca08f198",
    "model/model.safetensors": "ac4bce6775484f0c4f38d58031dc92e291e50130c0f39f032be3edbebe8fbbf4",
    "model/special_tokens_map.json": "b6d346be366a7d1d48332dbc9fdf3bf8960b5d879522b7799ddba59e76237ee3",
    "model/tokenizer.json": "343989712a36cd8b253efeaf8baf6a08b9d2583f78e395e83832e8ee9f8d8ee1",
    "model/tokenizer_config.json": "91955163d30880d1ef0928cfdd729baaf413401dc4456f3ab1a48b3eec2ec624",
    "model/vocab.txt": "eeaa9875b23b04b4c54ef759d03db9d1ba1554838f8fb26c5d96fa551df93d02",
}


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def download_file_with_progress(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(".tmp")

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ner-downloader/1.0"})
        with urllib.request.urlopen(req) as response, temp_path.open("wb") as out_file:
            total = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 1024 * 1024  # 1 MB chunks
            while chunk := response.read(chunk_size):
                out_file.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    percent = (downloaded / total) * 100
                    mb = downloaded / (1024 * 1024)
                    total_mb = total / (1024 * 1024)
                    sys.stdout.write(f"\r  {destination.name}: {mb:.1f} MB / {total_mb:.1f} MB ({percent:.1f}%)")
                    sys.stdout.flush()
                else:
                    mb = downloaded / (1024 * 1024)
                    sys.stdout.write(f"\r  {destination.name}: {mb:.1f} MB")
                    sys.stdout.flush()
        if total > 0:
            sys.stdout.write("\n")
            sys.stdout.flush()
        temp_path.replace(destination)
    except Exception as exc:
        if temp_path.exists():
            temp_path.unlink()
        raise RuntimeError(f"Download failed for {url}: {exc}") from exc


def download_selected_model(target_dir: Path = DEFAULT_TARGET_DIR, force: bool = False) -> Path:
    target_dir = Path(target_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    base_url = HF_BASE_URL.rstrip("/")

    print(f"Checking model weights in {target_dir}...")
    needs_download = []

    for rel_path, hf_name in MODEL_FILES.items():
        dest = target_dir / rel_path
        expected_hash = EXPECTED_SHA256[rel_path]
        if not force and dest.is_file():
            actual_hash = compute_sha256(dest)
            if actual_hash == expected_hash:
                continue
            print(f"Checksum mismatch for {rel_path}, will re-download.")
        needs_download.append((rel_path, hf_name, dest, expected_hash))

    if not needs_download:
        print("All model weights and assets are present and verified.")
        return target_dir

    print(f"Downloading {len(needs_download)} asset(s) from {base_url}...")
    for rel_path, hf_name, dest, expected_hash in needs_download:
        url = f"{base_url}/{hf_name}"
        download_file_with_progress(url, dest)
        actual_hash = compute_sha256(dest)
        if actual_hash != expected_hash:
            dest.unlink(missing_ok=True)
            raise ValueError(f"Integrity check failed for {rel_path}: expected {expected_hash}, got {actual_hash}")

    # Write marker for compatibility
    marker = target_dir / ".bundle_sha256"
    marker.write_text(EXPECTED_SHA256["model/model.safetensors"], encoding="ascii")
    print("All model weights successfully downloaded and verified.")
    return target_dir


def main():
    parser = argparse.ArgumentParser(description="Download model weights from Hugging Face Hub.")
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET_DIR, help="Target directory")
    parser.add_argument("--force", action="store_true", help="Force re-download even if files exist")
    args = parser.parse_args()

    download_selected_model(args.target, force=args.force)
    print("The application is now ready to run: python app.py")


if __name__ == "__main__":
    main()
