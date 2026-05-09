"""Download RAVDESS audio-only data from Zenodo."""

import os
import zipfile
import urllib.request
import sys

BASE_URL = "https://zenodo.org/records/1188976/files"
FILES = {
    "Audio_Speech_Actors_01-24.zip": "Speech audio (1,440 files, ~215 MB)",
    "Audio_Song_Actors_01-24.zip": "Song audio (1,012 files, ~198 MB)",
}

DATA_DIR = "data"


def download_file(url, dest_path):
    """Download a file with progress reporting."""
    print(f"Downloading: {url}")
    print(f"  -> {dest_path}")

    def reporthook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 / total_size)
            mb_down = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            sys.stdout.write(f"\r  Progress: {pct:.1f}% ({mb_down:.1f}/{mb_total:.1f} MB)")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, dest_path, reporthook=reporthook)
    print()  # newline after progress


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    for filename, desc in FILES.items():
        url = f"{BASE_URL}/{filename}"
        zip_path = os.path.join(DATA_DIR, filename)

        if not os.path.exists(zip_path):
            print(f"\n--- {desc} ---")
            download_file(url, zip_path)
        else:
            print(f"Already downloaded: {zip_path}")

        # Extract
        print(f"Extracting {filename}...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(DATA_DIR)
        print(f"  Done. Files extracted to {DATA_DIR}/")

    # Count extracted wav files
    wav_count = 0
    for root, dirs, files in os.walk(DATA_DIR):
        wav_count += sum(1 for f in files if f.endswith(".wav"))
    print(f"\nTotal .wav files in {DATA_DIR}/: {wav_count}")


if __name__ == "__main__":
    main()
