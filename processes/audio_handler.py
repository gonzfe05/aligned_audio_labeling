import os
import shutil
import logging
from glob import glob
from pathlib import Path
import time
from pydub import AudioSegment
from dotenv import load_dotenv
import typer
from audiotools.audio_processor import align_audio
from audiotools.utils import export_audio

load_dotenv()

logging.basicConfig(level=logging.INFO)


LOCAL_DATA_DIR = os.environ.get("LOCAL_DATA_DIR", f"{Path.home()}/.cache/audios")
LOCAL_SEGMENTS_DIR = os.environ.get(
    "LOCAL_SEGMENTS_DIR", f"{Path.home()}/.cache/segments"
)
LOCAL_DONE_DIR = os.environ.get("LOCAL_DONE_DIR", f"{Path.home()}/.cache/done")
LOCAL_FAIL_DIR = os.environ.get("LOCAL_FAIL_DIR", f"{Path.home()}/.cache/fail")

Path(LOCAL_DONE_DIR).mkdir(parents=True, exist_ok=True)
Path(LOCAL_FAIL_DIR).mkdir(parents=True, exist_ok=True)
Path(LOCAL_SEGMENTS_DIR).mkdir(parents=True, exist_ok=True)


def main():
    start = time.time()
    logging.info("scanning fow wav in %s", LOCAL_DATA_DIR)
    files = glob(os.path.join(LOCAL_DATA_DIR, "*.wav"))
    logging.info("%i files found", len(files))
    N = len(files)
    if N > 0:
        logging.info("starting alignment")
        aligned_audios = align_audio(LOCAL_DATA_DIR)
        logging.info("exporting segments to %s", LOCAL_SEGMENTS_DIR)
        logging.info("moving done files to %s", LOCAL_DONE_DIR)
        for f, (audio, label) in aligned_audios.items():
            export_audio(audio, os.path.join(LOCAL_SEGMENTS_DIR, f"{label}_{f}"))
            shutil.move(
                os.path.join(LOCAL_DATA_DIR, f"{f}.wav"),
                os.path.join(LOCAL_DONE_DIR, f"{f}.wav"),
            )
            shutil.move(
                os.path.join(LOCAL_DATA_DIR, f"{f}.txt"),
                os.path.join(LOCAL_DONE_DIR, f"{f}.txt"),
            )
    files = glob(os.path.join(LOCAL_DATA_DIR, "*.wav"))
    if len(files) > 0:
        logging.info("%i(%f) files failed", len(files), len(files) / N)
        logging.info("moving failed files to %s", LOCAL_FAIL_DIR)
        for f in files:
            shutil.move(
                os.path.join(LOCAL_DATA_DIR, f),
                os.path.join(LOCAL_FAIL_DIR, f"{Path(f).stem}.wav"),
            )
            shutil.move(
                os.path.join(LOCAL_DATA_DIR, f"{Path(f).stem}.txt"),
                os.path.join(LOCAL_FAIL_DIR, f"{Path(f).stem}.txt"),
            )
    logging.info("Finished: %i seconds", int(time.time() - start))


if __name__ == "__main__":
    typer.run(main)
