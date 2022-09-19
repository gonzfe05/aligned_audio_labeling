import os
import shutil
import sys
import logging
from glob import glob
from pathlib import Path
from audiotools.audio_processor import align_audio
from audiotools.utils import export_audio

LOCAL_RAW_DIR = os.environ.get('LOCAL_DATA_DIR', f'{Path.home()}/.cache/audios')
LOCAL_SEGMENTS_DIR = os.environ.get('LOCAL_SEGMENTS_DIR', f'{Path.home()}/.cache/segments')
LOCAL_DONE_DIR = os.environ.get('LOCAL_DONE_DIR', f'{Path.home()}/.cache/done')
LOCAL_FAIL_DIR = os.environ.get('LOCAL_FAIL_DIR', f'{Path.home()}/.cache/fail')



if __name__ == "__main__":
    logging.info("watching %s", LOCAL_RAW_DIR)
    while True:
        files = glob(os.path.join(LOCAL_RAW_DIR, '*.wav'))
        if len(files) > 0:
            logging.info("%i files found, starting alignment", len(files))
            aligned_audios = align_audio(LOCAL_RAW_DIR)
            for f, (audio, label) in aligned_audios.items():
                f_path = os.path.join(LOCAL_SEGMENTS_DIR, label, f)
                Path(f_path).mkdir(parents=True, exist_ok=True)
                logging.info("exporting to %s", f_path)
                export_audio(audio, f_path)
                logging.info("moving done file to %s", LOCAL_DONE_DIR)
                shutil.move(os.path.join(LOCAL_RAW_DIR, f, '.wav'), os.path.join(LOCAL_DONE_DIR, f, '.wav'))
                shutil.move(os.path.join(LOCAL_RAW_DIR, f, '.txt'), os.path.join(LOCAL_DONE_DIR, f, '.txt'))
        files = glob(os.path.join(LOCAL_RAW_DIR, '*.wav'))
        if len(files) > 0:
            logging.info("%i files failed", len(files))
            for f in files:
                logging.info("moving failed file to %s", f)
                shutil.move(os.path.join(LOCAL_RAW_DIR, f, '.wav'), os.path.join(LOCAL_FAIL_DIR, f, '.wav'))
                shutil.move(os.path.join(LOCAL_RAW_DIR, f, '.txt'), os.path.join(LOCAL_FAIL_DIR, f, '.txt'))
        sys.sleep(5)


