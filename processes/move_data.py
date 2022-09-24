import os
import logging
from pathlib import Path
import typer
from dotenv import load_dotenv
from audiotools.utils import move_files

load_dotenv()

logging.basicConfig(level=logging.INFO)

LOCAL_DATA_DIR = os.environ.get("LOCAL_DATA_DIR", f"{Path.home()}/.cache/audios")


def main(audios_path: str, txt_path: str):
    logging.info("moving to %s", LOCAL_DATA_DIR)
    move_files(audios_path, txt_path, LOCAL_DATA_DIR)


if __name__ == "__main__":
    typer.run(main)
