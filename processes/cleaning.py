from glob import glob
import json
import os
import logging
from pathlib import Path
import re
import typer
from dotenv import load_dotenv
from tqdm import tqdm
from string import punctuation

punctuation = punctuation.replace("'", "")

load_dotenv()

logging.basicConfig(level=logging.INFO)

LOCAL_DATA_DIR = os.environ.get("LOCAL_DATA_DIR", f"{Path.home()}/.cache/audios")
LOCAL_DONE_DIR = os.environ.get("LOCAL_DONE_DIR", f"{Path.home()}/.cache/done")
LOCAL_SEGMENTS_DIR = os.environ.get(
    "LOCAL_SEGMENTS_DIR", f"{Path.home()}/.cache/segments"
)
LOCAL_LS_TASKS_DIR = os.environ.get("LOCAL_LS_TASKS_DIR", f"{Path.home()}/.cache/ls")

app = typer.Typer()


@app.command()
def clean_transcripts():
    logging.info("Searching texts in %s", LOCAL_DATA_DIR)
    txts = [f for f in glob(os.path.join(LOCAL_DATA_DIR, "*txt"))]
    logging.info("found %i texts", len(txts))
    for t in tqdm(txts):
        with open(t, "r") as f:
            trans = f.read().split("/n")
        trans = [re.sub(r"\[.*?\]", "", text).lower() for text in trans]
        trans = [i.translate(str.maketrans("", "", punctuation)) for i in trans]
        trans = " ".join([" ".join(i.split()) for i in trans])
        with open(t, "w") as f:
            f.write(trans)


@app.command()
def prepare_for_ls():
    files_path = os.path.join(LOCAL_SEGMENTS_DIR, "*.wav")
    logging.info("scanning %s", files_path)
    files = glob(files_path)
    logging.info("found %i segments to move to %s", len(files), LOCAL_LS_TASKS_DIR)
    Path(LOCAL_LS_TASKS_DIR).mkdir(parents=True, exist_ok=True)
    for file in files:
        f_name = Path(file).stem
        parent = file.split("/")[-2]
        label = f_name.split("_")[0]
        audio = f"data/local-files?d={parent}/{f_name}.wav"
        with open(os.path.join(LOCAL_LS_TASKS_DIR, f"{f_name}.json"), "w") as f:
            json.dump({"data": {"audio": audio, "label": label}}, f)


if __name__ == "__main__":
    app()
