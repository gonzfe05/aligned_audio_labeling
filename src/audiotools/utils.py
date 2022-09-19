import subprocess
from glob import glob
import os
from pathlib import Path
from typing import List
from pydub import AudioSegment
from praatio import textgrid


def export_audio(segment: AudioSegment, path: str):
    return segment.export(f'{path}.wav', format="wav")

def force_align(corpus_directory: str, output_directory: str):
    dictionary = 'english_us_arpa'
    acoustic_model = 'english_us_arpa'
    return subprocess.run(["mfa", "align", corpus_directory, dictionary, acoustic_model, output_directory], capture_output=True, check=True)

def parse_textgrid_items(items: List[tuple], duration_seconds):
    for ix, (k, v) in enumerate(items):
        if 'phones' in k:
            continue
        for s, e, t in v.entryList:
            start = max(s-0.1, 0)*1000
            end = min(e+0.1, duration_seconds)*1000
            yield start, end, t 

def parse_textgrid(grid: str, corpus_directory: str):
    name = Path(grid).stem
    tg = textgrid.openTextgrid(grid, includeEmptyIntervals=False)
    audio = AudioSegment.from_wav(os.path.join(corpus_directory, f"{name}.wav"))
    parsed = [(s, e, t) for s, e, t in parse_textgrid_items(tg.tierDict.items(), audio.duration_seconds)]
    audios = [audio[start:end] for start, end, _ in parsed]
    transcriptions = [trans for _, _, trans in parsed]
    return audios, transcriptions

def parse_textgrids(corpus_directory: str, output_directory: str):
    grids = glob(os.path.join(output_directory, '*.TextGrid'))
    wavs = [Path(f).stem for f in glob(os.path.join(corpus_directory, '*.wav'))]
    grids = [f for f in grids if Path(f).stem in wavs]
    audios = []
    transcriptions = []
    files = []
    for grid in grids:
        audios, trans = parse_textgrid(grid, corpus_directory)
        audios.extend(audios)
        transcriptions.extend(trans)
        files.append(Path(grid).stem)
    return audios, transcriptions, files