from pathlib import Path
import os
import numpy as np
from pydantic import BaseModel
from audiotools.utils import force_align, parse_textgrids



class AudioArray(BaseModel):
    array: np.ndarray
    class Config:
        arbitrary_types_allowed = True

    
def parse_aligned_audio(corpus_directory: str):
    output_directory = os.path.join(Path.home(), '.cache/mfa/')
    mfa_output = force_align(corpus_directory, output_directory)
    if mfa_output.returncode == 1:
        raise ValueError(mfa_output.stderr)
    return parse_textgrids(corpus_directory, output_directory)

def align_audio(corpus_directory: str):
    parsed, transcriptions, files = parse_aligned_audio(corpus_directory)
    parsed = [{'array': np.array(p.get_array_of_samples(), dtype=float)} for p in parsed]
    audios = [AudioArray.parse_obj(p) for p in parsed]
    return {file: (audio, trans) for file, audio, trans in zip(files, audios, transcriptions)}
