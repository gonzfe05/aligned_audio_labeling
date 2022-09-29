import os
from pathlib import Path
import json
import logging
from typing import Any, List, Literal, Optional
import numpy as np
from numpy.typing import ArrayLike

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import DATA_UNDEFINED_NAME

from annoy import AnnoyIndex
from transformers import Wav2Vec2ForXVector, AutoFeatureExtractor
from pydantic import BaseModel
import torch
from pydub import AudioSegment


logger = logging.getLogger(__name__)


class AnnoyHandler(object):
    """Wraps Annoy Index interface"""

    def __init__(self, dimentions: int, index_path: Optional[str] = None):
        if not dimentions:
            raise ValueError(f"Need dimentions, got '{dimentions}'")
        self.dimentions = dimentions
        self.index = AnnoyIndex(self.dimentions, "angular")
        self.id2label = {}
        if index_path:
            self.load_index(index_path)
        self.index.set_seed(1991)

    def load_index(self, index_path: str) -> None:
        """Load index alongside its labels dict"""
        self.index.load(index_path)
        dir = Path(index_path).parent
        name = Path(index_path).stem + ".json"
        with open(os.path.join(dir, name), "r") as f:
            self.id2label = json.load(f)
        self.items = len(self.id2label)

    def add_item(self, vector, label) -> None:
        """Update index and label dict"""
        i = len(self.id2label)
        self.id2label[i] = label if label else None
        self.index.add_item(i, vector)

    def build(self, trees: int = 10) -> Literal[True]:
        """Close index"""
        return self.index.build(trees, n_jobs=4)

    def unbuild(self) -> None:
        """Open index"""
        self.index.unbuild()

    def save(self, path: str) -> None:
        """Write index and labels dict to disc"""
        self.index.save(path)
        dir = Path(path).parent
        name = Path(path).stem + ".json"
        with open(os.path.join(dir, name), "w") as f:
            json.dump(self.id2label, f)

    def get_nns(self, vector, n: int) -> tuple:
        """Get nearest neighbours with their scores and labels"""
        indexes, similarities = self.index.get_nns_by_vector(
            vector, n, include_distances=True
        )
        labels = [self.id2label[str(i)] for i in indexes]
        return indexes, similarities, labels


class AudioArray(BaseModel):
    array: np.ndarray

    class Config:
        arbitrary_types_allowed = True


class XvectorModel(object):
    def __init__(
        self, model_checkpoint: str, annoy_index_path: Optional[str] = None
    ) -> None:
        self.model_checkpoint = model_checkpoint
        self.model = Wav2Vec2ForXVector.from_pretrained(self.model_checkpoint)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            self.model_checkpoint
        )
        # Wav2vec emb dimention
        self.embeddings_dimention = 512
        self.annoy_handler = AnnoyHandler(self.embeddings_dimention, annoy_index_path)

    def preprocess_function(
        self, audio_arrays: List[AudioArray], max_duration: float = 1.0
    ) -> Any:
        """preprocess audio array into model input"""
        return self.feature_extractor(
            [a.array for a in audio_arrays],
            sampling_rate=self.feature_extractor.sampling_rate,
            max_length=int(self.feature_extractor.sampling_rate * max_duration),
            truncation=True,
            return_tensors="pt",
            padding=True,
        )

    def get_embeddings(self, inputs: Any):
        """Forward pass of the model"""
        with torch.no_grad():
            result = self.model(**inputs).embeddings
        return result.detach().cpu().numpy()

    def get_predicted_labels(
        self, embedding: ArrayLike, neighbours: int = 10, use_max: bool = False
    ) -> tuple:
        """Use nearest neighbours by cosine distance to get the prediction"""
        _, similarities, labels = self.annoy_handler.get_nns(embedding, neighbours)
        if use_max:
            return labels[0]
        return similarities, labels

    def predict(self, audio_paths: List[str]) -> List[tuple]:
        audios = [AudioSegment.from_wav(audio) for audio in audio_paths]
        parsed = [
            {"array": np.array(p.get_array_of_samples(), dtype=float)} for p in audios
        ]
        parsed = [AudioArray.parse_obj(p) for p in parsed]
        inputs = self.preprocess_function(parsed)
        embs = self.get_embeddings(inputs)
        return [self.get_predicted_labels(e) for e in embs]


class Xvector(LabelStudioMLBase):
    def __init__(self, **kwargs):
        super(Xvector, self).__init__(**kwargs)
        self.model = XvectorModel(kwargs["model-dir"])
        self.from_name, self.to_name, self.value = self._bind_to_choices()

    def predict(self, tasks, **kwargs):
        output = []
        audio_paths = []
        labels = []
        for task in tasks:
            # We used the label as the header of the task, thus it wont show in parsed_label_config
            label = task["data"].get("label")
            audio_url = task["data"].get(self.value) or task["data"].get(
                DATA_UNDEFINED_NAME
            )
            audio_path = self.get_local_path(audio_url)
            audio_paths.append(audio_path)
            labels.append(label)
        predictions = self.model.predict(audio_paths)
        for (pred_label, score), label in zip(predictions, labels):
            if score > 0.7:
                res = "Correct" if pred_label == label else "Incorrect"
            else:
                res = "Uncertain"
            output.append(
                {
                    "result": [
                        {
                            "from_name": self.from_name,
                            "to_name": self.to_name,
                            "type": "choices",
                            "value": {"choices": [res]},
                        }
                    ],
                    "score": score,
                }
            )
        return output

    def fit(self, completions, workdir=None, **kwargs):
        project_path = kwargs.get("project_full_path")
        if project_path and os.path.exists(project_path):
            logger.info("Found project in local path " + project_path)
        else:
            logger.error(
                "Project not found in local path "
                + project_path
                + ". Serving uploaded data will fail."
            )
        return {"project_path": project_path}

    def _bind_to_choices(self):
        from_name, to_name, value = None, None, None
        for tag_name, tag_info in self.parsed_label_config.items():
            if tag_info["type"] == "Choices":
                from_name = tag_name
                if len(tag_info["inputs"]) > 1:
                    logger.warning(
                        "Model works with single Audio or AudioPlus input, "
                        "but {0} found: {1}. We'll use only the first one".format(
                            len(tag_info["inputs"]), ", ".join(tag_info["to_name"])
                        )
                    )
                if tag_info["inputs"][0]["type"] not in ("Audio", "AudioPlus"):
                    raise ValueError(
                        "{0} tag expected to be of type Audio or AudioPlus, but type {1} found".format(
                            tag_info["to_name"][0], tag_info["inputs"][0]["type"]
                        )
                    )
                to_name = tag_info["to_name"][0]
                # Audio key used in the task data json
                value = tag_info["inputs"][0]["value"]
        if from_name is None:
            raise ValueError(
                "Model expects <Choices> tag to be presented in a label config."
            )
        return from_name, to_name, value


if __name__ == "__main__":
    MODEL_DIR = "data/models"
    EXAMPLE_TASK = "data/example_task.json"
    assert os.path.exists(MODEL_DIR), f"Missing {MODEL_DIR}"
    assert os.path.exists(EXAMPLE_TASK), f"Missing {EXAMPLE_TASK}"
    params = {}
    # params['label_config'] = {
    #     "class": {
    #         "type": "Choices",
    #         "to_name": ["audio"],
    #         "inputs": [{"type": "AudioPlus", "value": "$audio"}]
    #     }
    # }
    params[
        "label_config"
    ] = """<View>

    <Header value="$label"/>

   <AudioPlus name="audio" value="$audio"/>

   <Choices name="class" toName="audio" perRegion="false" required="true">
     <Choice value="Correct"/>
     <Choice value="Incorrect"/>
     <Choice value="Uncertain"/>
   </Choices>
 </View>"""
    params["model-dir"] = MODEL_DIR
    model = Xvector(**params)
    assert isinstance(model, Xvector)
    with open(EXAMPLE_TASK, "r") as f:
        annot = json.load(f)
    prediction = model.predict([annot["task"]])
    assert isinstance(prediction, list)


# class NemoASR(LabelStudioMLBase):

#     def __init__(self, model_name='QuartzNet15x5Base-En', **kwargs):
#         super(NemoASR, self).__init__(**kwargs)

#         # Find TextArea control tag and bind ASR model to it
#         self.from_name, self.to_name, self.value = self._bind_to_textarea()

#         # This line will download pre-trained QuartzNet15x5 model from NVIDIA's NGC cloud and instantiate it for you
#         self.model = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name=model_name)

#     def predict(self, tasks, **kwargs):
#         output = []
#         audio_paths = []
#         for task in tasks:
#             audio_url = task['data'].get(self.value) or task['data'].get(DATA_UNDEFINED_NAME)
#             audio_path = self.get_local_path(audio_url)
#             audio_paths.append(audio_path)

#         # run ASR
#         transcriptions = self.model.transcribe(paths2audio_files=audio_paths)

#         for transcription in transcriptions:
#             output.append({
#                 'result': [{
#                     'from_name': self.from_name,
#                     'to_name': self.to_name,
#                     'type': 'textarea',
#                     'value': {
#                         'text': [transcription]
#                     }
#                 }],
#                 'score': 1.0
#             })
#         return output

#     def _bind_to_textarea(self):
#         from_name, to_name, value = None, None, None
#         for tag_name, tag_info in self.parsed_label_config.items():
#             if tag_info['type'] == 'TextArea':
#                 from_name = tag_name
#                 if len(tag_info['inputs']) > 1:
#                     logger.warning(
#                         'ASR model works with single Audio or AudioPlus input, '
#                         'but {0} found: {1}. We\'ll use only the first one'.format(
#                             len(tag_info['inputs']), ', '.join(tag_info['to_name'])))
#                 if tag_info['inputs'][0]['type'] not in ('Audio', 'AudioPlus'):
#                     raise ValueError('{0} tag expected to be of type Audio or AudioPlus, but type {1} found'.format(
#                         tag_info['to_name'][0], tag_info['inputs'][0]['type']))
#                 to_name = tag_info['to_name'][0]
#                 value = tag_info['inputs'][0]['value']
#         if from_name is None:
#             raise ValueError('ASR model expects <TextArea> tag to be presented in a label config.')
#         return from_name, to_name, value

#     def fit(self, completions, workdir=None, **kwargs):
#         project_path = kwargs.get('project_full_path')
#         if os.path.exists(project_path):
#             logger.info('Found project in local path ' + project_path)
#         else:
#             logger.error('Project not found in local path ' + project_path + '. Serving uploaded data will fail.')
#         return {'project_path': project_path}
