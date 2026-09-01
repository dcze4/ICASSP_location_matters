import os
from typing import Literal

import click
import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from lhotse import MonoCut
from tqdm import tqdm
from transformers import AutoModelForAudioClassification, Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)
from utils import HDF5StorageManager


def setup_audeering_model(
    model_name: str = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim",
    device: str = "cuda:0",
):
    print("Setting up audeering model...")

    class RegressionHead(nn.Module):
        r"""Classification head."""

        def __init__(self, config):
            super().__init__()

            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
            self.dropout = nn.Dropout(config.final_dropout)
            self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

        def forward(self, features, **kwargs):
            x = features
            x = self.dropout(x)
            x = self.dense(x)
            x = torch.tanh(x)
            x = self.dropout(x)
            x = self.out_proj(x)

            return x

    class EmotionModel(Wav2Vec2PreTrainedModel):
        r"""Speech emotion classifier."""

        def __init__(self, config):
            super().__init__(config)

            self.config = config
            self.wav2vec2 = Wav2Vec2Model(config)
            self.classifier = RegressionHead(config)
            self.init_weights()

        def forward(
            self,
            input_values,
        ):
            outputs = self.wav2vec2(input_values)
            hidden_states = outputs[0]
            hidden_states = torch.mean(hidden_states, dim=1)
            logits = self.classifier(hidden_states)

            return hidden_states, logits

    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = EmotionModel.from_pretrained(model_name).to(device)  # type: ignore

    return processor, model


def setup_odyssey_with_embedding_hook(
    model_name: str = "3loi/SER-Odyssey-Baseline-WavLM-Multi-Attributes",
    device: str = "cuda:0",
):
    print("Setting up odyssey model with embedding hook...")
    odyssey_model = AutoModelForAudioClassification.from_pretrained(
        model_name, trust_remote_code=True
    )
    odyssey_model.to(device)

    # Storage for embeddings
    odyssey_model.embeddings_storage = {}

    # This is the embedding after the Attentive Pooling Layer
    def hook_fn(module, input, output):
        # output is the result of Attentive Pooling Layer
        odyssey_model.embeddings_storage["embeddings"] = output.detach()

    # Register hook on ser_model.out (the final Sequential layer)
    odyssey_model.pool_model.register_forward_hook(hook_fn)

    return odyssey_model.eval()


class EmotionEmbedder:
    def __init__(
        self,
        model: Literal["audeering", "odyssey"] = "audeering",
        device: str = "cuda:0",
    ):
        if model == "audeering":
            self.processor, self.model = setup_audeering_model(device=device)
            self.model_name = "audeering"
            self.predict = self._predict_with_audeering
        elif model == "odyssey":
            self.model = setup_odyssey_with_embedding_hook(device=device)
            self.model_name = "odyssey"
            self.predict = self._predict_odyssey_with_embeddings
        self.device = device

    def _predict_with_audeering(self, audio_path: str | MonoCut):
        if isinstance(audio_path, MonoCut):
            raw_wav = audio_path.resample(16000).load_audio()
        else:
            raw_wav, _ = librosa.load(audio_path, sr=16000)
            raw_wav = np.expand_dims(raw_wav, axis=0)

        inputs = self.processor(raw_wav, sampling_rate=16000)
        input_values = inputs["input_values"][0]
        input_values = input_values.reshape(1, -1)
        input_values = torch.from_numpy(input_values).to(self.device)

        # run through model
        with torch.no_grad():
            embeddings, logits = self.model(input_values)

        # convert to numpy
        logits = logits.cpu().numpy()
        embeddings = embeddings.squeeze().cpu().numpy()

        arousal, dominance, valence = [
            np.round(a, decimals=6).item() for a in logits.squeeze().tolist()
        ]

        emotion_dict = {
            f"A_{self.model_name}": arousal,
            f"D_{self.model_name}": dominance,
            f"V_{self.model_name}": valence,
        }

        return embeddings, emotion_dict

    def _predict_odyssey_with_embeddings(self, audio_path: str | MonoCut):
        if isinstance(audio_path, MonoCut):
            raw_wav = audio_path.resample(self.model.config.sampling_rate).load_audio()
        else:
            raw_wav, _ = librosa.load(audio_path, sr=self.model.config.sampling_rate)
            raw_wav = np.expand_dims(raw_wav, axis=0)
        norm_wav = (raw_wav - self.model.config.mean) / (
            self.model.config.std + 0.000001
        )
        mask = torch.ones(1, len(norm_wav)).to(self.device)
        wavs = torch.tensor(norm_wav).to(self.device)

        with torch.no_grad():
            pred = self.model(wavs, mask)

        # Get the embeddings captured by the hook
        embeddings = (
            self.model.embeddings_storage.get("embeddings").squeeze().cpu().numpy()
        )

        arousal, dominance, valence = [
            np.round(a, decimals=6).item() for a in pred.cpu().squeeze().tolist()
        ]

        emotion_dict = {
            f"A_{self.model_name}": arousal,
            f"D_{self.model_name}": dominance,
            f"V_{self.model_name}": valence,
        }

        return embeddings, emotion_dict


@click.command()
@click.option(
    "-i",
    "--input_csv_path",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="Path to the dataset (csv) file",
)
@click.option(
    "-oc",
    "--output_csv_path",
    type=click.Path(exists=False, dir_okay=False),
    required=True,
    help="Path to the dataset (csv) file",
)
@click.option(
    "-oh",
    "--output_hdf5_path",
    type=click.Path(dir_okay=False),
    required=True,
    help="Path to the output HDF5 file",
)
@click.option(
    "-a",
    "--audio_dir",
    type=click.Path(exists=True, file_okay=False),
    required=True,
    help="Directory containing audio files",
)
@click.option(
    "-m",
    "--model",
    type=click.Choice(["audeering", "odyssey"]),
    default="audeering",
    help="model name",
)
@click.option(
    "-d",
    "--device",
    type=str,
    default="cuda:0",
    help="Device to run the model on (e.g., 'cpu', 'cuda:0')",
)
def main(
    input_csv_path: str,
    output_csv_path: str,
    output_hdf5_path: str,
    audio_dir: str,
    model: str,
    device: str,
):
    emotion_embedder = EmotionEmbedder(model=model, device=device)
    storage_manager = HDF5StorageManager(hdf5_path=output_hdf5_path)

    df = pd.read_csv(input_csv_path)

    emotion_data = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        audio_filename = row["output_filename"]
        try:
            file_path = os.path.join(audio_dir, audio_filename)

            embedding, emotion_dict = emotion_embedder.predict(file_path)
            storage_manager.store_embedding(audio_filename, embedding)

            emotion_data.append({"index": idx, **emotion_dict})

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            emotion_data.append(
                {
                    "index": idx,
                    f"A_{emotion_embedder.model_name}": None,
                    f"D_{emotion_embedder.model_name}": None,
                    f"V_{emotion_embedder.model_name}": None,
                }
            )

    emotion_df = pd.DataFrame(emotion_data).set_index("index")
    df = df.join(emotion_df)

    df.to_csv(output_csv_path, index=False)


if __name__ == "__main__":
    main()
