import os

import click
import librosa
import pandas as pd
import torch
import torchaudio
from utils import HDF5StorageManager


class EmbeddingModelManager:
    def __init__(
        self, source="speechbrain/spkrec-ecapa-voxceleb", device: str = "cuda:0"
    ):
        self.model = None
        self.source = source
        self.device = device
        if source == "speechbrain/spkrec-ecapa-voxceleb":
            self.load_speechbrain_model(source)
            self.extract = self._speechbrain_extract_features

        if source == "nvidia/speakerverification_en_titanet_large":
            self.load_nvidia_model(source)
            self.extract = self._nvidia_extract_features

        if source == "microsoft/wavlm-base-plus-sv":
            self.load_microsoft_model(source)
            self.extract = self._microsoft_extract_features

    def load_nvidia_model(self, source):
        import nemo.collections.asr as nemo_asr

        print("Loading NVIDIA model...")

        self.model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(source)
        self.model.eval()
        self.model.to(self.device)

    def load_microsoft_model(self, source):
        from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector

        print("Loading Microsoft model...")

        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(source)
        self.model = WavLMForXVector.from_pretrained(source)
        self.model.eval()
        self.model.to(self.device)

    def load_speechbrain_model(self, source):
        from speechbrain.inference.speaker import EncoderClassifier

        print("Loading SpeechBrain model...")

        self.model = EncoderClassifier.from_hparams(source=source)
        self.model.eval()
        self.model.to(self.device)
        # Assign device back to model otherwise it throws error even though model weights on cuda
        self.model.device = self.device

    def _nvidia_extract_features(self, audio_path):
        """Extract speaker embeddings from audio"""
        with torch.no_grad():
            embedding = self.model.get_embedding(audio_path)
        return embedding.squeeze().cpu().numpy()

    def _speechbrain_extract_features(self, audio_path):
        """Extract speaker embeddings from audio"""
        signal, _ = torchaudio.load(audio_path)
        signal = signal.to(self.device)
        with torch.no_grad():
            embedding = self.model.encode_batch(signal)
        return embedding.squeeze().cpu().numpy()

    def _microsoft_extract_features(self, audio_path):
        raw_wav, _ = librosa.load(audio_path, sr=16000)
        inputs = self.feature_extractor(
            raw_wav, padding=True, sampling_rate=16000, return_tensors="pt"
        )
        inputs.to(self.device)
        with torch.no_grad():
            embeddings = self.model(**inputs).embeddings
        embeddings = torch.nn.functional.normalize(embeddings, dim=-1).cpu()
        return embeddings.squeeze().numpy()


@click.command()
@click.option(
    "-i",
    "--input_csv_path",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="Path to the dataset (csv) file",
)
@click.option(
    "-o",
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
    "-s",
    "--source",
    type=str,
    default="speechbrain/spkrec-ecapa-voxceleb",
    help="Source of the pre-trained model",
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
    output_hdf5_path: str,
    audio_dir: str,
    source: str,
    device: str,
):
    embedding_model_manager = EmbeddingModelManager(source=source, device=device)
    storage_manager = HDF5StorageManager(hdf5_path=output_hdf5_path)

    df = pd.read_csv(input_csv_path)

    for audio_filename in df.loc[:, "output_filename"]:
        try:
            file_path = os.path.join(audio_dir, audio_filename)

            embedding = embedding_model_manager.extract(file_path)
            storage_manager.store_embedding(audio_filename, embedding)

        except Exception as e:
            print(f"Error processing {file_path}: {e}")


if __name__ == "__main__":
    main()
