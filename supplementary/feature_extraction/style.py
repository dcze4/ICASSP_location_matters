import os

import click
import librosa
import numpy as np
import pandas as pd
import parselmouth
import penn
import torch
import webrtcvad
from emotion import EmotionEmbedder
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from tqdm import tqdm
from utils import HDF5StorageManager


def compute_pitch(
    audio,
    sample_rate=24000,
    hopsize=0.01,
    fmin=30.0,
    fmax=1000.0,
    gpu=0,
    batch_size=2048,
    checkpoint=None,
    center="half-hop",
    interp_unvoiced_at=0.065,
    decoder="viterbi",
):
    audio = torch.from_numpy(audio).unsqueeze(dim=0)

    pitch, periodicity = penn.from_audio(
        audio,
        sample_rate,
        hopsize=hopsize,
        fmin=fmin,
        fmax=fmax,
        checkpoint=checkpoint,
        batch_size=batch_size,
        center=center,
        decoder=decoder,
        interp_unvoiced_at=interp_unvoiced_at,
        gpu=gpu,
    )

    pitch = pitch.squeeze().cpu().numpy()
    periodicity = periodicity.squeeze().cpu().numpy()

    return pitch, periodicity


def compute_pitch_variation(audio, sample_rate, periodicity_threshold=0.5):
    """
    Compute pitch variation WITHIN an utterance (style feature)
    Not absolute pitch (which is a voice characteristic)
    """
    # Extract F0 contour
    f0, periodicity = compute_pitch(audio, sample_rate)
    voiced_mask = periodicity > periodicity_threshold
    f0_voiced = f0[voiced_mask]

    if len(f0_voiced) < 5:  # Need minimum frames
        return 0.0, 0.0

    # Convert to semitones (relative to utterance mean)
    f0_mean = np.mean(f0_voiced)
    f0_semitones = 12 * np.log2(f0_voiced / f0_mean + 1e-10)

    # Compute variation metrics
    pitch_std = np.std(f0_semitones)  # Variation in semitones
    pitch_range = np.percentile(f0_semitones, 95) - np.percentile(f0_semitones, 5)

    return pitch_std, pitch_range


def compute_speech_rate_final(audio, sr=24000, threshold=72.0, debug=False):
    """
    Final speech rate + energy features for style representation.

    Args:
        audio: Audio signal (numpy array)
        sr: Sample rate (default 24000)
        return_dim: 5 or 8 dimensions
        threshold: 72.0 dB threshold
        debug: Print debug information

    Returns:
        Fixed-size feature vector
    """
    # Ensure mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # Store original for energy computation
    audio_original = audio.copy()

    # Normalize to [-1, 1] for consistent processing
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val

    duration_seconds = len(audio) / sr

    # 1. Syllable detection with FIXED threshold
    syllable_rate, syll_interval_std = estimate_syllable_rate_fair(
        audio,
        sr,
        fixed_threshold=threshold,
        debug=debug,
    )

    # 2. VAD ratio (keep just the ratio, drop pause stats)
    vad_ratio = compute_vad_ratio_only(audio, sr)

    # 4. Energy mean and std
    energy_mean_db, energy_std = compute_energy_features(
        audio_original, sr, debug=debug
    )

    # Picth variation
    pitch_std, pitch_range = compute_pitch_variation(audio_original, sr)

    if debug:
        print(f"\nSpeech Rate + Energy Features (duration: {duration_seconds:.2f}s):")
        print(f"  Syllable rate: {syllable_rate:.2f}/s")
        print(f"  VAD ratio: {vad_ratio:.3f}")
        print(f"  Energy mean (dB): {energy_mean_db:.1f}")
        print(f"  Energy std: {energy_std:2f}")
        print(f"  Pitch std (semitones): {pitch_std:.2f}")
        print(f"  Pitch range (semitones): {pitch_range:.2f}")

    return np.array(
        [
            syllable_rate,  # Speech rate
            syll_interval_std,  # Std inter-syllable interval (rhythm regularity)
            vad_ratio,  # Speech ratio
            energy_mean_db,  # Mean energy (dB)
            energy_std,  # Energy std (dynamics)
            pitch_std,  # Pitch variation (style)
            pitch_range,  # Pitch range (style)
        ],
        dtype=np.float32,
    )


def compute_energy_features(audio, sr, debug=False):
    """
    Compute energy-based features that capture loudness and dynamics.
    These are crucial for style (e.g., whispered vs. shouted speech)
    """
    # Frame-based energy computation
    frame_length = int(0.025 * sr)  # 25ms frames
    hop_length = int(0.010 * sr)  # 10ms hop

    # Compute RMS energy
    rms_energy = librosa.feature.rms(
        y=audio, frame_length=frame_length, hop_length=hop_length
    )[0]

    # Convert to dB
    rms_db = 20 * np.log10(rms_energy + 1e-10)

    # Remove silence frames (bottom 10%)
    threshold = np.percentile(rms_db, 10)
    speech_energy = rms_db[rms_db > threshold]

    # Energy statistics
    if len(speech_energy) > 0:
        energy_mean = np.mean(speech_energy)
        energy_std = np.std(speech_energy)

    else:
        energy_mean = np.mean(rms_db)
        energy_std = np.std(rms_db)

    if debug:
        print(f"Energy stats: mean={energy_mean:.1f}dB, std={energy_std:.2f}")

    return np.array(
        [
            energy_mean,  # Overall loudness
            energy_std,  # Dynamic variability
        ],
        dtype=np.float32,
    )


def compute_vad_ratio_only(audio, sr):
    """
    Simplified VAD - just return the ratio of speech vs silence
    """
    # WebRTC VAD for robust detection
    supported_rates = [8000, 16000, 32000, 48000]

    if sr in supported_rates:
        vad_sr = sr
        vad_audio = audio
    else:
        vad_sr = min(supported_rates, key=lambda r: abs(r - sr))
        vad_audio = librosa.resample(
            audio, orig_sr=sr, target_sr=vad_sr, res_type="kaiser_best"
        )

    vad_audio = np.clip(vad_audio, -1.0, 1.0)
    audio_16bit = (vad_audio * 32767).astype(np.int16)

    frame_ms = 30
    frame_length = int(vad_sr * frame_ms / 1000)

    n_frames = int(np.ceil(len(audio_16bit) / frame_length))
    pad_length = n_frames * frame_length - len(audio_16bit)
    if pad_length > 0:
        audio_16bit = np.concatenate(
            [audio_16bit, np.zeros(pad_length, dtype=np.int16)]
        )

    vad = webrtcvad.Vad(2)

    voiced_frames = []
    for i in range(n_frames):
        start = i * frame_length
        frame = audio_16bit[start : start + frame_length].tobytes()
        try:
            is_speech = vad.is_speech(frame, vad_sr)
            voiced_frames.append(is_speech)
        except:
            voiced_frames.append(False)

    return float(np.mean(voiced_frames)) if voiced_frames else 0.0


def estimate_syllable_rate_fair(audio, sr, fixed_threshold=72.0, debug=False):
    """
    Syllable detection using FIXED threshold
    """
    snd = parselmouth.Sound(audio.astype(np.float64), sampling_frequency=sr)
    intensity = snd.to_intensity(minimum_pitch=50)

    times = intensity.xs()
    vals = intensity.values[0]
    vals_smooth = gaussian_filter1d(vals, sigma=1)

    threshold = fixed_threshold

    if debug:
        print(f"Using fixed threshold: {threshold:.1f} dB")
        print(f"Intensity range: [{vals.min():.1f}, {vals.max():.1f}] dB")

    time_step = times[1] - times[0] if len(times) > 1 else 0.01
    min_distance_samples = int(0.08 / time_step)

    peaks, _ = find_peaks(
        vals_smooth,
        height=threshold,
        distance=min_distance_samples,
        prominence=2,
    )

    duration_seconds = len(audio) / sr
    syllable_count = len(peaks)
    syllable_rate = syllable_count / duration_seconds if duration_seconds > 0 else 0

    if len(peaks) > 1:
        peak_times = times[peaks]
        inter_syllable_intervals = np.diff(peak_times)
        # Skip mean computation, it is only inverse of syllable rate
        # mean_interval = float(np.mean(inter_syllable_intervals))
        std_interval = float(np.std(inter_syllable_intervals))
    else:
        # mean_interval = duration_seconds if syllable_count == 1 else 0.0
        std_interval = 0.0

    return syllable_rate, std_interval


@click.command()
@click.option(
    "-i",
    "--input_csv_path",
    type=click.Path(exists=True, dir_okay=False),
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
    output_hdf5_path: str,
    audio_dir: str,
    model: str,
    device: str,
):
    emotion_embedder = EmotionEmbedder(model=model, device=device)  # type: ignore
    storage_manager = HDF5StorageManager(hdf5_path=output_hdf5_path)

    df = pd.read_csv(input_csv_path)

    for _, row in tqdm(df.iterrows(), total=len(df)):
        audio_filename = row["output_filename"]
        try:
            file_path = os.path.join(audio_dir, audio_filename)  # type: ignore

            _, emotion_dict = emotion_embedder.predict(file_path)

            audio, _ = librosa.load(file_path, sr=24000)
            features = compute_speech_rate_final(
                audio, 24000, threshold=72.0, debug=False
            )

            # Use predicted emotion values as part of style embedding
            emotions = np.array(
                [
                    emotion_dict[f"A_{emotion_embedder.model_name}"],
                    emotion_dict[f"D_{emotion_embedder.model_name}"],
                    emotion_dict[f"V_{emotion_embedder.model_name}"],
                ],
                dtype=np.float32,
            )

            # Combine style features and emotion embeddings
            embedding = np.concatenate([features, emotions], axis=0)

            # Store in HDF5
            storage_manager.store_embedding(audio_filename, embedding)

        except Exception as e:
            print(f"Error processing {file_path}: {e}")


if __name__ == "__main__":
    main()
