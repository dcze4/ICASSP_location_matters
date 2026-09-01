import os
from pathlib import Path

import click
import numpy as np
import pandas as pd
import penn
import torch
from lhotse import CutSet, combine, load_manifest_lazy
from tqdm import tqdm


def collect_cuts(dataset_dir) -> CutSet:
    dir_path = Path(dataset_dir)
    dataset_files = [
        dir_path / file
        for file in os.listdir(dataset_dir)
        if file.endswith(".jsonl.gz")
    ]
    combined_cuts = combine(
        [load_manifest_lazy(file) for file in dataset_files]  # type: ignore
    )
    return combined_cuts


def filter_by_speaker(cuts, target_speaker: str, n: int = 10):
    filtered = cuts.filter(
        lambda c: c.supervisions[0].speaker == target_speaker  # type: ignore
    ).to_eager()

    top_n = filtered.sort_by_duration(ascending=False).subset(first=n)

    print(f"Found {len(filtered)} cuts for speaker {target_speaker}")
    print(f"Selecting top {len(top_n)} longest cuts")

    return top_n


def compute_pitch(
    cut,
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
    audio, sample_rate = cut.load_audio(), cut.sampling_rate
    audio = torch.from_numpy(audio)

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


def compute_stats(pitch, periodicity, periodicity_threshold=0.5):
    voiced_mask = periodicity > periodicity_threshold
    f0_voiced = pitch[voiced_mask]
    return {
        "f0_median": np.median(f0_voiced),
        "f0_mean": np.mean(f0_voiced),
        "f0_std": np.std(f0_voiced),
        "mean_periodicity": np.mean(periodicity),
        "voiced_ratio": np.sum(voiced_mask) / len(periodicity),
    }


def categorize_pitch(
    pitch_df: pd.DataFrame, categorized_df: pd.DataFrame
) -> pd.DataFrame:
    # get gender_category from categorized_df
    temp_df = categorized_df[["spk_id", "gender_category"]]
    temp_df = temp_df.rename(columns={"spk_id": "speaker"})  # type: ignore
    df = pitch_df.set_index("speaker").join(temp_df.set_index("speaker")).reset_index()

    # Aggregate per speaker (take mean of their recordings)
    speaker_stats = (
        df.groupby(["speaker", "gender_category"])
        .agg({"f0_median": "mean"})
        .reset_index()
    )

    def assign_pitch_category(series: pd.Series) -> pd.Series:
        """Assign low/medium/high based on quartiles."""
        q1, _, q3 = series.quantile([0.25, 0.5, 0.75])
        return pd.cut(  # type: ignore
            series,
            bins=[-np.inf, q1, q3, np.inf],
            labels=["low", "medium", "high"],
        )

    # Categorize within each gender group
    speaker_stats["pitch_category"] = speaker_stats.groupby("gender_category")[
        "f0_median"
    ].transform(assign_pitch_category)

    categorized_df["pitch_category"] = categorized_df["spk_id"].map(
        speaker_stats.set_index("speaker")["pitch_category"]  # type: ignore
    )

    return categorized_df


def get_gender_pitch_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Get quantile boundaries for each gender category."""
    speaker_stats = (
        df.groupby(["speaker", "gender_category"])
        .agg(
            {
                "f0_median": "mean",
                "f0_mean": "mean",
            }
        )
        .reset_index()
    )

    stats = speaker_stats.groupby("gender_category").agg(
        {
            "f0_median": [
                ("q1", lambda x: x.quantile(0.25)),
                ("q2", lambda x: x.quantile(0.5)),
                ("q3", lambda x: x.quantile(0.75)),
                ("min", "min"),
                ("max", "max"),
            ],
            "f0_mean": [
                ("q1", lambda x: x.quantile(0.25)),
                ("q2", lambda x: x.quantile(0.5)),
                ("q3", lambda x: x.quantile(0.75)),
                ("min", "min"),
                ("max", "max"),
            ],
        }
    )

    return stats


def extract_pitch_values(input_path, output_csv):
    cuts = collect_cuts(input_path)

    speakers = list(cuts.speakers)

    print(f"Number of speakers: {len(speakers)}")

    # filter params
    n_top_cuts = 10

    # pitch computation params
    hopsize = 0.01
    fmin = 30.0
    fmax = 1000.0
    gpu = 0
    batch_size = 2048
    checkpoint = None
    center = "half-hop"
    interp_unvoiced_at = 0.065
    decoder = "viterbi"

    all_data = pd.DataFrame()

    for speaker in tqdm(speakers, total=len(speakers)):
        top_cuts = filter_by_speaker(cuts=cuts, target_speaker=speaker, n=n_top_cuts)
        rows = []
        for cut in top_cuts:  # type: ignore
            pitch, periodicity = compute_pitch(
                cut,
                hopsize,
                fmin,
                fmax,
                gpu,
                batch_size,
                checkpoint,
                center,
                interp_unvoiced_at,
                decoder,
            )
            stats = compute_stats(pitch, periodicity)

            # pandas row dict
            row = {"cut_id": cut.id, "speaker": speaker, **stats}
            rows.append(row)
        df = pd.DataFrame.from_records(rows)
        all_data = pd.concat([all_data, df], ignore_index=True)
        all_data.to_csv(output_csv, index=False)


@click.command()
@click.option(
    "-i",
    "--input_dir",
    type=click.Path(exists=True, file_okay=False),
    required=True,
    help="Path to the dataset directory",
)
@click.option(
    "-o",
    "--output_csv_path",
    type=click.Path(exists=False, dir_okay=False),
    required=True,
    help="Path to the dataset (csv) file",
)
@click.option(
    "-t",
    "--task",
    type=click.Choice(
        ["extract_pitch_values", "categorize_pitch_values"],
        case_sensitive=False,
    ),
    required=True,
    help="Task to run.",
)
def main(input_dir, output_csv_path, task):
    if task == "extract_pitch_values":
        extract_pitch_values(input_dir, output_csv_path)
    elif task == "categorize_pitch_values":
        # "pitch_stats_per_speaker.csv", index=False)
        pitch_df = pd.read_csv("pitch_stats_per_speaker.csv")
        categorized_df = pd.read_csv("categorized_speakers.csv")
        result_df = categorize_pitch(pitch_df, categorized_df)
        result_df.to_csv("categorized_speakers_with_pitch.csv", index=False)


if __name__ == "__main__":
    main()
