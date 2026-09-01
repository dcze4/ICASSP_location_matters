import os
from pathlib import Path

import click
import numpy as np
import pandas as pd
from lhotse import CutSet, combine, load_manifest_lazy
from tqdm import tqdm


def collect_cuts(dataset_dir: str) -> CutSet:
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


def extract_info(cut):
    custom = cut.supervisions[0].custom
    row = {
        "id": cut.id,
        "spk_id": custom["spk_id"],
        "syllable_rate": custom["syllable_rate"],
        "syll_interval_std": custom["syll_interval_std"],
        "vad_ratio": custom["vad_ratio"],
        "energy_mean_db": custom["energy_mean_db"],
        "energy_std": custom["energy_std"],
        "pitch_std": custom["pitch_std"],
        "pitch_range": custom["pitch_range"],
        "A_odyssey": custom["A_odyssey"],
        "D_odyssey": custom["D_odyssey"],
        "V_odyssey": custom["V_odyssey"],
    }
    return row


def construct_df(cuts):
    rows = []
    for cut in tqdm(cuts, total=len(cuts)):
        row = extract_info(cut)
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def assign_category(series: pd.Series, labels: list[str]) -> pd.Series:
    """Assign labels based on quartiles."""
    q1, q2, q3 = series.quantile([0.25, 0.5, 0.75])
    print(f"Quantiles: {q1}, {q2}, and {q3}\n")
    return pd.cut(  # type: ignore
        series,
        bins=[-np.inf, q1, q3, np.inf],
        labels=labels,
    )


def map_emotional_tone(arousal_category: str, valence_category: str) -> str:
    """Map arousal and valence categories to emotional tone."""
    if valence_category == "low" and arousal_category == "low":
        return "sad"
    elif valence_category == "low" and arousal_category == "high":
        return "angry"
    elif valence_category == "high" and arousal_category == "high":
        return "happy"
    else:
        return "neutral"


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
def main(input_dir, output_csv_path):
    cuts = collect_cuts(input_dir)
    df = construct_df(cuts)

    df["A_category"] = df["A_odyssey"].transform(
        lambda x: assign_category(x, ["low", "neutral", "high"])
    )
    df["V_category"] = df["V_odyssey"].transform(
        lambda x: assign_category(x, ["low", "neutral", "high"])
    )

    df["energy_category"] = df["energy_mean_db"].transform(
        lambda x: assign_category(x, ["quiet", "normal", "loud"])
    )
    df["speed_category"] = df["syllable_rate"].transform(
        lambda x: assign_category(x, ["slow", "normal", "fast"])
    )

    df["emotional_tone"] = df.apply(
        lambda row: map_emotional_tone(row["A_category"], row["V_category"]),
        axis=1,
    )

    df.to_csv(output_csv_path, index=False)


if __name__ == "__main__":
    main()
