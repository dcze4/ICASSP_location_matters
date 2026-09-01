import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

VOICE_KEYS = {
    "masculine": "gender",
    "feminine": "gender",
    "gender-neutral": "gender",
    "young": "age",
    "adult-like": "age",
    "middle-aged": "age",
    "mature": "age",
    "old": "age",
    "raspy": "texture",
    "nasal": "texture",
    "muffled": "texture",
    "clear": "texture",
    "bright": "texture",
    "dark": "texture",
    "thick": "texture",
    "thin": "texture",
    "soft": "texture",
    "hard": "texture",
    "sharp": "texture",
}

STYLE_KEYS = {
    "calm": "arousal",
    "relaxed": "arousal",
    "tensed": "arousal",
    "intense": "arousal",
    "lively": "arousal",
    "wild": "arousal",
    "friendly": "valence",
    "kind": "valence",
    "sweet": "valence",
    "reassuring": "valence",
    "refreshing": "valence",
    "sincere": "valence",
    "strict": "valence",
    "fluent": "speaking_rate",
    "halting": "speaking_rate",
    "powerful": "loudness",
    "weak": "loudness",
}

AMBIGUOUS_KEYS = {
    "cute",
    "sexy",
    "elegant",
    "intellectual",
    "cool",
    "modest",
    "unique",
}

TEXTURE_TRAITS = {
    "slightly bright",  # Spectral Brightness bright <--> dark
    "bright",  # Spectral Brightness bright <--> dark
    "very bright",  # Spectral Brightness bright <--> dark
    "slightly dark",  # Spectral Brightness bright <--> dark
    "dark",  # Spectral Brightness bright <--> dark
    "very dark",  # Spectral Brightness bright <--> dark
    "slightly thin",  # Density thin <--> thick
    "thin",  # Density thin <--> thick
    "very thin",  # Density thin <--> thick
    "slightly thick",  # Density thin <--> thick
    "thick",  # Density thin <--> thick
    "very thick",  # Density thin <--> thick
    "slightly clear",  # Surface/Roughness clear <--> raspy
    "clear",  # Surface/Roughness clear <--> raspy
    "very clear",  # Surface/Roughness clear <--> raspy
    "slightly raspy",  # Surface/Roughness clear <--> raspy
    "raspy",  # Surface/Roughness clear <--> raspy
    "very raspy",  # Surface/Roughness clear <--> raspy
    "slightly soft",  # Hardness soft <--> hard
    "soft",  # Hardness soft <--> hard
    "very soft",  # Hardness soft <--> hard
    "slightly hard",  # Hardness soft <--> hard
    "hard",  # Hardness soft <--> hard
    "very hard",  # Hardness soft <--> hard
    "slightly muffled",
    "muffled",
    "very muffled",
    "slightly nasal",  # Resonance
    "nasal",  # Resonance
    "slightly sharp",  # Sharpness
    "sharp",  # Sharpness
    "very sharp",  # Sharpness
}

TEXTURE_TRAITS_CLASSIFIED = {
    "bright",  # Spectral Brightness bright <--> dark
    "dark",  # Spectral Brightness bright <--> dark
    "thin",  # Density thin <--> thick
    "thick",  # Density thin <--> thick
    "clear",  # Surface/Roughness clear <--> raspy
    "raspy",  # Surface/Roughness clear <--> raspy
    "soft",  # Hardness soft <--> hard
    "hard",  # Hardness soft <--> hard
    "muffled",
    "nasal",  # Resonance
    "sharp",  # Sharpness
}


def texture_contribution(traits: set, poles: tuple[str, str]):
    intensity_modifiers = {"slightly": -0.5, "": 0, "very": +0.5}

    neg, pos = poles
    keyword_weights = {
        neg: -1,
        pos: 1,
    }

    total_score = 0
    filtered_traits = [trait for trait in traits if neg in trait or pos in trait]

    if not filtered_traits:
        return 0

    for ann in filtered_traits:
        for kw, weight in keyword_weights.items():
            # compute contribution total
            if kw in ann:
                # Parse intensity
                if ann.startswith("very "):
                    intensity = intensity_modifiers["very"]
                elif ann.startswith("slightly "):
                    intensity = intensity_modifiers["slightly"]
                else:
                    intensity = intensity_modifiers[""]

                # We might change weight but direction should stay 1 or -1
                direction = 1 if weight >= 0 else -1
                total_score += weight + direction * intensity
                break

    # return normalized contribution
    # each annotation can reach max absolute 1.5 value
    return total_score / (
        len(filtered_traits) * (keyword_weights[pos] + intensity_modifiers["very"])
    )


def texture_score(row, poles: tuple[str, str] = ("dark", "bright")):
    contribs = [
        texture_contribution(
            set(row[key].split(":")),
            poles,
        )
        for key in ["traits1", "traits2", "traits3"]
    ]
    # return average contribution for given polarity
    return np.mean(contribs)


def compute_texture_score(df: pd.DataFrame):
    polarities = [
        ("brightness", ("bright", "dark")),
        ("thickness", ("thin", "thick")),
        ("raspiness", ("clear", "raspy")),
        ("hardness", ("soft", "hard")),
    ]

    for texture, poles in polarities:
        df[f"{texture}_score"] = df.apply(lambda row: texture_score(row, poles), axis=1)
    return df


def plot_texture_scores(df: pd.DataFrame, output_path: str):
    polarities = [
        "brightness_score",
        "thickness_score",
        "raspiness_score",
        "hardness_score",
    ]

    for texture in polarities:
        plt.figure(figsize=(8, 6))
        sns.histplot(df[texture], bins=20, kde=True)  # type: ignore
        plt.title(f"Distribution of {texture.replace('_', ' ').title()}")
        plt.xlabel("Score")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.savefig(f"{output_path}/{texture}_distribution.png")
        plt.close()


def test_polarities(poles: tuple[str, str]):
    neg, pos = poles
    max_abs = 1.5

    check = lambda x, y: np.isclose(texture_contribution(x, poles), y)

    x, y = {pos}, 1 / max_abs
    assert check(x, y), (
        f"For {x} expected {y}, but got {texture_contribution(x, poles)}"
    )

    x, y = {f"very {pos}"}, 1
    assert check(x, y), (
        f"For {x} expected {y}, but got {texture_contribution(x, poles)}"
    )

    x, y = {f"slightly {neg}"}, (-1 + 0.5) / max_abs
    assert check(x, y), (
        f"For {x} expected {y}, but got {texture_contribution(x, poles)}"
    )

    x, y = {neg}, -1 / max_abs
    assert check(x, y), (
        f"For {x} expected {y}, but got {texture_contribution(x, poles)}"
    )

    x, y = {f"very {neg}"}, -1
    assert check(x, y), (
        f"For {x} expected {y}, but got {texture_contribution(x, poles)}"
    )

    print(f"All tests passed for poles: {poles}\n")


if __name__ == "__main__":
    polarities = [
        ("brightness", ("bright", "dark")),
        ("thickness", ("thin", "thick")),
        ("raspiness", ("clear", "raspy")),
        ("hardness", ("soft", "hard")),
    ]

    show = lambda x, poles: print(
        f"  {' + '.join(list(x))} = {texture_contribution(set(x), poles)}"
    )

    for poles in polarities:
        print(f"Testing texture polarity: {poles[0]} with poles {poles[1]}...")
        test_polarities(poles[1])
        print("\nChecking current behavior:")
        behaviors = [
            [f"very {poles[1][0]}"],
            [poles[1][0]],
            [f"slightly {poles[1][0]}"],
            [f"slightly {poles[1][1]}"],
            [poles[1][1]],
            [f"very {poles[1][1]}"],
            [poles[1][0], f"slightly {poles[1][1]}"],
            [poles[1][0], poles[1][1]],
            [poles[1][0], f"very {poles[1][1]}"],
            [poles[1][1], f"slightly {poles[1][0]}"],
            [poles[1][1], poles[1][0]],
            [poles[1][1], f"very {poles[1][0]}"],
        ]
        for behavior in behaviors:
            show(behavior, poles[1])
        print("-----------------------------\n")
