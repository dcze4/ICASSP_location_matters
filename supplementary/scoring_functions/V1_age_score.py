import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def age_contribution(traits: set):
    intensity_modifiers = {"slightly": -0.5, "": 0, "very": +0.5}
    keyword_weights = {
        "young": -2,
        "adult-like": -1,
        "middle-aged": 0,
        "mature": 1,
        "old": 2,
    }

    total_score = 0

    if not traits:
        return 0

    for ann in traits:
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

                direction = 1 if weight >= 0 else -1
                total_score += weight + direction * intensity
                break

    # return normalized contribution
    # each annotation can reach max absolute 2.5 value
    return total_score / (len(traits) * 2.5)


def get_traits(traits: set, keys: set) -> set:
    return {trait for trait in traits if any([key in trait for key in keys])}


def extract(traits):
    age_keys = {"adult-like", "mature", "middle-aged", "old", "young"}
    return get_traits(set(traits.split(":")), age_keys)


def age_score(row):
    contribs = [
        age_contribution(extract(row[key]))
        for key in ["traits1", "traits2", "traits3"]
    ]
    # return average
    return np.mean(contribs)


def compute_age_score(df: pd.DataFrame):
    df["age_score"] = df.apply(lambda row: age_score(row), axis=1)
    return df


def plot_age_score_distribution(df: pd.DataFrame):
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 8))

    sns.histplot(
        df["age_score"].dropna(),
        kde=True,
        edgecolor="black",
        ax=ax1,
        alpha=0.7,
    )
    ax1.set_yscale("log")
    ax1.set_xlabel("Age Score")
    ax1.set_ylabel("Count (log scale)")
    ax1.set_title("Age Score Distribution")
    ax1.axvline(x=0, color="red", linestyle="--", alpha=0.5, label="Neutral")
    plt.show()


def test_age_contribution():
    # Single annotation tests
    assert age_contribution({"young"}) == -2 / 2.5  # -0.8
    assert age_contribution({"very young"}) == -2.5 / 2.5  # -1.0
    assert age_contribution({"slightly young"}) == -1.5 / 2.5  # -0.6

    assert age_contribution({"middle-aged"}) == 0 / 2.5  # 0.0
    assert age_contribution({"very middle-aged"}) == 0.5 / 2.5  # 0.2
    assert age_contribution({"slightly middle-aged"}) == -0.5 / 2.5  # -0.2

    assert age_contribution({"old"}) == 2 / 2.5  # 0.8
    assert age_contribution({"very old"}) == 2.5 / 2.5  # 1.0
    assert age_contribution({"slightly old"}) == 1.5 / 2.5  # 0.6

    # Multiple annotations (same annotator)
    # "young:old" -> (-2 + 2) / (2 * 2.5) = 0.0
    assert age_contribution({"young", "old"}) == 0.0

    # "young:young" -> (-2 + -2) / (2 * 2.5) = -0.8
    assert age_contribution({"young", "young"}) == -0.8

    # "slightly middle-aged:slightly mature" -> (-0.5 + 0.5) / (2 * 2.5) = 0.0
    assert age_contribution({"slightly middle-aged", "slightly mature"}) == 0.0

    # "adult-like:middle-aged" -> (-1 + 0) / (2 * 2.5) = -0.2
    assert age_contribution({"adult-like", "middle-aged"}) == -0.2

    # Verify ordering: [slightly middle-aged:slightly mature] > [adult-like:middle-aged]
    assert age_contribution(
        {"slightly middle-aged", "slightly mature"}
    ) > age_contribution({"adult-like", "middle-aged"})
    # Wait, both are 0.0 and -0.2, so 0.0 > -0.2 ✓

    print("All age_contribution tests passed!")


def test_age_score():
    # Create test rows
    test_cases = [
        # All annotators agree: young
        {
            "traits1": "young",
            "traits2": "young",
            "traits3": "young",
            "expected": -0.8,
        },
        # All annotators agree: old
        {
            "traits1": "old",
            "traits2": "old",
            "traits3": "old",
            "expected": 0.8,
        },
        # Mixed: one young, one old, one neutral
        {
            "traits1": "young",
            "traits2": "old",
            "traits3": "middle-aged",
            "expected": (-0.8 + 0.8 + 0.0) / 3,  # 0.0
        },
        # All neutral
        {
            "traits1": "middle-aged",
            "traits2": "adult-like",
            "traits3": "mature",
            "expected": (0.0 + -0.4 + 0.4) / 3,  # 0.0
        },
        # Intensity variation
        {
            "traits1": "very old",
            "traits2": "slightly old",
            "traits3": "old",
            "expected": (1.0 + 0.6 + 0.8) / 3,  # 0.8
        },
    ]

    for i, tc in enumerate(test_cases):
        row = pd.Series(tc)
        result = age_score(row)
        expected = tc["expected"]
        assert np.isclose(result, expected), (
            f"Test {i} failed: got {result}, expected {expected}"
        )

    print("All age_score tests passed!")


if __name__ == "__main__":
    test_age_contribution()
    test_age_score()
