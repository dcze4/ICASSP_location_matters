import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def gender_contribution(traits: set):
    # trait: (score, weight)
    keyword_weights = {
        "very feminine": (3, 3),
        "feminine": (2, 3),
        "slightly feminine": (1, 3),
        "slightly gender-neutral": (0, 1),
        "gender-neutral": (0, 2),
        "very gender-neutral": (0, 3),
        "slightly masculine": (-1, 3),
        "masculine": (-2, 3),
        "very masculine": (-3, 3),
    }

    if not traits:
        return 0

    gen_traits = []
    for trait in traits:
        if trait in keyword_weights.keys():
            gen_traits.append(trait)

    if not gen_traits:
        return 0

    total_score = 0
    total_weight = 0

    for ann in gen_traits:
        val, weight = keyword_weights[ann]

        total_score += val
        # The weight of gender-neutral is important because it's score is 0.
        # But, different weights help us to distinguish between slightly, '',
        # and very gender-neutral contributions
        total_weight += weight

    # return normalized contribution
    result = total_score / total_weight

    return result


def gender_score(row):
    contribs = [
        gender_contribution(row[key].split(":"))
        for key in ["traits1", "traits2", "traits3"]
    ]
    # return average
    return np.mean(contribs)


def compute_gender_score(df: pd.DataFrame):
    df["gender_score"] = df.apply(lambda row: gender_score(row), axis=1)
    return df


def plot_gender_score_distribution(df: pd.DataFrame):
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 8))

    sns.histplot(
        df["gender_score"].dropna(),
        kde=True,
        edgecolor="black",
        ax=ax1,
        alpha=0.7,
    )
    # ax1.set_yscale("log")
    ax1.set_xlabel("Gender Score")
    ax1.set_ylabel("Count (log scale)")
    ax1.set_title("Gender Score Distribution")
    ax1.axvline(x=0, color="red", linestyle="--", alpha=0.5, label="Neutral")
    plt.show()


if __name__ == "__main__":

    def test_gender_contribution():
        max_possible_abs = 3
        check = lambda x, y: np.isclose(gender_contribution(x), y)
        show = lambda x: print(
            f"  {' + '.join(list(x))} = {gender_contribution(set(x))}"
        )

        x, y = {"feminine"}, 2 / max_possible_abs
        assert check(x, y), (
            f"For {x} expected {y}, but got {gender_contribution(x)}"
        )

        x, y = {"very feminine"}, 1
        assert check(x, y), (
            f"For {x} expected {y}, but got {gender_contribution(x)}"
        )

        x, y = {"slightly masculine"}, -1 / max_possible_abs
        assert check(x, y), (
            f"For {x} expected {y}, but got {gender_contribution(x)}"
        )

        x, y = {"gender-neutral"}, 0
        assert check(x, y), (
            f"For {x} expected {y}, but got {gender_contribution(x)}"
        )
        print("All tests passed!\n")

        print("Checking current behavior:")
        behaviors = [
            ["very masculine"],
            ["masculine"],
            ["slightly masculine"],
            ["slightly gender-neutral"],
            ["gender-neutral"],
            ["very gender-neutral"],
            ["slightly feminine"],
            ["feminine"],
            ["very feminine"],
            ["feminine", "slightly gender-neutral"],
            ["feminine", "gender-neutral"],
            ["feminine", "very gender-neutral"],
            ["masculine", "slightly gender-neutral"],
            ["masculine", "gender-neutral"],
            ["masculine", "very gender-neutral"],
        ]

        for behavior in behaviors:
            show(behavior)

    test_gender_contribution()
