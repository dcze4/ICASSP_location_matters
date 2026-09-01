import random
from typing import Literal


# Loose versions
def generate_voice_description_templated(
    age: Literal["young", "adult", "old"],
    gender: Literal["male", "female", "gender-neutral"],
    pitch: Literal["low", "medium", "high"],
    textures: list[str] | None = None,
    template_id: int = -1,
):
    if textures is None:
        textures = []

    article_age = f"{'a ' + age if age == 'young' else 'an ' + age}"
    sounds_like_age = (
        f"sounds like {article_age if age == 'adult' else article_age + ' person'}"
    )
    sounds_adj_age = f"sounds {'like an ' + age if age == 'adult' else age}"

    match textures:
        case t1, t2, t3:
            templates = [
                f"This is {article_age}, {gender} voice with {pitch} pitch and {t1}, {t2}, {t3} quality.",
                f"{'A ' + age if age == 'young' else 'An ' + age} {gender} voice featuring {pitch} pitch with {t1}, {t2}, and {t3} tone.",
                f"The voice is {gender} and {age}, with {pitch} pitch characterized by {t1}, {t2}, and {t3} qualities.",
                f"This {age}, {gender} voice has {pitch} pitch with {t1}, {t2}, and {t3} characteristics.",
                f"A {t1}, {t2}, {t3}, {pitch}-pitched voice that is {gender} and {sounds_adj_age}.",
                f"{gender.capitalize()} voice that {sounds_adj_age}, with {pitch} pitch and {t1}, {t2}, {t3} quality.",
                f"{age.capitalize()} and {gender}, this voice features {pitch} pitch with {t1}, {t2}, and {t3} tone.",
                f"{gender.capitalize()} speaker who is {'an ' + age if age == 'adult' else age}, displaying {pitch} pitch with {t1}, {t2}, and {t3} characteristics.",
                f"{age.capitalize()} {gender} voice with {t1}, {t2}, {t3} quality and {pitch} pitch.",
                f"{age.capitalize()}, {gender} voice characterized by {pitch} pitch with {t1}, {t2}, and {t3} tone.",
                f"The speaker's voice is {age} and {gender}, featuring {t1}, {t2}, and {t3} quality with {pitch} pitch.",
                f"A {t1}, {t2}, {t3}, and {pitch}-pitched voice, {gender} and {age} in character.",
                f"This voice has {pitch} pitch with {t1}, {t2}, and {t3} quality, sounding {age} and {gender}.",
                f"{pitch.capitalize()}-pitched, {t1}, {t2}, and {t3}, this {gender} voice {sounds_adj_age}.",
                f"The voice presents as {gender} and {age}, with {pitch} pitch that sounds {t1}, {t2}, and {t3}.",
            ]
        case t1, t2:
            templates = [
                f"This is {article_age}, {gender} voice with {pitch} pitch, {t1} and {t2} tone.",
                f"A {t1}, {t2}, {pitch}-pitched voice that is {gender} and {sounds_adj_age}.",
                f"A {t1}, {t2} voice with {pitch} pitch, {gender} and {age} in character.",
                f"{'A ' + age if age == 'young' else 'An ' + age} {gender} voice featuring {pitch} pitch with {t1} and {t2} tone.",
                f"{age.capitalize()} and {gender}, this voice features {pitch} pitch with {t1} and {t2} tone.",
                f"{age.capitalize()}, {gender} voice characterized by {pitch} pitch with {t1} and {t2} tone.",
                f"This {age}, {gender} voice has {pitch} pitch with {t1} and {t2} characteristics.",
                f"{gender.capitalize()} speaker who is {'an ' + age if age == 'adult' else age}, displaying {pitch} pitch with {t1} and {t2} characteristics.",
                f"The voice is {gender} and {age}, with {pitch} pitch, {t1} and {t2} timbre.",
                f"This voice has {pitch} pitch, sounding {age} and {gender}, with {t1} and {t2} tone.",
                f"{age.capitalize()} {gender} voice with {pitch} pitch, {t1} and {t2} qualities.",
                f"{gender.capitalize()} voice that {sounds_adj_age}, {pitch}-pitched, {t1} and {t2}.",
                f"The speaker's voice is {age} and {gender}, {t1} and {t2}, with {pitch} pitch.",
                f"{pitch.capitalize()}-pitched, {t1}, and {t2}, this {gender} voice {sounds_adj_age}.",
                f"The voice presents as {gender} and {age}, {pitch}-pitched, featuring {t1} and {t2} qualities.",
            ]
        case (t1,):
            templates = [
                f"This is {article_age}, {gender} voice with {pitch} pitch, sounding {t1}.",
                f"{'A ' + age if age == 'young' else 'An ' + age} {gender} voice featuring {pitch} pitch and {t1} timbre.",
                f"The voice is {gender} and {age}, with {pitch} pitch, characterized by {t1} tone.",
                f"This {age}, {gender} voice has {pitch} pitch and {t1} character.",
                f"A {t1}, {pitch}-pitched voice that is {gender} and {sounds_adj_age}.",
                f"{gender.capitalize()} voice that {sounds_adj_age}, with {pitch} pitch and {t1} texture.",
                f"{age.capitalize()} and {gender}, this voice features {pitch} pitch with {t1} tone.",
                f"{gender.capitalize()} speaker who is {'an ' + age if age == 'adult' else age}, displaying {pitch} pitch and {t1} characteristic.",
                f"{age.capitalize()} {gender} voice with {t1} tone and {pitch} pitch.",
                f"{age.capitalize()}, {gender} voice characterized by {pitch} pitch and {t1} tone.",
                f"The speaker's voice is {age} and {gender}, featuring {t1} tone with {pitch} pitch.",
                f"A {t1} and {pitch}-pitched voice, {gender} and {age} in character.",
                f"This voice has {pitch} pitch and {t1} tone, sounding {age} and {gender}.",
                f"{pitch.capitalize()}-pitched and {t1}, this {gender} voice {sounds_adj_age}.",
                f"The voice presents as {gender} and {age}, with {pitch} pitch that is {t1} quality.",
            ]
        case _:
            # 14 templates, reviewed
            templates = [
                f"This is {article_age}, {gender} voice with {pitch} pitch.",
                f"{'A ' + age if age == 'young' else 'An ' + age} {gender} voice featuring {pitch} pitch.",
                f"The voice is {gender} and {age}, with {pitch} pitch.",
                f"This {age}, {gender} voice has {pitch} pitch.",
                f"A {pitch}-pitched voice that is {gender} and {sounds_like_age}.",
                f"{gender.capitalize()} voice that {sounds_like_age}, with {pitch} pitch.",
                f"{age.capitalize()} and {gender}, this voice features {pitch} pitch.",
                f"{gender.capitalize()} speaker who is {'an ' + age if age == 'adult' else age}, displaying {pitch} pitch.",
                f"{age.capitalize()} {gender} voice with {pitch} pitch.",
                f"{age.capitalize()}, {gender} voice characterized by {pitch} pitch.",
                f"The speaker's voice is {age} and {gender}, with {pitch} pitch.",
                f"A {pitch}-pitched voice, {gender} and {age} in character.",
                f"This voice has {pitch} pitch, sounding {age} and {gender}.",
                f"{pitch.capitalize()}-pitched, this {gender} voice {sounds_adj_age}.",
                f"The voice presents as {gender} and {age}, with {pitch} pitch.",
            ]

    if template_id == -1:
        template_id = random.randint(0, len(templates) - 1)
    return templates[template_id]


def generate_style_description_templated(
    speed: Literal["slow", "normal", "fast"],
    loudness: Literal["quiet", "normal", "loud"],
    arousal: Literal["low", "neutral", "high"],
    valence: Literal["negative", "neutral", "positive"],
    template_id: int = -1,
):
    SPEED_ADVERB = {
        "slow": "slowly",
        "normal": "at normal speed",
        "fast": "quickly",
    }
    LOUDNESS_ADVERB = {
        "quiet": "quietly",
        "normal": "at normal volume",
        "loud": "loudly",
    }
    templates = [
        f"This is spoken {SPEED_ADVERB[speed]} with {arousal} arousal and {valence} valence, at a {loudness} volume.",
        f"Speaking {SPEED_ADVERB[speed]} and {LOUDNESS_ADVERB[loudness]} with {arousal} arousal and {valence} sentiment.",
        f"A {speed} pace with {loudness} volume, {arousal} arousal, and {valence} valence.",
        f"Speaks {SPEED_ADVERB[speed]} and {LOUDNESS_ADVERB[loudness]}, with {arousal} arousal and {valence} valence.",
        f"Delivered with {arousal} arousal and {valence} sentiment, paced {SPEED_ADVERB[speed]}, at a {loudness} volume.",
        f"A {speed} pace, {loudness} volume with {arousal} arousal and {valence} valence.",
        f"The arousal is {arousal}, the valence {valence}, the pace {speed}, the volume {loudness}.",
        f"{speed.capitalize()} tempo, {loudness} volume, {arousal} arousal, {valence} valence.",
        f"Spoken with {arousal} arousal and {valence} tone, {f'{speed}-paced' if speed != 'normal' else 'normally paced'}, {loudness} volume.",
        f"This has {arousal} arousal and {valence} valence, spoken {SPEED_ADVERB[speed]} and {LOUDNESS_ADVERB[loudness]}.",
        f"{LOUDNESS_ADVERB[loudness].capitalize()} and {SPEED_ADVERB[speed]}, with {arousal} arousal and {valence} valence.",
        f"Speech characterized by {f'{speed} pacing' if speed == 'normal' else f'{speed}-paced'}, {arousal} arousal, {valence} valence, and {loudness} volume.",
        f"{arousal.capitalize()} arousal, {valence} valence {f'and {speed} delivery' if speed != 'normal' else 'delivery at normal speed'}, spoken {LOUDNESS_ADVERB[loudness]}.",
        f"Speech is expressed {SPEED_ADVERB[speed]}, {LOUDNESS_ADVERB[loudness]}, with {arousal} arousal and {f'{valence} feeling' if valence != 'neutral' else 'neutral valence'}.",
        f"It is said {SPEED_ADVERB[speed]} with {arousal} arousal and {f'{valence} feeling' if valence != 'neutral' else 'neutral valence'}, keeping it {f'{loudness}' if loudness != 'normal' else 'at normal volume'}.",
        f"Speech is delivered {SPEED_ADVERB[speed]} at {loudness} volume, {arousal} arousal, {valence} valence.",
        f"A {arousal} arousal, {valence} valence, {f'{speed}-paced' if speed != 'normal' else 'normally paced'} utterance at {loudness} volume.",
        f"With {loudness} volume and {f'{speed}-paced' if speed != 'normal' else 'normally paced'} delivery, showing {arousal} arousal and {valence} valence.",
        f"The speech carries {arousal} arousal and {valence} tone, moving {SPEED_ADVERB[speed]} and {LOUDNESS_ADVERB[loudness]}.",
        f"Speech with {arousal} arousal and {valence} mood, delivered {SPEED_ADVERB[speed]} and {LOUDNESS_ADVERB[loudness]}.",
    ]

    if template_id == -1:
        template_id = random.randint(0, len(templates) - 1)
    return templates[template_id]


# Structured versions
def generate_voice_description_structured(
    age: Literal["young", "adult", "old"],
    gender: Literal["male", "female", "gender-neutral"],
    pitch: Literal["low", "medium", "high"],
    textures: list[str] | None = None,
):
    if textures is None:
        textures = []

    feature_list = [age, gender, f"{pitch} pitch"]
    texture = ", ".join(textures) + " texture"

    description = "Voice: " + ", ".join(feature_list)

    if len(textures):
        description = description + ", and " + texture

    description = description + "."

    return description


def generate_style_description_structured(
    speed: Literal["slow", "normal", "fast"],
    loudness: Literal["quiet", "normal", "loud"],
    arousal: Literal["low", "neutral", "high"],
    valence: Literal["negative", "neutral", "positive"],
):
    feature_list = [
        f"{speed} speed",
        f"{loudness} loudness" if loudness == "normal" else loudness,
        f"{arousal} arousal",
        f"{valence} valence",
    ]

    description = "Style: " + ", ".join(feature_list) + "."

    return description
