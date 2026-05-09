"""
This script is used to parse the paths of files and get the information
of each recording and determine the following:

Modality (01 = full-AV, 02 = video-only, 03 = audio-only).
Vocal channel (01 = speech, 02 = song).
Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the 'neutral' emotion.
Statement (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door").
Repetition (01 = 1st repetition, 02 = 2nd repetition).
Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).
"""

from pathlib import Path
import pandas as pd

# Here is the dictionary that will be used as map between the identifiers and the categories
MODALITY = {1: "full-AV", 2: "video-only", 3: "audio-only"}
CHANNEL = {1: "speech", 2: "song"}
EMOTION = {
    1: "neutral", 2: "calm", 3: "happy", 4: "sad",
    5: "angry", 6: "fearful", 7: "disgust", 8: "surprised",
}
INTENSITY = {1: "normal", 2: "strong"}
STATEMENT = {
    1: "Kids are talking by the door",
    2: "Dogs are sitting by the door",
}



def parse_filename( filename: str) -> dict:
    """
    This function is used to parse a single RAVDESS filename into its metadaya fileds.

    Parameters
    ----------
    filename: str
    Filename or full path. However, the basename is the only thing that is parsed.
    
    Returns
    -------
    dict
        Parset metadat, including string labels, gender (derived from actor parity)
        and the interger emotion code (useful as a label for ML models)
    """
    stem = Path(filename).stem
    parts = stem.split("-")
    if len(parts) != 7:
        raise ValueError ( f"Expected 7 dash parts, but got {len(parts)}" )
    codes = [int(p) for p in parts]
    modality, channel, emotion, intensity, statement, repetition, actor = codes
    gender = "male" if actor %2 ==1 else "female"

    return {
        "filename": Path(filename).name,
        "modality": MODALITY[modality],
        "channel": CHANNEL[channel],
        "emotion": EMOTION[emotion],
        "emotion_code": emotion,
        "intensity": INTENSITY[intensity],
        "statement": STATEMENT[statement],
        "repetition": repetition,
        "actor": actor,
        "gender": gender,
    }
