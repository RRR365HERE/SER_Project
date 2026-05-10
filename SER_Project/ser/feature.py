"""
In this file, I am computing frame-level features using librosa, then
summarize each over time with mean and standard deviation. Why this this exactly?
This produces one fixed-length vector per file that any classical ML model can consume
"""

from pathlib import Path
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm


# Hyperparameters
SAMPLE_RATE = 22050 # while RAVDESS is 4kHZ, after searching online about similar problems, i understood that it could be haved to double the extraction speed
N_MFCC = 13 # Got it from similar problems after searching
N_FFT = 2048
HOP_LENGTH = 512
TRIM_TOP_DB = 30

def _summarize( arr: np.ndarray ) -> np.ndarray:
    """
    Reduce each frame feature to mean, std across the time axis.

    Parameters
    ----------
    arr : np.ndarray
        It could be 1D of shape (n_frames,) or 2D of shape(n_features, n_frames).
    
    Returns
    -------
    np.ndarray
        Concatenation of means then stds.
    """

    if arr.ndim == 1:
        return np.array([arr.mean(), arr.std()])
    return np.concatenate([arr.mean(axis=1), arr.std(axis=1)])


def extract_features(audio_path, sr: int = SAMPLE_RATE,
                     n_mfcc: int = N_MFCC, trim: bool = True) -> np.ndarray:
    """Extract a fixed-length feature vector from one audio file.

    Pipeline:
      1. Load audio at `sr` Hz.
      2. Trim leading/trailing silence (RAVDESS files start with ~1s of silence).
      3. Compute frame-level features.
      4. Summarize each with mean and standard deviation.
      5. Concatenate into one vector.

    Returns
    -------
    np.ndarray of shape (112,) when n_mfcc=13.
    """
    y, sr = librosa.load(str(audio_path), sr=sr)

    if trim:
        y, _ = librosa.effects.trim(y, top_db=TRIM_TOP_DB)

    # Cepstral features: shape of the spectral envelope
    mfcc        = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc,
                                       n_fft=N_FFT, hop_length=HOP_LENGTH)
    delta_mfcc  = librosa.feature.delta(mfcc, order=1)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)

    # Tonal content
    chroma = librosa.feature.chroma_stft(y=y, sr=sr,
                                         n_fft=N_FFT, hop_length=HOP_LENGTH)

    # Energy and dynamics (1D per-frame signals)
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=HOP_LENGTH)[0]
    rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)[0]

    # Spectral shape (1D per-frame signals)
    centroid  = librosa.feature.spectral_centroid(y=y, sr=sr,
                                                  hop_length=HOP_LENGTH)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr,
                                                   hop_length=HOP_LENGTH)[0]
    rolloff   = librosa.feature.spectral_rolloff(y=y, sr=sr,
                                                 hop_length=HOP_LENGTH)[0]

    return np.concatenate([
        _summarize(mfcc),         # 26
        _summarize(delta_mfcc),   # 26
        _summarize(delta2_mfcc),  # 26
        _summarize(chroma),       # 24
        _summarize(zcr),          # 2
        _summarize(rms),          # 2
        _summarize(centroid),     # 2
        _summarize(bandwidth),    # 2
        _summarize(rolloff),      # 2
    ])  # total = 112


def get_feature_names(n_mfcc: int = N_MFCC) -> list[str]:
    """Return the ordered column names matching extract_features() output."""
    names = []
    # Cepstral blocks
    for block in ["mfcc", "delta_mfcc", "delta2_mfcc"]:
        for stat in ["mean", "std"]:
            for i in range(n_mfcc):
                names.append(f"{block}{i+1}_{stat}")
    # Chroma
    for stat in ["mean", "std"]:
        for i in range(12):
            names.append(f"chroma{i+1}_{stat}")
    # Scalar features
    for feat in ["zcr", "rms", "centroid", "bandwidth", "rolloff"]:
        for stat in ["mean", "std"]:
            names.append(f"{feat}_{stat}")
    return names

def extract_dataset_features(metadata_df: pd.DataFrame,
                             output_csv=None,
                             sr: int = SAMPLE_RATE,
                             n_mfcc: int = N_MFCC) -> pd.DataFrame:
    """Run extract_features on every row of metadata_df and return a combined DataFrame.

    Parameters
    ----------
    metadata_df : pd.DataFrame
        Must contain a 'path' column with absolute paths to WAV files.
    output_csv : str or Path, optional
        If provided, the combined DataFrame is saved to this CSV.

    Returns
    -------
    pd.DataFrame
        Original metadata columns + feature columns. Failed extractions get NaN rows.
    """
    feature_names = get_feature_names(n_mfcc=n_mfcc)
    feature_matrix = np.full((len(metadata_df), len(feature_names)), np.nan)

    failed = []
    for i, path in enumerate(tqdm(metadata_df["path"].tolist(),
                                   desc="Extracting features")):
        try:
            feature_matrix[i] = extract_features(path, sr=sr, n_mfcc=n_mfcc)
        except Exception as e:
            failed.append((Path(path).name, str(e)))

    if failed:
        print(f"\nFailed extractions: {len(failed)}")
        for name, err in failed[:5]:
            print(f"  {name}: {err}")

    features_df = pd.DataFrame(feature_matrix, columns=feature_names)
    result = pd.concat([metadata_df.reset_index(drop=True), features_df], axis=1)

    if output_csv:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_csv, index=False)
        print(f"Saved {len(result)} rows × {len(result.columns)} cols to {output_csv}")

    return result
