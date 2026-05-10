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


