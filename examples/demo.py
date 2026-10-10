"""Synthetic or local original-EEG algorithm demonstration."""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from eeg_dataset_explorer.signals import synthetic, signal_array, sampling_rate
from eeg_preprocessing.pipeline import bandpass, resample, average_reference, quality_flags


def main() -> None:

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, help="Local NPZ with data and fs arrays")
    parser.add_argument("--figure", type=Path, help="Optional source signal plot (dataset explorer only)")
    args = parser.parse_args()
    if args.sample:
        with np.load(args.sample, allow_pickle=False) as sample:
            data = signal_array(sample["data"])
            fs = sampling_rate(float(sample["fs"]))
        source = "Original local EEG excerpt: " + args.sample.name
    else:
        fs = 256.0
        data = synthetic(fs)
        source = "Synthetic signal: not patient data"
    print(source)

    clean = bandpass(data, fs, 0.5, 40)
    clean = resample(average_reference(clean), fs, 128)
    print("Processed dimensions:", clean.shape, "quality flags:", quality_flags(data))


if __name__ == "__main__":
    main()
