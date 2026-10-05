"""EEG feature extraction utilities."""

import numpy as np
from scipy.signal import welch


# EEG scalp regions
ROIS = {
    "frontal": [
        "Fp1", "Fp2",
        "AF7", "AF3", "AFz", "AF4", "AF8",
        "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8"
    ],
    "central": [
        "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6",
        "C5", "C3", "C1", "C2", "C4", "C6"
    ],
    "temporal": [
        "FT9", "FT7", "FT8", "FT10",
        "T7", "T8",
        "TP7", "TP8", "TP10"
    ],
    "parietal": [
        "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6",
        "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8"
    ],
    "occipital": [
        "PO7", "PO3", "POz", "PO4", "PO8",
        "O1", "Oz", "O2"
    ]
}

# EEG frequency bands
BANDS = {
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30)
}

WELCH_SEGMENT_SECONDS = 1.0


def calculate_psd(epochs):
    """Calculate the PSD for all epochs and EEG channels."""
    data = epochs.get_data()

    nperseg = int(epochs.info["sfreq"] * WELCH_SEGMENT_SECONDS)

    freqs, psd = welch(
        data,
        fs=epochs.info["sfreq"],
        nperseg=nperseg,
        axis=-1
    )

    return freqs, psd


def build_feature_matrix(epochs):
    """Build ROI-based bandpower features for all epochs."""
    freqs, psd = calculate_psd(epochs)

    features = []

    # Map ROI channel names to indices once
    roi_indices = {
        roi_name: [epochs.ch_names.index(ch) for ch in channels]
        for roi_name, channels in ROIS.items()
    }

    for epoch_idx in range(len(epochs)):
        epoch_features = []

        for channel_indices in roi_indices.values():
            roi_psd = np.mean(
                psd[epoch_idx, channel_indices, :],
                axis=0
            )

            for fmin, fmax in BANDS.values():
                freq_mask = (freqs >= fmin) & (freqs < fmax)

                bandpower = np.trapezoid(
                    roi_psd[freq_mask],
                    freqs[freq_mask]
                )

                epoch_features.append(bandpower)

        # One epoch = 5 ROIs × 3 frequency bands = 15 features
        features.append(epoch_features)

    return np.array(features)