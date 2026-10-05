"""EEG loading and preprocessing utilities."""

import mne


LOW_FREQ = 1.0
HIGH_FREQ = 40.0

EPOCH_START = 0.0
EPOCH_END = 2.0


def load_and_epoch(file_path, event_code):
    """Load, filter, and epoch EEG data."""

    raw = mne.io.read_raw_eeglab(file_path, preload=True)

    events, event_id = mne.events_from_annotations(raw)

    raw_eeg = raw.copy().pick("eeg")
    raw_eeg.filter(
        l_freq=LOW_FREQ,
        h_freq=HIGH_FREQ
    )

    trial_event = event_id[event_code]

    epochs = mne.Epochs(
        raw_eeg,
        events,
        event_id=trial_event,
        tmin=EPOCH_START,
        tmax=EPOCH_END,
        baseline=None,
        preload=True
    )

    return epochs