import mne

def load_and_epoch(file_path, event_code):
    raw = mne.io.read_raw_eeglab(file_path, preload=True)

    events, event_id = mne.events_from_annotations(raw)

    raw_eeg = raw.copy().pick("eeg")
    raw_eeg.filter(l_freq=1.0, h_freq=40.0)

    trial_event = event_id[event_code]

    epochs = mne.Epochs(
        raw_eeg,
        events,
        event_id=trial_event,
        tmin=0,
        tmax=2.0,
        baseline=None,
        preload=True
    )
    return epochs