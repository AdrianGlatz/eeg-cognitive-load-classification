"""Build EEG datasets for machine learning."""

import numpy as np

from .preprocessing import load_and_epoch
from .features import build_feature_matrix

TASKS = {
    "zeroBACK": {"event_code": "6021", "label": 0},
    "oneBACK": {"event_code": "6121", "label": 1},
    "twoBACK": {"event_code": "6221", "label": 2}
}

SESSIONS = ["ses-S1", "ses-S2", "ses-S3"]

def build_session_dataset(eeg_dir):
    """Build X and y for one EEG session."""
    X_parts = []
    y_parts = []

    for task_name, task_info in TASKS.items():
        file_path = eeg_dir / f"{task_name}.set"

        event_code = task_info["event_code"]
        epochs = load_and_epoch(file_path, event_code)
        X_task = build_feature_matrix(epochs)

        label = task_info["label"]
        y_task = np.full(len(X_task), label)

        X_parts.append(X_task)
        y_parts.append(y_task)

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)

    return X, y

def build_subject_dataset(subject_dir):
    """Build X and y for one EEG subject."""

    X_parts = []
    y_parts = []
    for session_name in SESSIONS:
        eeg_dir = subject_dir / session_name / "eeg"

        X_session, y_session = build_session_dataset(eeg_dir)

        X_parts.append(X_session)
        y_parts.append(y_session)

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)

    return X, y


def build_dataset(data_dir, subject_names):
    """Build X, y, and subject groups for the full dataset."""

    X_parts = []
    y_parts = []
    group_parts = []

    for subject_name in subject_names:
        subject_dir = data_dir / subject_name

        X_subject, y_subject = build_subject_dataset(subject_dir)

        groups_subject = np.full(len(X_subject), subject_name)

        X_parts.append(X_subject)
        y_parts.append(y_subject)
        group_parts.append(groups_subject)

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    groups = np.concatenate(group_parts)

    return X, y, groups