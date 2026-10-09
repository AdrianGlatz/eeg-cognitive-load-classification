"""ML models for cognitive load classification."""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

CV_SPLITS = 5


def build_model():
    """Build the cognitive load classification model."""

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )

    return model


def cross_validation_predictions(X, y, groups):
    """Generate subject-wise cross-validation predictions."""

    model = build_model()
    cv = GroupKFold(n_splits=CV_SPLITS)

    predictions = cross_val_predict(model, X, y, groups=groups, cv=cv)

    return predictions


def evaluate_predictions(y, predictions):
    """Evaluate classification predictions."""

    accuracy = accuracy_score(y, predictions)

    f1 = f1_score(y, predictions, average="macro")

    matrix = confusion_matrix(y, predictions)

    return accuracy, f1, matrix
