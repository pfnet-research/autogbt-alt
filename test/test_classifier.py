from sklearn.utils.estimator_checks import check_estimator
from autogbt.classifier import AutoGBTClassifier


def test_check_estimator():
    check_estimator(AutoGBTClassifier)
