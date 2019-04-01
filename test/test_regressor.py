from sklearn.utils.estimator_checks import check_estimator
from autogbt import AutoGBTRegressor


def test_check_estimator():
    check_estimator(AutoGBTRegressor)
