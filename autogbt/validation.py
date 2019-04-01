import numpy as np
import pandas as pd
from sklearn.utils.testing import SkipTest
from sklearn.utils.validation import check_X_y


def validate_dataset(optimizer, X, y):
    if isinstance(X, pd.DataFrame) and isinstance(y, pd.Series):
        pass
    elif isinstance(X, np.ndarray) and isinstance(y, np.ndarray):
        X, y = check_X_y(X, y)
        X = pd.DataFrame(X)
        y = pd.Series(y)
    else:
        raise SkipTest(
            'pandas.DataFrame or np.ndarray are only supported')

    if not optimizer.trainer.is_valid_dataset(X, y):
        raise SkipTest('invalid dataset')

    return X, y
