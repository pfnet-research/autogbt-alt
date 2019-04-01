import pytest
import numpy as np
import pandas as pd
from autogbt.sampler import TrainDataSampler


def test_init():
    TrainDataSampler()
    # train
    TrainDataSampler(train_frac=0.5)
    TrainDataSampler(train_size=200)
    with pytest.raises(AttributeError):
        TrainDataSampler(train_frac=0.5, train_size=200)
    # valid
    TrainDataSampler(valid_frac=0.5)
    TrainDataSampler(valid_size=200)
    with pytest.raises(AttributeError):
        TrainDataSampler(valid_frac=0.5, valid_size=200)


def test_sample():
    X = pd.DataFrame(np.zeros((2000, 64)))
    y = pd.Series(np.zeros((2000)))
    train_idx = np.arange(1000)
    valid_idx = np.arange(1000) + 1000
    s = TrainDataSampler(train_size=100, valid_size=200, majority_frac=None)
    train_X, train_y, valid_X, valid_y = s.sample(X, y, train_idx, valid_idx)
    assert train_X.shape[0] == 100
    assert valid_X.shape[0] == 200

    s = TrainDataSampler(train_frac=0.1, valid_frac=0.2, majority_frac=None)
    train_X, train_y, valid_X, valid_y = s.sample(X, y, train_idx, valid_idx)
    assert train_X.shape[0] == 100
    assert valid_X.shape[0] == 200
