import numpy as np
import pandas as pd
from autogbt.sampler import MajorityUnderSampler


def _test_sample(y):
    sampler = MajorityUnderSampler()
    idx = sampler.sample(y, 40000, 3.0)
    assert len(idx) == 40000
    assert y[idx].sum() == 10000


def test_sample_with_series():
    y = pd.Series(np.concatenate([np.ones((10000)), np.zeros((100000))]))
    y = y.sample(frac=1.0)
    _test_sample(y)


def test_sample_with_ndarray():
    y = np.concatenate([np.ones((10000)), np.zeros((100000))])
    _test_sample(y)


def test_sample_for_regression():
    y = np.concatenate([
        2*np.ones((10000)),
        1*np.ones((10000)),
        0*np.ones((10000)),
    ])
    sampler = MajorityUnderSampler()
    idx = sampler.sample(y, 0.1, 3.0)
    assert len(idx) == 3000
