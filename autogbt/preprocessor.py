import numpy as np
import pandas as pd
from sklearn.utils import check_random_state
from .sampler import MajorityUnderSampler
from . import logging


def _datetime_columns(df):
    columns = [
        c for c, t in zip(df.columns, df.dtypes)
        if np.issubdtype(t, np.datetime64)
    ]
    return columns


def _extract_datetime_difference(df):
    columns = _datetime_columns(df)
    for i in range(len(columns)):
        for j in range(i+1, len(columns)):
            new = 'diff_%s_%s' % (columns[i], columns[j])
            df[new] = (df[columns[i]] - df[columns[j]]).astype('int64')


def _extract_datetime_features(df):
    columns = _datetime_columns(df)
    attributes = [
        'dayofweek',
        'dayofyear',
        'month',
        'weekofyear',
        'day',
        'hour',
        'minute',
        'year',
    ]
    for c in columns:
        for a in attributes:
            new = '%s_%s' % (c, a)
            df[new] = getattr(df[c].dt, a)


class Preprocessor:

    def __init__(
        self,
        train_frac=None,
        train_size=None,
        test_frac=None,
        test_size=None,
        majority_frac=3.0,
        random_state=None,
    ):
        self._init_train_param(train_frac, train_size)
        self._init_test_param(test_frac, test_size)
        self.majority_frac = majority_frac
        self.random_state = check_random_state(random_state)
        self.categorical_encoders = {}
        self.datetime_encoders = {}
        self.sampler = MajorityUnderSampler(self.random_state)

    def _init_train_param(self, train_frac, train_size):
        if train_frac is None and train_size is None:
            train_size = 1000000
        elif train_frac is None or train_size is None:
            pass
        else:
            raise AttributeError(
                'both of train_frac and train_size are specified')
        self.train_frac = train_frac
        self.train_size = train_size

    def _init_test_param(self, test_frac, test_size):
        if test_frac is None and test_size is None:
            test_size = 1000000
        elif test_frac is None or test_size is None:
            pass
        else:
            raise AttributeError(
                'both of test_frac and test_size are specified')
        self.test_frac = test_frac
        self.test_size = test_size

    def _get_train_frac(self, size):
        if self.train_frac is not None:
            return self.train_frac

        train_frac = self.train_size / size
        return min(1.0, train_frac)

    def _get_test_size(self, size):
        if self.test_size is not None:
            return min(size, self.test_size)

        return int(size*self.test_frac)

    def transform(self, train_X, test_X, train_y):
        if not isinstance(train_X, pd.DataFrame):
            train_X = pd.DataFrame(train_X)
        if not isinstance(test_X, pd.DataFrame):
            test_X = pd.DataFrame(test_X)
        if not isinstance(train_y, pd.Series):
            train_y = pd.Series(train_y)

        logger = logging.get_logger(__name__)

        logger.info('fillna')
        self._fillna(train_X)
        self._fillna(test_X)

        logger.info('extract_datetime_features')
        _extract_datetime_difference(train_X)
        _extract_datetime_difference(test_X)
        _extract_datetime_features(train_X)
        _extract_datetime_features(test_X)

        logger.info('prepare datetime encoding')
        self._prepareDatetimeEncoding(train_X)
        self._prepareDatetimeEncoding(test_X)

        logger.info('encode datetime feature')
        self._transform_datetime(train_X)
        self._transform_datetime(test_X)

        logger.info('sample for frequency encoding')
        train_frac = self._get_train_frac(train_y.shape[0])
        idx = self.sampler.sample(train_y, train_frac, self.majority_frac)
        sampled_train_X = train_X.loc[idx]
        test_size = self._get_test_size(test_X.shape[0])
        sampled_test_X =\
            test_X.sample(n=test_size, random_state=self.random_state)
        X = pd.concat(
            [sampled_train_X, sampled_test_X], sort=False)
        X = X.reset_index(drop=True)
        logger.info('sampled datasize %s' % str(X.shape))

        logger.info('prepare for frequency encoding')
        self._prepareFrequencyEncoding(X)
        del X

        logger.info('encode categorical feature')
        self._transform_categorical(train_X)
        self._transform_categorical(test_X)

        return train_X, test_X, train_y

    def _prepareFrequencyEncoding(self, X):
        columns = [c[0] for c in zip(X.columns, X.dtypes) if c[1] == 'object']
        for c in columns:
            self.categorical_encoders[c] = dict(X[c].value_counts())

    def _prepareDatetimeEncoding(self, X):
        columns = _datetime_columns(X)
        for c in columns:
            if c not in self.datetime_encoders:
                self.datetime_encoders[c] = X[c].min()
            else:
                self.datetime_encoders[c] = min(
                    self.datetime_encoders[c], X[c].min())

    def _transform_datetime(self, X):
        for c, min_date in self.datetime_encoders.items():
            X[c] = (X[c] - min_date).astype('int64')

    def _transform_categorical(self, X):
        for c, encoder in self.categorical_encoders.items():
            X[c] = X[c].map(encoder)
            X[c].fillna(0, inplace=True)
        return X

    def _fillna(self, X):
        for c, t in zip(X.columns, X.dtypes):
            if t == 'object':
                X[c].fillna('nan', inplace=True)
            else:
                X[c].fillna(0, inplace=True)
