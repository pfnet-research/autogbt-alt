from sklearn.utils import check_random_state
from autogbt import logging
from .majority_under_sampler import MajorityUnderSampler


class TrainDataSampler:

    def __init__(
        self,
        train_frac=None,
        train_size=None,
        valid_frac=None,
        valid_size=None,
        majority_frac=3.0,
        random_state=None,
    ):
        self._init_train_param(train_frac, train_size)
        self._init_valid_param(valid_frac, valid_size)
        self.majority_frac = majority_frac
        self.random_state = check_random_state(random_state)
        self.sampler = MajorityUnderSampler(random_state=random_state)

    def _init_train_param(self, train_frac, train_size):
        if train_frac is None and train_size is None:
            train_size = 500000
        elif train_frac is None or train_size is None:
            pass
        else:
            raise AttributeError(
                'both of train_frac and train_size are specified')
        self.train_frac = train_frac
        self.train_size = train_size

    def _init_valid_param(self, valid_frac, valid_size):
        if valid_frac is None and valid_size is None:
            valid_size = 500000
        elif valid_frac is None or valid_size is None:
            pass
        else:
            raise AttributeError(
                'both of valid_frac and valid_size are specified')
        self.valid_frac = valid_frac
        self.valid_size = valid_size

    def _get_train_frac(self, size):
        if self.train_frac is not None:
            return self.train_frac

        train_frac = self.train_size / size
        return min(1.0, train_frac)

    def _get_valid_size(self, size):
        if self.valid_size is not None:
            return min(size, self.valid_size)

        return int(size*self.valid_frac)

    def sample(self, X, y, train_idx, valid_idx):
        logger = logging.get_logger(__name__)
        train_y = y.loc[train_idx].reset_index(drop=True)
        train_frac = self._get_train_frac(train_idx.shape[0])
        idx = self.sampler.sample(train_y, train_frac, self.majority_frac)
        if len(idx) > 0:
            train_idx = train_idx[idx]

        train_X = X.loc[train_idx].reset_index(drop=True)
        train_y = y.loc[train_idx].reset_index(drop=True)
        logger.info(
            'downsampled training data length=%d' % len(train_X))

        valid_size = self._get_valid_size(valid_idx.shape[0])
        valid_idx = self.random_state.choice(
            valid_idx, size=valid_size, replace=True)
        valid_X = X.loc[valid_idx].reset_index(drop=True)
        valid_y = y.loc[valid_idx].reset_index(drop=True)
        return train_X, train_y, valid_X, valid_y
