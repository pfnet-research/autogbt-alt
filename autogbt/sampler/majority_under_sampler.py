import numpy as np
import pandas as pd
from sklearn.utils import check_random_state


class MajorityUnderSampler:

    def __init__(self, random_state=None):
        self.random_state = check_random_state(random_state)

    def _get_sample_index(self, y, majority_frac):
        if isinstance(y, pd.Series):
            indices = y.index
        else:
            indices = np.arange(0, len(y))

        if len(np.unique(y)) != 2:
            return indices

        if majority_frac is None:
            return indices

        class_0_freq = len(y[y == 0])
        class_1_freq = len(y[y == 1])

        if class_1_freq > class_0_freq:
            majority_class = 1
            minority_count = class_0_freq
        else:
            majority_class = 0
            minority_count = class_1_freq

        minority_class = 1 - majority_class

        majority_index = indices[y == majority_class]
        minority_index = indices[y == minority_class]

        if int(minority_count*majority_frac) > len(majority_index):
            majority_size = len(majority_index)
        else:
            majority_size = int(minority_count*majority_frac)

        # downsample majority
        majority_index = self.random_state.choice(
            majority_index, size=majority_size, replace=False)
        sample_index = np.concatenate(
            [minority_index, majority_index]).tolist()
        return sample_index

    def sample(self, y, data_frac, majority_frac):
        index = self._get_sample_index(y, majority_frac)

        # extra sampling for large dataset
        size = int(len(index)*data_frac)
        if len(index) > size:
            index = self.random_state.choice(index, size=size, replace=False)

        return index
