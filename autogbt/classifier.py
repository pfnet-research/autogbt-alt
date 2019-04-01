import numpy as np
import copy
from autogbt import logging
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.testing import SkipTest
from sklearn.utils.validation import check_array, check_is_fitted
from .trainer import create_trainer, default_n_jobs
from .optimizer import create_optimizer, default_n_trials
from .validation import validate_dataset


class AutoGBTClassifier(BaseEstimator, ClassifierMixin):

    def __init__(
        self,
        n_trials=default_n_trials,
        objective=None,
        sampler=None,
        n_jobs=default_n_jobs,
        cv=None,
        create_validation=True,
        random_state=None,
    ):
        """
        Args:
            n_trials: int
                the number of trials on hyperparameter optimization
            objective: Objective
                objective for optimizer
            sampler:
                sampling strategy for training data
            n_jobs:
                the number of CPUs to use to do the computation
            cv:
                cross-validation splitting strategy
            create_validation:
                If True, create a valid dataset from train split for early
                stopping. Otherwise, use valid split for early stopping.
            random_state:
                it is used by the random number generator
        """
        self.n_trials = n_trials
        self.objective = objective
        self.sampler = sampler
        self.n_jobs = n_jobs
        self.cv = cv
        self.create_validation = create_validation
        self.random_state = random_state

    def fit(self, X, y):
        logger = logging.get_logger(__name__)
        trainer = create_trainer(
            objective='binary',
            metric='auc',
            sampler=self.sampler,
            n_jobs=self.n_jobs,
            create_validation=self.create_validation,
            cv=self.cv,
            random_state=self.random_state,
        )
        optimizer = create_optimizer(
            objective=self.objective,
            trainer=trainer,
            n_trials=self.n_trials,
            random_state=self.random_state,
        )
        X, y = validate_dataset(optimizer, X, y)
        if len(y.unique()) != 2:
            raise SkipTest('binary classification is only supported')

        logger.info('start optimization')
        optimizer.optimize(X, y)
        self._optimizer = optimizer

    def predict_proba(self, X):
        check_is_fitted(self, ['_optimizer'])

        if len(X) == 0:
            return np.zeros((0))

        X = check_array(X)

        return self._optimizer.best_model.predict(X)

    def predict(self, X):
        pred = self.predict_proba(X)
        return (pred >= 0.5).astype('i')

    def get_params(self, deep=True):
        params = {
            'n_trials': self.n_trials,
            'objective': self.objective,
            'sampler': self.sampler,
            'n_jobs': self.n_jobs,
            'cv': self.cv,
            'create_validation': self.create_validation,
            'random_state': self.random_state,
        }
        return copy.deepcopy(params) if deep else params

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    @property
    def best_score(self):
        return 1 - self._optimizer.best_score
