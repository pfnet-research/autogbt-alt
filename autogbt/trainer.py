import contextlib
import lightgbm as lgb
from pathlib import Path
from tempfile import TemporaryDirectory
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import roc_auc_score, mean_squared_error
from .average import AveragingLGBMClassifier
from .sampler import TrainDataSampler
default_n_jobs = 1


class GBTCVTrainer:

    X = None
    y = None
    work_dir = None

    def __init__(
        self, objective, metric, sampler, n_jobs, create_validation,
        cv=None, random_state=None,
    ):
        if cv is None:
            cv = KFold(
                n_splits=5, shuffle=True, random_state=random_state)
        if sampler is None:
            sampler = TrainDataSampler()
        n_jobs = n_jobs or default_n_jobs

        self.objective = objective
        self.metric = metric
        self.n_jobs = n_jobs
        self.cv = cv
        self.sampler = sampler
        self.create_validation = create_validation
        self.random_state = random_state

    def is_valid_dataset(self, X, y):
        if len(X) < self.cv.n_splits:
            return False

        return True

    @contextlib.contextmanager
    def dataset(self, X, y):
        self.X, self.y = X, y
        with TemporaryDirectory() as work_dir:
            self.work_dir = Path(work_dir)
            yield
        self.work_dir = None
        self.X, self.y = None, None

    def get_model(self, trial_id):
        model_dir = self.work_dir/str(trial_id)
        models = []
        for model_path in model_dir.glob('*.lgbm'):
            model = lgb.Booster(model_file=str(model_path))
            models.append(model)
        return AveragingLGBMClassifier(models)

    def _store_model(self, trial_id, fold, bst):
        model_dir = self.work_dir/str(trial_id)
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir/('%d.lgbm' % fold)
        bst.save_model(str(model_path))

    def _score(self, true_y, pred_y):
        if self.objective == 'binary':
            return 1 - roc_auc_score(true_y, pred_y)
        elif self.objective == 'regression':
            return mean_squared_error(true_y, pred_y)
        else:
            ValueError('invalid objective')

    def train(self, trial_id, param):
        base_param = {
            'objective': self.objective,
            'n_jobs': self.n_jobs,
            'metric': self.metric,
            'verbosity': -1,
            'boosting_type': 'gbrt',
        }
        param.update(base_param)
        X, y = self.X, self.y
        pred_y = y.copy()
        for fold, (train_idx, test_idx) in enumerate(self.cv.split(X, y)):
            if self.create_validation:
                train_idx, valid_idx = train_test_split(
                    train_idx, random_state=self.random_state)
            else:
                valid_idx = test_idx

            train_X, train_y, valid_X, valid_y =\
                self.sampler.sample(X, y, train_idx, valid_idx)

            dtrain = lgb.Dataset(train_X, label=train_y)
            dvalid = lgb.Dataset(valid_X, label=valid_y)

            bst = lgb.train(
                param,
                dtrain,
                valid_sets=[dtrain, dvalid],
                valid_names=['train', 'valid'],
                num_boost_round=1000,
                early_stopping_rounds=100,
                verbose_eval=100,
            )
            self._store_model(trial_id, fold, bst)
            pred_y.loc[test_idx] = bst.predict(X.loc[test_idx])

        score = self._score(y, pred_y)
        return score


def create_trainer(
    objective,
    metric,
    sampler,
    n_jobs,
    cv,
    create_validation,
    random_state,
):
    trainer = GBTCVTrainer(
        objective=objective,
        metric=metric,
        sampler=sampler,
        n_jobs=n_jobs,
        create_validation=create_validation,
        cv=cv,
        random_state=random_state,
    )
    return trainer
