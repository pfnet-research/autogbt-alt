import optuna
from sklearn.utils import check_random_state
from .objective import Objective
default_n_trials = 20


class OptunaOptimizer:

    best_model = None
    best_score = None

    def __init__(self, objective, trainer, n_trials, random_state=None):
        random_state = check_random_state(random_state)
        n_trials = n_trials or default_n_trials
        if objective is None:
            objective = Objective()
        self.objective = objective
        self.trainer = trainer
        self.n_trials = n_trials
        self.sampler = optuna.samplers.TPESampler(
            seed=random_state.randint(0, 2**32-1))

    def optimize(self, X, y):
        study = optuna.create_study(sampler=self.sampler)
        self.objective.set_trainer(self.trainer)
        with self.trainer.dataset(X, y):
            study.optimize(
                self.objective, n_trials=self.n_trials)
            best_trial_id = study.best_trial.trial_id
            best_model = self.trainer.get_model(best_trial_id)
            best_score = study.best_value
        self.best_model = best_model
        self.best_score = best_score


def create_optimizer(
    objective,
    n_trials,
    trainer,
    random_state,
):
    optimizer = OptunaOptimizer(
        objective=objective,
        trainer=trainer,
        n_trials=n_trials,
        random_state=random_state,
    )
    return optimizer
