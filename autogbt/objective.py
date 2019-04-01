class Objective:

    trainer = None

    def get_param(self, trial):
        return {
            'num_leaves': trial.suggest_int('num_leaves', 40, 80),
            'learning_rate': trial.suggest_loguniform(
                'learning_rate', 0.005, 0.015),
            'feature_fraction': trial.suggest_uniform(
                'feature_fraction', 0.5, 0.7),
            'bagging_fraction': trial.suggest_uniform(
                'bagging_fraction', 0.5, 0.7),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 3),
        }

    def set_trainer(self, trainer):
        self.trainer = trainer

    def __call__(self, trial):
        if self.trainer is None:
            raise RuntimeError('`trainer` should be setby using `set_trainer`')

        param = self.get_param(trial)
        score = self.trainer.train(trial.trial_id, param)
        return score
