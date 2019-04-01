from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from autogbt import AutoGBTClassifier, Objective


class CustomObjective(Objective):

    def get_param(self, trial):
        param = super(CustomObjective, self).get_param(trial)
        param.update({
            'lambda_l1': trial.suggest_uniform('lambda_l1', 0.0, 5.0),
            'lambda_l2': trial.suggest_uniform('lambda_l2', 0.0, 5.0),
        })
        return param


def main():
    X, y = load_breast_cancer(return_X_y=True)
    train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size=0.1)
    model = AutoGBTClassifier(n_trials=5, objective=CustomObjective())
    model.fit(train_X, train_y)
    print('valid AUC: %.3f' % (roc_auc_score(valid_y, model.predict(valid_X))))
    print('CV AUC: %.3f' % (model.best_score))


if __name__ == '__main__':
    main()
