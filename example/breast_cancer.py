import argparse
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from autogbt import AutoGBTClassifier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-trials', type=int)
    args = parser.parse_args()

    X, y = load_breast_cancer(return_X_y=True)
    train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size=0.1)
    model = AutoGBTClassifier(n_trials=args.n_trials)
    model.fit(train_X, train_y)
    print('valid AUC: %.3f' % (roc_auc_score(valid_y, model.predict(valid_X))))
    print('CV AUC: %.3f' % (model.best_score))


if __name__ == '__main__':
    main()
