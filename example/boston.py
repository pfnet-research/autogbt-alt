import argparse
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from autogbt import AutoGBTRegressor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-trials', type=int)
    args = parser.parse_args()

    X, y = load_boston(return_X_y=True)
    train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size=0.1)
    model = AutoGBTRegressor(n_trials=args.n_trials)
    model.fit(train_X, train_y)
    print('valid MSE: %.3f' % (
        mean_squared_error(valid_y, model.predict(valid_X))))
    print('CV MSE: %.3f' % (model.best_score))


if __name__ == '__main__':
    main()
