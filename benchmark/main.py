import time
import argparse
import pandas as pd
from pathlib import Path
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import roc_auc_score

import dataset
import autogbt
from autogbt.sampler import TrainDataSampler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--preprocess-train-frac', type=float)
    parser.add_argument('--preprocess-test-frac', type=float)
    parser.add_argument('--model-train-frac', type=float)
    parser.add_argument('--model-valid-frac', type=float)
    parser.add_argument('--result-dir', default='./result')
    parser.add_argument('--n-trials', type=int)
    parser.add_argument('--n-jobs', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    logger = autogbt.logging.get_logger()
    logger.info(args)

    model = args.model
    task = args.task
    n_trials = args.n_trials
    seed = args.seed
    n_jobs = args.n_jobs
    model_train_frac = args.model_train_frac
    model_valid_frac = args.model_valid_frac
    with open('../.git/refs/heads/master') as fp:
        commit = next(fp).strip()

    res_dir = Path(args.result_dir)/commit
    res_dir.mkdir(parents=True, exist_ok=True)

    name = '-'.join(map(str, [model, task, n_trials, model_train_frac, seed]))
    result_path = res_dir/('%s.csv' % (name))

    if result_path.exists():
        return

    res = []
    logger.info('load dataset %s' % task)
    logger.info('model %s' % model)
    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    train_X, train_y, test_X = dataset.get(task)
    start = time.time()
    prep = autogbt.Preprocessor(
        train_frac=args.preprocess_train_frac,
        test_frac=args.preprocess_test_frac,
        random_state=seed,
    )
    train_X, valid_X, train_y = prep.transform(train_X, test_X, train_y)
    if model == 'auto':
        sampler = TrainDataSampler(
            train_frac=model_train_frac,
            valid_frac=model_valid_frac,
            random_state=seed,
        )
        est = autogbt.AutoGBTClassifier(
            n_trials=n_trials,
            sampler=sampler,
            n_jobs=n_jobs,
            cv=cv,
            random_state=seed,
        )
        est.fit(train_X, train_y)
        score = est.best_score
    else:
        n_trials = 1
        model_train_frac = 1.0
        model_valid_frac = 1.0
        if model == 'xgb':
            import xgboost as xgb
            est = xgb.XGBClassifier(n_jobs=n_jobs, random_state=seed)
            pred = cross_val_predict(
                est, train_X, train_y, cv=cv, method='predict_proba')[:, 1]
            score = roc_auc_score(train_y, pred)
        if model == 'lgb':
            import lightgbm as lgb
            est = lgb.LGBMClassifier(n_jobs=n_jobs, random_state=seed)
            pred = cross_val_predict(
                est, train_X, train_y, cv=cv, method='predict_proba')[:, 1]
            score = roc_auc_score(train_y, pred)

    end = time.time()
    duration = end - start

    logger.info('CV AUC: %.6f' % score)
    res = pd.DataFrame(
        [[
            task, model, n_trials,
            args.preprocess_train_frac, args.preprocess_test_frac,
            model_train_frac, model_valid_frac,
            duration, score, commit,
        ]],
        columns=[
            'dataset', 'model', 'n_trials',
            'preprocess_train_frac', 'preprocess_test_frac',
            'model_train_frac', 'model_valid_frac',
            'duration[s]', 'CV AUC', 'commit',
        ])

    res.to_csv(result_path, index=False)


if __name__ == '__main__':
    main()
