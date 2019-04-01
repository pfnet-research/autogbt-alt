import argparse
import seaborn as sns  # noqa
from matplotlib import pyplot as plt
from pathlib import Path
from dask import dataframe as dd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True)
    args = parser.parse_args()

    data_dir = Path(args.input)
    name = data_dir.parent.name
    path = str(data_dir/'*.csv')
    df = dd.read_csv(path).compute()
    df = df.groupby(['model', 'n_trials', 'model_train_frac']).agg({
        'CV AUC': ['mean', 'std'],
        'duration[s]': ['mean', 'std'],
    })
    df.columns = ['%s_%s' % (a, b) for a, b in df.columns]
    df = df.reset_index()
    print(df)

    # plot
    fracs = sorted(df['model_train_frac'].unique())
    plt.figure(figsize=(12, 5))
    for j, frac in enumerate(fracs):
        for i, n_trials in enumerate([1, 10, 20, 30]):
            idx = (df['model'] == 'auto') &\
                  (df['n_trials'] == n_trials) &\
                  (df['model_train_frac'] == frac)
            x = df.loc[idx, 'duration[s]_mean']
            y = df.loc[idx, 'CV AUC_mean']
            xerr = df.loc[idx, 'duration[s]_std']
            yerr = df.loc[idx, 'CV AUC_std']
            fmt = '%sC%d' % (['x', 'o', 's', 'D'][i], j)
            label = 'n_trials=%d, model_train_frac=%.2f' % (n_trials, frac)  # noqa
            plt.errorbar(x, y, xerr=xerr, yerr=yerr, fmt=fmt, label=label)

    plt.title('Parameter Comparison (dataset=%s)' % (name))
    plt.xlabel('Training Time[s]')
    plt.ylabel('CV AUC')
    plt.legend()
    plt.savefig(str(data_dir/'frac-and-n_trials.png'))


if __name__ == '__main__':
    main()
