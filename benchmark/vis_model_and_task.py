import argparse
import seaborn as sns  # noqa
from matplotlib import pyplot as plt
from pathlib import Path
from dask import dataframe as dd
import const


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True)
    args = parser.parse_args()

    data_dir = Path(args.input)
    path = str(data_dir/'*.csv')
    df = dd.read_csv(path).compute()
    df['model'] = df['model'].map(const.models)
    df = df.sort_values(['dataset', 'model']).reset_index(drop=True)
    df = df.groupby(['model', 'dataset']).agg({
        'CV AUC': ['mean', 'std'],
        'duration[s]': ['mean', 'std'],
    })
    df.columns = ['%s_%s' % (a, b) for a, b in df.columns]
    df = df.reset_index()
    df['model'] = df['model'].apply(lambda d: d[1])
    print(df)

    # plot
    plt.figure(figsize=(8, 6))
    for i, (_, model) in enumerate(const.models.values()):
        for j, dset in enumerate(['airline', 'amazon', 'bank']):
            idx = (df['model'] == model) &\
                  (df['dataset'] == dset)
            x = df.loc[idx, 'duration[s]_mean']
            y = df.loc[idx, 'CV AUC_mean']
            xerr = df.loc[idx, 'duration[s]_std']
            yerr = df.loc[idx, 'CV AUC_std']
            fmt = '%sC%d' % (['o', 's', 'D', '^'][j], i)
            label = 'model=%s, dataset=%s' % (model, dset)  # noqa
            plt.errorbar(x, y, xerr=xerr, yerr=yerr, fmt=fmt, label=label)

    plt.title('Model Comparison')
    plt.xlabel('Training Time[s]')
    plt.ylabel('CV AUC')
    plt.legend(loc='lower right')
    plt.savefig(data_dir/'model_and_task.png')


if __name__ == '__main__':
    main()
