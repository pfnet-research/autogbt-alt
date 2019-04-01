import argparse
import pandas as pd
from pathlib import Path
from dask import dataframe as dd
from tabulate import tabulate
import const


def _handle_nan(score):
    if score.find('nan') == 0:
        return 'OoM'
    return score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True)
    args = parser.parse_args()

    path = str(Path(args.input)/'*.csv')
    df = dd.read_csv(
        path,
        dtype={
            'n_trials': 'float64',
        },
    ).compute()
    df = df.groupby(['model', 'dataset']).agg({
        'duration[s]': ['mean', 'std'],
        'CV AUC': ['mean', 'std'],
    })
    dummy = pd.DataFrame(index=[
        ('xgb', 'avazu'),
        ('lgb', 'avazu'),
    ])
    df = df.append(dummy)
    df.columns = ['%s_%s' % (a, b) for a, b in df.columns]
    df = df.reset_index()
    for c in ['duration[s]', 'CV AUC']:
        mean = '%s_mean' % (c)
        std = '%s_std' % (c)
        df[std] = df[std].apply(lambda d: '±%.3f' % (d))
        df[mean] = df[mean].apply(lambda d: '%.3f' % (d))
        df[c] = (df[mean] + df[std]).apply(_handle_nan)
    df['dataset'] = df['dataset'].map(const.competitions)
    df['model'] = df['model'].map(const.models)
    df = df[['dataset', 'model', 'duration[s]', 'CV AUC']]
    df = df.sort_values(['dataset', 'model'])
    df = df.reset_index(drop=True)
    df['model'] = df['model'].apply(lambda d: d[1])

    for dset, grp in df.groupby('dataset'):
        grp.pop('dataset')
        md = tabulate(grp.values, grp.columns, tablefmt='pipe', floatfmt='.3f')
        print('#### %s\n' % (dset))
        print(md + '\n')


if __name__ == '__main__':
    main()
