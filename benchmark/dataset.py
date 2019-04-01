from pathlib import Path
import pandas as pd


def get(name):
    data_dir = Path(__file__).parent/'dataset'
    if name == 'bank':
        X = pd.read_csv(data_dir/'bank/bank-full.csv', delimiter=';')
        y = (X.pop('y') == 'yes').astype('i')
        return X, y, pd.DataFrame([], columns=X.columns)
    if name == 'avazu':
        X = pd.read_csv(data_dir/'avazu/train')
        y = X.pop('click')
        test_X = pd.read_csv(data_dir/'avazu/test')
        return X, y, test_X
    if name == 'amazon':
        X = pd.read_csv(data_dir/'amazon/train.csv')
        y = X.pop('ACTION')
        test_X = pd.read_csv(data_dir/'amazon/test.csv')
        del test_X['id']
        return X, y, test_X
    if name == 'airline':
        X = pd.read_csv(data_dir/'airline/AirlinesCodrnaAdult.csv')
        y = X.pop('Delay')
        test_X = pd.DataFrame([], columns=X.columns)
        return X, y, test_X
    if name == 'redhat':
        X = pd.read_csv(data_dir/'redhat/act_train.csv')
        y = X.pop('outcome')
        test_X = pd.read_csv(data_dir/'redhat/act_test.csv')

        X['date'] = pd.to_datetime(X['date'])
        test_X['date'] = pd.to_datetime(test_X['date'])
        return X, y, test_X
    raise ValueError('invalid dataset %s' % (name))
