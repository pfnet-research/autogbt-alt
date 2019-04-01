import pytest
import pandas as pd
from autogbt.preprocessor import (
    _extract_datetime_features,
    _extract_datetime_difference,
    Preprocessor,
)


def test_extract_datetime_features():
    df = pd.DataFrame([pd.to_datetime('2019-01-02')], columns=['dt'])
    _extract_datetime_features(df)
    assert (df.columns == [
        'dt',
        'dt_dayofweek',
        'dt_dayofyear',
        'dt_month',
        'dt_weekofyear',
        'dt_day',
        'dt_hour',
        'dt_minute',
        'dt_year',
    ]).all()


def test_extract_datetime_difference():
    df = pd.DataFrame([
        [pd.to_datetime('2019-01-02'), pd.to_datetime('2019-02-02')],
    ], columns=['dt1', 'dt2'])
    _extract_datetime_difference(df)
    assert (df['diff_dt1_dt2'] == -2678400000000000).all()


def test_init():
    Preprocessor()
    # train
    Preprocessor(train_frac=0.5)
    Preprocessor(train_size=200)
    with pytest.raises(AttributeError):
        Preprocessor(train_frac=0.5, train_size=200)
    # test
    Preprocessor(test_frac=0.5)
    Preprocessor(test_size=200)
    with pytest.raises(AttributeError):
        Preprocessor(test_frac=0.5, test_size=200)
