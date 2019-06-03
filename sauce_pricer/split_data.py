"""
Split the data `PSPOption_RAW_Data.pkl` into 3 data sets:

    - `./data/train_data.pkl` contains *complete data* from January 2017 to August 2017; 
    - `./data/valid_data.pkl` contains *complete data* from September and October 2017; 
    - `./data/test_data.pkl` contains *complete data* from November and December 2017.

*complete data* means that only lines without missing values are kept.
Data from Jan to Au with missing values are saved under `./data/train_data_missing.pkl`
"""
from typing import List
import numpy as np

from sauce_pricer.utils import file_utils
from sauce_pricer.globals import FULL_DATA_PATH


def cli(sys_argv: List[str]):
    data = file_utils.pickle2dataframe(FULL_DATA_PATH)
    data = data[data.PX_LAST > 0.]
    data['LOG_PX_LAST'] = np.log(data.PX_LAST)

    valuation_dates = data['ValuationDate']
    valuation_years = [dates.to_pydatetime().year for dates in valuation_dates]
    valuation_months = [dates.to_pydatetime().month for dates in valuation_dates]

    years = set(valuation_years)
    months = set(valuation_months)

    data['ValuationYear'] = valuation_years
    data['ValuationMonth'] = valuation_months

    data = data[data.ValuationYear == 2017]

    train_months = list(range(1, 9))
    valid_months = [9, 10]
    test_months = [11, 12]

    all_index = data.index

    # remove line with missing values
    data_complete = data.copy().dropna()

    # split data
    train_data = data_complete[data_complete.ValuationMonth.isin(train_months)]
    valid_data = data_complete[data_complete.ValuationMonth.isin(valid_months)]
    test_data = data_complete[data_complete.ValuationMonth.isin(test_months)]

    # data_missing contains only lines with missing data
    missing_index = [i for i in all_index if i not in data_complete.index]
    data_missing = data.loc[missing_index]
    train_data_missing = data_missing[data_missing.ValuationMonth.isin(train_months)]

    # save the data to pickles
    file_utils.dataframe2pickle(train_data, constants.TRAIN_PATH)
    file_utils.dataframe2pickle(valid_data, constants.VALID_PATH)
    file_utils.dataframe2pickle(test_data, constants.TEST_PATH)
    file_utils.dataframe2pickle(train_data_missing, constants.TRAIN_PATH_MISSING)
