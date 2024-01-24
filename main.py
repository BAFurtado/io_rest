import pickle
from collections import defaultdict
from typing import List
import math
import pandas as pd


def calculate_slq_k(list_regions: List[str],
                    y: pd.DataFrame,
                    classification_col: str = 'isic_r4',
                    region_col: str = 'codemun',
                    gdp_col: str = 'massa_salarial_sum') \
        -> pd.DataFrame:
    """ Regional GDP of industry k in a generic region r divided by the sum for industry k of all GDP
        divided by GDP of industry k nationally divided by all GDP
        SLQ_k = y_k^r / y^r /
                y_n^k / y^n
    """
    output = pd.DataFrame(columns=['SLQ'])
    lambda_ = defaultdict(float)
    y_k_r, y_r, y_n_k, y_n = defaultdict(int), 1, defaultdict(int), 1
    # Each sector is eack k industry
    for sector in y[classification_col].unique():
        for region in list_regions:
            try:
                y_k_r[sector] += y.loc[(y[region_col] == region)
                                       & (y[classification_col] == sector)][gdp_col].reset_index().loc[0, gdp_col]
            except KeyError:
                # Missing sector for this municipality
                pass
        y_r = sum(y_k_r.values()) or 1
        y_n_k[sector] += y.loc[massa[classification_col] == sector][gdp_col].reset_index().loc[0, gdp_col]
        y_n = sum(y_n_k.values())
        SLQ_r = (y_k_r[sector] / y_r) / (y_n_k[sector] / y_n)
        output.loc[sector, 'SLQ'] = SLQ_r
        print(f'SLQ_{sector} = {SLQ_r}')
        lambda_[sector] = calculate_lambda(y_r, y_n)
    return output, lambda_


def calculate_lambda(y_r, y_n, delta=.15):
    # To what extent region size affects regional coefficients
    return (math.log2(1 + y_r / y_n)) ** delta


def calculate_cilq_kl(slq: pd.DataFrame):
    """ Given the sectors of each region, calculate the SLQ by dividing k by l sectors
    """
    cilq_matrix = pd.DataFrame()
    for k in slq.index:
        for l in slq.index:
            cilq_matrix.loc[k, l] = slq.loc[k, 'SLQ'] / slq.loc[l, 'SLQ']
    return cilq_matrix


def calculate_flq_kl():
    pass


def calculate_rho():
    pass


if __name__ == '__main__':
    metro = pd.read_csv('data/list_mun_to_matrix.csv')
    massa = pd.read_csv('data/mun_isis12_2010.csv')

    metro_list = metro.codemun.to_list()
    rest_list = [code for code in massa.codemun.to_list() if code not in metro_list]

    slq_me, lbda_me = calculate_slq_k(metro_list, massa)
    slq_re, lbda_re = calculate_slq_k(rest_list, massa)

    cilq_me = calculate_cilq_kl(slq_me)
    cilq_re = calculate_cilq_kl(slq_re)

    # with open('slq_me_slq_re', 'wb') as handler:
    #     pickle.dump([slq_me, slq_re], handler)

    with open('slq_re_slq_me', 'rb') as handler:
        slq_re, slq_me = pickle.load(handler)
