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
        -> (pd.DataFrame, pd.DataFrame):
    """ Regional GDP of industry k in a generic region r divided by the sum for industry k of all GDP
        divided by GDP of industry k nationally divided by all GDP
        SLQ_k = (y_k^r / y^r) /
                (y_n^k / y^n)
        k is an industry/sector
        l is another sector within the same region
        "CILQ_kl > 1 implies that GDP of regional sector k is larger than GDP of regional sector l than it
        is at the national levels. Thus, the demand of l can be fully met by the supply of k.
        On the other hand, if CILQkl < 1, parts of l’s demand need to be imported."
    """
    output = pd.DataFrame(columns=['SLQ'])
    lambda_ = defaultdict(float)
    y_k_r, y_r, y_n_k, y_n = defaultdict(int), 1, defaultdict(int), 1
    # Each sector is each k industry
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
    lambda_ = pd.DataFrame.from_dict(lambda_, orient='index', columns=['lambda'])
    return output, lambda_


def calculate_lambda(y_r, y_n, delta=.15):
    # To what extent region size affects regional coefficients
    # delta [0, 1)
    return (math.log2(1 + y_r / y_n)) ** delta


def calculate_cilq_kl(slq: pd.DataFrame):
    """ Given the sectors of each region, calculate the SLQ by dividing k by l sectors
    """
    cilq_matrix = pd.DataFrame()
    for k in slq.index:
        for l in slq.index:
            cilq_matrix.loc[k, l] = slq.loc[k, 'SLQ'] / slq.loc[l, 'SLQ']
    return cilq_matrix


def calculate_flq_kl(lambda_, slq, cilq):
    """ If k == l,  lambda * CILQ_kl
        else:       lambda * SLQ_k
        """
    flq = pd.DataFrame(columns=cilq.columns)
    for k in cilq.index:
        for l in cilq.columns:
            if k == l:
                flq.loc[k, l] = lambda_.loc[k, 'lambda'] * slq.loc[k, 'SLQ']
            else:
                flq.loc[k, l] = lambda_.loc[k, 'lambda'] * cilq.loc[k, l]
    return flq


def calculate_rho(flq):
    for k in flq.index:
        for l in flq.columns:
            if flq.loc[k, l] >= 1:
                flq.loc[k, l] = 1
    return flq


def main():
    pass


if __name__ == '__main__':
    metro = pd.read_csv('data/list_mun_to_matrix.csv')
    massa = pd.read_csv('data/mun_isis12_2010.csv')

    metro_list = metro.codemun.to_list()
    rest_list = [code for code in massa.codemun.to_list() if code not in metro_list]

    # DEBUG. Restrict rest_list to a small number to implement things faster
    # In this example, BRASÍLIA is the METRO and the REST OF BR is the rest
    rest_list = rest_list[:1000]

    slq_me, lbda_me = calculate_slq_k(metro_list, massa)
    slq_re, lbda_re = calculate_slq_k(rest_list, massa)

    cilq_me = calculate_cilq_kl(slq_me)
    cilq_re = calculate_cilq_kl(slq_re)

    flq_me = calculate_flq_kl(lbda_me, slq_me, cilq_me)
    flq_re = calculate_flq_kl(lbda_re, slq_re, cilq_re)

    flq_me = calculate_rho(flq_me)
    flq_re = calculate_rho(flq_re)

    # Will need to enter existing technical matrix to derive the proportions... (later)

    # with open('slq_me_slq_re', 'wb') as handler:
    #     pickle.dump([slq_me, slq_re], handler)

    # with open('slq_re_slq_me', 'rb') as handler:
    #     slq_re, slq_me = pickle.load(handler)
