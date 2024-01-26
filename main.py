import math
import pickle
from collections import defaultdict
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def calculate_slq_k(list_regions: List[int],
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
    y_k_r, y_r, y_n_k, y_n = defaultdict(float), 1, defaultdict(float), 1
    # Each sector is each k industry
    for sector in y[classification_col].unique():
        for region in list_regions:
            try:
                y_k_r[sector] += y.loc[(y[region_col] == region)
                                       & (y[classification_col] == sector)][gdp_col].reset_index().loc[0, gdp_col]
            except KeyError:
                # Missing sector for this municipality
                pass
        y_n_k[sector] += y.loc[y[classification_col] == sector][gdp_col].reset_index().loc[0, gdp_col]
    y_r = sum(y_k_r.values())
    y_n = sum(y_n_k.values())
    for sector in y[classification_col].unique():
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


def calculate_regional_technical_matrix_from_rho(A, rho):
    reg_matrix = pd.DataFrame(columns=A.columns)
    for k in A.index:
        for l in A.columns:
            reg_matrix.loc[k, l] = A.loc[k, l] * rho.loc[k, l]
    return reg_matrix


def calculate_residual_matrix(A, regional):
    residual_matrix = pd.DataFrame(columns=A.columns)
    for k in A.index:
        for l in A.columns:
            residual_matrix.loc[k, l] = A.loc[k, l] - regional.loc[k, l]
    return residual_matrix


def putting_together_full_matrix(upper_left, upper_right, bottom_left, bottom_right,
                                 metro_name='BSB', rest='RestBR', col_interest='massa_salarial_sum'):
    cols = [f'{name}_{col}' for name in [metro_name, rest] for col in upper_left.columns]
    number_of_sectors = 12
    # Create final matrix
    result_matrix = pd.DataFrame(columns=cols, index=cols)
    result_matrix.iloc[:number_of_sectors, :number_of_sectors] = upper_left
    result_matrix.iloc[:number_of_sectors, number_of_sectors:] = upper_right
    result_matrix.iloc[number_of_sectors:, :number_of_sectors] = bottom_left
    result_matrix.iloc[number_of_sectors:, number_of_sectors:] = bottom_right
    result_matrix = result_matrix.astype(float)

    plt.figure(figsize=(12, 12))
    # Create a heatmap using Seaborn
    sns.heatmap(result_matrix, cmap="viridis", cbar=True)
    plt.xlabel('Industry of destination')
    plt.ylabel('Industry of origin')
    plt.title('$\\rho$ coefficient')
    plt.xticks(fontsize=12, rotation=90)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{metro_name}_{col_interest}.png')
    plt.show()
    return result_matrix


def main(metro_list=None, metro_name='BRASILIA', rest='RestBR', debug=False, col_interest='massa_salarial_sum'):
    """ Receives a list of municipalities codes as integer and return the technical matrix and final demand
        for that metro region, plus the rest of Brazil.
        """
    if not metro_list:
        acps = pd.read_csv('data/ACPs_MUN_CODES.csv', sep=';')
        acps['cod_mun'] = acps['cod_mun'].astype(str).str[:6].astype(int)
        metro_list = acps[acps['ACPs'] == metro_name]['cod_mun'].to_list()
    # A is the technical coefficient matrix. But it can also be final demand matrix
    A_kl = pd.read_csv('data/technical_matrix.csv').set_index('sector')
    # fd is the final demand vectors
    fd = pd.read_csv('data/final_demand.csv')
    # Read the list of all municipalities with code and sum of all salaries per sector
    massa_salarial = pd.read_csv('data/mun_isic12_2010.csv')
    # Derive the list of the rest of BRAZIL
    rest_list = [code for code in massa_salarial.codemun.to_list() if code not in metro_list]
    # DEBUG
    if debug:
        rest_list = rest_list[:100]
    # Calculate SLQ and lambda for both groups of municipalities
    slq_me, lbda_me = calculate_slq_k(metro_list, massa_salarial, gdp_col=col_interest)
    slq_re, lbda_re = calculate_slq_k(rest_list, massa_salarial, gdp_col=col_interest)
    # Calculate CILQ for both groups of municipalities
    cilq_me = calculate_cilq_kl(slq_me)
    cilq_re = calculate_cilq_kl(slq_re)
    # Calculate FLQ for both groups of municipalities
    flq_me = calculate_flq_kl(lbda_me, slq_me, cilq_me)
    flq_re = calculate_flq_kl(lbda_re, slq_re, cilq_re)
    # Calculate RHO for both groups of municipalities
    rho_me = calculate_rho(flq_me)
    rho_re = calculate_rho(flq_re)
    # Calculate rho final demand for both groups
    rho_me_final = np.diag(flq_me.values)
    rho_re_final = np.diag(flq_re.values)
    # Calculating the deriving matrices
    A_me = calculate_regional_technical_matrix_from_rho(A_kl, rho_me)
    A_re = calculate_regional_technical_matrix_from_rho(A_kl, rho_re)
    # Calculating residuals matrices
    A_re_me = calculate_residual_matrix(A_kl, A_me)
    A_me_re = calculate_residual_matrix(A_kl, A_re)
    # Putting it all together and plotting
    result_matrix = putting_together_full_matrix(A_me, A_me_re, A_re, A_re_me, metro_name, rest, col_interest)
    return result_matrix


if __name__ == '__main__':
    # metr = pd.read_csv('data/list_mun_to_matrix.csv')
    # metr_list = metr.codemun.to_list()
    # metr_name = 'BSB'
    alternative_col = 'qtde_vinc_ativos_sum'
    metr_name = 'SAO PAULO'
    res = main(metro_name=metr_name, debug=False, col_interest=alternative_col)

    with open(f'matrix_{metr_name}', 'wb') as handler:
        pickle.dump(res, handler)
