"""  Calculate betas and expected returns for all the stocks using CAPM
	V. Ragulin - 8-Jun-2024
"""

import pandas as pd
import pickle as pk
import config as cfg
import numpy as np

from typing import Tuple
from pathlib import Path
from sklearn.linear_model import LinearRegression

# Constants
MIN_BETA = 0.4
MAX_BETA = 2.0


def est_betas(stks: pd.DataFrame, bmk: pd.Series, adj_method: str = "raw") -> pd.Series:
	""" Caclculate betas of stocks.
	:param stks: stocks excess returns ( dates x stocks )
	:param bmk:  benchmark excess returns (assume same date index)
	:param adj_method: adjustment method ('raw' - no adj, 'bloomberg' - 33% squeeze towards 1)
	:return: betas for all stocks
	"""

	assert np.array_equal(stks.index.values, bmk.index.values), "Index mistmatch"

	betas = pd.DataFrame(np.nan, index=stks.columns, columns=['raw_beta', 'adj_beta', 'beta'])
	linear_regressor = LinearRegression()

	for ticker in stks.columns:
		x = bmk.values.reshape(-1, 1)
		y = stks[ticker].values.reshape(-1, 1)
		reg = linear_regressor.fit(x,y)
		betas.loc[ticker, 'raw_beta'] = reg.coef_[0]

	# Perform adjustments
	if adj_method == 'raw':
		betas['adj_beta'] = betas['beta'] = betas['raw_beta']
	elif adj_method == 'bloomberg':
		betas['adj_beta'] = (betas['raw_beta'] * 2 + 1) / 3
		betas['beta'] = np.maximum(MIN_BETA, np.minimum(MAX_BETA, betas['adj_beta']))
	else:
		raise ValueError(f'Invalid beta adjustment method: {adj_method}')

	return betas


def calc_betas() -> pd.Series:
	""" Caclulate betas """
	# Load prices
	data_dir = Path(cfg.BASE_DIR) / cfg.DATA_DIR
	with open(data_dir / cfg.STOCK_XRETS_FILE, "rb") as f:
		xrets = pk.load(f)

	r_stks = xrets[cfg.STOCKS]
	r_idx = xrets[cfg.INDEX[0]]

	return est_betas(r_stks, r_idx, adj_method="bloomberg")


def main():
	stk_rets = calc_betas()
	stk_rets['exp_xret'] = stk_rets.beta * cfg.EQ_RISK_PREM
	print(stk_rets)
	stk_rets.to_csv(Path(cfg.BASE_DIR) / cfg.DATA_DIR / cfg.STOCK_EXP_XRETS)


if __name__ == "__main__":
	main()
