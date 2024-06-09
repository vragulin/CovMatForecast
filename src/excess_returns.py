""" Calculate excess returns on all stocks
	V. Ragulin - 8-Jun-2024
"""
import numpy as np
import pandas as pd
import pickle as pk
import config as cfg
import datetime as dt

from pathlib import Path


def main():

	# Load prices
	data_dir = Path(cfg.BASE_DIR) / cfg.DATA_DIR
	with open(data_dir / cfg.STOCK_RETURNS_FILE, "rb") as f:
		rets_dict = pk.load(f)

	# Load Fed Funds
	ff = pd.read_csv(data_dir / cfg.FED_FUNDS_HIST_FILE, index_col="date",
	                 parse_dates=True)

	rets = rets_dict['total'].copy()
	rets['ff'] = ff

	rets['days'] = np.nan
	rets.loc[rets.index[1]:, 'days'] = (rets.index[1:] - rets.index[:-1]).days
	rets.loc[rets.index[0], 'days'] = 1 if rets.index[0].isoweekday() <= 5 else 3

	rets['r_ff'] = rets.ff / 36500 * rets.days

	xrets = rets_dict['total'].copy() * 0
	for col in xrets.columns:
		xrets[col] = (1 + rets[col]) / (1 + rets.r_ff) - 1

	xrets.to_pickle(data_dir / cfg.STOCK_XRETS_FILE)
	print("Done")


if __name__ == "__main__":
	main()
