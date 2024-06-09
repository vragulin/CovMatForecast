""" Generate historical cov matrices
	V. Ragulin - 8-Jun-2024
"""
import numpy as np
import pandas as pd
import config as cfg
import pickle as pk
import datetime as dt
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Optional, Tuple, List
from dateutil.relativedelta import relativedelta
from abc import ABC


class CovMatSim:
	def __init__(self, method: str = 'hist'):
		data_dir = Path(cfg.BASE_DIR) / cfg.DATA_DIR
		with open(data_dir / cfg.STOCK_XRETS_FILE, "rb") as f:
			xrets = pk.load(f)

		rets = xrets[cfg.STOCKS]
		rets = rets.reindex(sorted(rets.columns), axis=1)  # Sort columns

		# Data array dimensions; store covmats in an np array
		nstocks = len(cfg.STOCKS)
		start, end = get_sim_dates()
		sim_index = xrets.index[(xrets.index >= start) & (xrets.index <= end)]
		ndates = len(sim_index)

		# Load expected returns
		exp_rets = pd.read_csv(data_dir / cfg.STOCK_EXP_XRETS, index_col='Ticker')

		# Pack data into instance variables
		self.rets = rets
		self.rets_arr = rets.values
		self.exp_rets = exp_rets
		self.nstocks = nstocks
		self.ndates = ndates
		self.start = start
		self.end = end
		self.sim_index = sim_index
		self.covmats = np.zeros((ndates, nstocks, nstocks))
		self.ports = np.zeros((ndates, nstocks))
		self.r_port = np.zeros(ndates)
		self.method = method
		self.context = dict()
		self.init_context()

	def calc_hist_covmats(self):
		""" Calculate covariance matrices for each simulation date
		:param method: how to calculate the cov_matrix
		:param method: method used for covmat calculation
		"""
		for i in range(self.ndates - 1):
			# print(f"{i}, {self.sim_index[i]}")
			self.covmats[i, :, :] = self.calc_one_day_covmat(i)

	def init_context(self):
		if self.method == "hist":
			pass
		elif self.method in ["robust", "garch"]:

			self.context['lt_corr_mat'] = self.calc_lt_corr_mat(same_sector=0.5, diff_sector=0.4)
		else:
			raise NotImplementedError(f"Method = {self.method}")

	def calc_lt_corr_mat(self, same_sector: float, diff_sector: float) -> np.array:
		cormat = np.zeros((self.nstocks, self.nstocks))
		sector_ref = {k: v for k, v in zip(cfg.STOCKS, cfg.SECTORS)}
		for i, s_i in enumerate(self.rets.columns):
			for j, s_j in enumerate(self.rets.columns):
				if i == j:
					cormat[i, j] = 1
				elif sector_ref[s_i] == sector_ref[s_j]:
					cormat[i, j] = cfg.CORR_SAME_SECTOR
				else:
					cormat[i, j] = cfg.CORR_DIFF_SECTOR
		return cormat

	def calc_one_day_covmat(self, i: int) -> np.array:
		""" Calculate cov matrix for a single day
		:param i: index of the date in the sim_index array
		:param method: how to calculate the cov_matrix
		:return: annualized cov matrix as an nstocks x nstocks array
		"""

		if self.method == "hist":
			return self.calc_one_day_covmat_hist(i)
		elif self.method == "robust":
			return self.calc_one_day_covmat_robust(i)
		elif self.method == "garch":
			return self.calc_one_day_covmat_garch(i)
		else:
			raise NotImplementedError(f"Method = {self.method}")

	def calc_one_day_covmat_hist(self, i: int):
		""" Wrapper that takes the window from the config file"""
		return self._calc_one_day_covmat_hist(i, cfg.COV_WINDOW)

	def _calc_one_day_covmat_hist(self, i: int, win: int) -> np.array:
		""" Calculate cov matrix for a single day using historical method
		:param i: index of the date in the sim_index array
		:param win: lookback window
		:return: annualized cov matrix as an nstocks x nstocks array
		"""

		# Get indices of the estimation window in the rets array
		end_date = self.sim_index[i]
		end_idx = self.rets.index.get_loc(end_date)
		start_idx = max(end_idx - win, 0)
		covmat = np.cov(self.rets_arr[start_idx:end_idx, :], rowvar=False) * cfg.ANN_FACTOR
		return covmat

	def calc_one_day_covmat_robust(self, i: int) -> np.array:
		cov_hist = self.calc_one_day_covmat_hist(i)
		vols_lt = np.sqrt(np.diagonal(cov_hist))
		cov_lt = (vols_lt[:, None] @ vols_lt[None, :]) * self.context['lt_corr_mat']
		cov_robust = cov_lt * cfg.SHRINK_COEF + cov_hist * (1 - cfg.SHRINK_COEF)
		return cov_robust

	def hist_vols_one_day(self, i, win):
		""" Calculate historical vols for a day
		:param i: index of the date in the sim_index array
		:param win: lookback window
		:return: vector of annualized vols
		"""
		return np.sqrt(np.diagonal(self._calc_one_day_covmat_hist(i, win)))

	def calc_one_day_covmat_garch(self, i: int) -> np.array:
		""" Covariance matrix with diagonal terms driven by a GARCH model
		"""
		# Calculate the correlation matrix and the LT vol component
		cov_lt = self.calc_one_day_covmat_robust(i)
		vol_lt = np.sqrt(np.diagonal(cov_lt))
		cormat = cov_lt / (vol_lt[:, None] @ vol_lt[None, :])

		# Calc short-term vol and yesterday's variance
		vol_st = self.hist_vols_one_day(i, cfg.GARCH_ST_WIN)

		end_date = self.sim_index[i]
		end_idx = self.rets.index.get_loc(end_date)
		vol_last = self.rets_arr[end_idx, :] * np.sqrt(cfg.ANN_FACTOR)

		# Take a weighted average for GARCH
		w = cfg.GARCH_WEIGHTS
		var_garch = w['ON'] * vol_last ** 2 + w['ST'] * vol_st ** 2 + w['LT'] * vol_lt ** 2
		vol_garch = np.sqrt(var_garch)
		cov_garch = (vol_garch[:, None] @ vol_garch[None, :]) * cormat
		return cov_garch

	def calc_hist_ports(self):
		""" Calculae historical portfolios"""
		rets = self.exp_rets.exp_xret
		for i in range(0, self.ndates - 1):
			self.ports[i, :] = opt_mv_port(self.covmats[i, :, :], rets, cfg.VOL_TGT)

	def calc_hist_port_rets(self):
		""" Historical port returns """
		sim_rets = self.rets.loc[self.sim_index, :]
		sim_rets_arr = sim_rets.values
		self.r_port[1:] = np.sum(sim_rets_arr[1:, :] * self.ports[:-1, :], axis=1)

	def an_sim_resutls(self):
		""" Historical port returns """
		vol = self.r_port[1:].std() * np.sqrt(cfg.ANN_FACTOR)
		print(f"Out of sample vol = {vol * 100:.2f}%.")

		df_r = pd.DataFrame(self.r_port[:, None], index=self.sim_index,
		                    columns=['port_rets'])
		df_r['hvol_3m'] = df_r.rolling(63).std() * np.sqrt(cfg.ANN_FACTOR)
		df_r.hvol_3m.plot(title="Historical Rolling 3mo portfolio vol")
		plt.show()

	def plot_covmat_series(self, s_list: List[int]):
		for s in s_list:
			plt.plot(self.sim_index[:-1], np.sqrt(self.covmats[:-1, s, s]))
		plt.legend(s_list)
		plt.title("Historical Vol Forecasts")
		plt.show()

	def plot_w_series(self, s_list: List[int]):
		for s in s_list:
			plt.plot(self.sim_index[:-1], self.ports[:-1, s])
		plt.legend(s_list)
		plt.title("Historical Portfolio Weights")
		plt.show()

	def save_port_returns(self):
		""" Save portfolio returns into a csv file
		"""
		ret_file = cfg.PORT_RET_FILE.format(method=self.method)
		df = pd.DataFrame(self.r_port, index=self.sim_index, columns=[self.method])
		df.to_csv(Path(cfg.BASE_DIR) / cfg.DATA_DIR / ret_file)


def get_sim_dates() -> Tuple[dt.datetime, dt.datetime]:
	end_date = dt.datetime.strptime(cfg.END_DATE, "%Y-%m-%d")
	start_date = end_date - relativedelta(years=cfg.SIM_YEARS)
	return start_date, end_date


def opt_mv_port(covmat: np.array, rets: np.array, vol_tgt: float) -> np.array:
	""" Calculate an unconstrainted """
	intermed = np.linalg.inv(covmat) @ rets
	w = intermed / np.sqrt(rets.T @ intermed) * vol_tgt;
	return w


def main():
	sim = CovMatSim(method=cfg.COV_METHOD)

	# Calculate historical covariance matrices
	sim.calc_hist_covmats()
	sim.plot_covmat_series([0, 1, 9])

	# Caclulate portfolios
	sim.calc_hist_ports()
	sim.plot_w_series(list(range(10)))

	# Calculate and analyse portfolio returns
	sim.calc_hist_port_rets()
	sim.an_sim_resutls()
	sim.save_port_returns()
	print('Done')


if __name__ == "__main__":
	main()
