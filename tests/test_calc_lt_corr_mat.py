""" Test the long-term correlation matrix """
import numpy as np
import pandas as pd
import pytest as pt

from cov_sim import CovMatSim


def test_calc_lt_corr_mat1():
	sim = CovMatSim()
	mat = sim.calc_lt_corr_mat(same_sector=0.7, diff_sector=0.4)
	for i in range(sim.nstocks):
		assert mat[i, i] == 1
	df = pd.DataFrame(mat, sim.rets.columns, sim.rets.columns)
	assert df.loc['AAPL', 'MSFT'] == 0.7  # , "Error corr(AAPL,MSFT)"
	assert df.loc['AAPL', 'JPM'] == 0.4  # , "Error corr(AAPL,JPM)"
