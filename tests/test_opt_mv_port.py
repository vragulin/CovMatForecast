""" Pytest - optimal mean-variance portfolio
	V. Ragulin - 8-Jun-2024
"""
import numpy as np
import pytest as pt

from cov_sim import opt_mv_port


def test_opv_mv_port1():

	vols = np.array([0.2, 0.1])
	rets = np.array([0.025, 0.01])
	cor_mat = np.array([
		[1, 0.7],
		[0.7, 1]
	])
	vol_tgt = 0.1
	cov_mat = (vols[:, None] @ vols[None, :]) * cor_mat

	w = opt_mv_port(cov_mat, rets, vol_tgt)
	expected = np.array([0.427204601, 0.194183909])
	np.testing.assert_allclose(w, expected)
