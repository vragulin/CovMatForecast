""" Plot Historical Volatilties of Diffeernt Portfolios
"""
import numpy as np
import pandas as pd
import config as cfg
import datetime as dt
import matplotlib.pyplot as plt

from pathlib import Path
from typing import List

# Global parameters
MODELS = ["hist", "robust", "garch"]
VOL_WIN = 63


def load_all_rets(models: List[str]) -> pd.DataFrame:
	df = None
	for m in MODELS:
		ret_file = cfg.PORT_RET_FILE.format(method=m)
		df_method = pd.read_csv(Path(cfg.BASE_DIR) / cfg.DATA_DIR / ret_file,
		                        index_col='Date', parse_dates=True)
		if df is None:
			df = df_method
		else:
			df[m] = df_method[m]
	return df


def do_plot(rets: pd.DataFrame):
	vols = rets.rolling(VOL_WIN).std() * np.sqrt(cfg.ANN_FACTOR) * 100
	ax = vols.plot(title="Historical Rolling 3mo Vol of the MV Optimal 10% Portfolio")
	plt.xticks(rotation=25)
	plt.ylabel("Rolling 3mo Vol (%)")
	plt.grid()
	plt.show()


def main():
	df = load_all_rets(MODELS)
	do_plot(df)


if __name__ == "__main__":
	main()
