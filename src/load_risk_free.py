""" Load risk-free rates for the required period
	V. Ragulin  - 8-Jun-2024
"""
import pandas as pd
import fredpy as fp
import config as cfg

from pathlib import Path


def load_FedFunds() -> pd.DataFrame:
	fp.api_key = fp.load_api_key('FRED_API_Key.txt')
	return fp.series('DFF').data


def main():
	ff = load_FedFunds()
	ff.to_csv(str(Path(cfg.BASE_DIR) / cfg.DATA_DIR / cfg.FED_FUNDS_HIST_FILE))
	print(ff)


if __name__ == "__main__":
	main()