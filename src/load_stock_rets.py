"""
Load historical stock data
V. Ragulin : 7-Jun-24
"""
import datetime as dt
from dateutil.relativedelta import relativedelta
import pandas as pd
import yfinance as yf
import config as cfg
import pickle

from pathlib import Path
from typing import Optional, Tuple, List

# Constants
UPDATE_PRICES = True
PRICES_FILE = cfg.STOCK_PRICES_FILE
RETURNS_FILE = cfg.STOCK_RETURNS_FILE


def fetch_ticker_prices(tickers: List[str], start: Optional[dt.datetime] = None,
                        end: Optional[dt.datetime] = None
                        ) -> Tuple[pd.DataFrame, List[str]]:
	""" Get historical ticker prices from Yahoo
	:param tickers: list of exchange tickers (e.g. AAPL)
	:param start: start date
	:param end: end date
	:return: tuple - dataframe with prices ,list of missing tickers
	"""
	if end is None:
		end = dt.datetime.now()

	if start is None:
		start = end - dt.timedelta(days=7)

	df_data = yf.download(tickers, start=start, end=end, progress=False, group_by="ticker")

	missing = []
	for ticker in tickers:
		if pd.isnull(df_data[ticker].values).all():
			missing.append(ticker)

	# Drop missing tickers from the dataframe
	df_clean = df_data.dropna(axis='columns', how='all')
	return df_clean, missing


def calc_returns(price_info: pd.DataFrame) -> dict:
	"""
	Calculate total and price returns from price
	:param price_info: - dataframe of prices
	:return: dict with 2 dataframes of price and total returns
	"""

	# Extract dataframes of closes and adjusted closes
	idx = pd.IndexSlice
	rets = {}
	for code, field in zip(['price', 'total'], ['Close', 'Adj Close']):
		px = price_info.loc[:, idx[:, field]].droplevel(level=1, axis=1)
		rets[code] = px.pct_change().dropna(how='all').sort_index(axis=1)

	return rets


def get_dates() -> Tuple[dt.datetime, dt.datetime]:
	end_date = dt.datetime.strptime(cfg.END_DATE, "%Y-%m-%d")
	start_date = end_date - relativedelta(years=cfg.HIST_YEARS)
	return start_date, end_date


def fetch_prices() -> pd.DataFrame:
	start, end = get_dates()
	data_file = Path(cfg.BASE_DIR) / cfg.DATA_DIR / PRICES_FILE
	if UPDATE_PRICES:
		prices, missing = fetch_ticker_prices(cfg.ALL_TICKERS, start, end)
		prices.to_pickle(str(data_file))
		if len(missing) > 0:
			print("Missing tickers: ", missing)
	else:
		prices = pd.read_pickle(data_file)

	return prices


def calc_save_returns(prices):
	rets = calc_returns(prices)
	data_file = Path(cfg.BASE_DIR) / cfg.DATA_DIR / RETURNS_FILE
	pickle.dump(rets, open(data_file, "wb"))
	return rets


def main():
	prices = fetch_prices()
	print(prices)
	rets = calc_save_returns(prices)
	print(rets)


if __name__ == "__main__":
	main()
