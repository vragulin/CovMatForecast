STOCKS = ['MSFT', 'AAPL', 'TSLA', 'BRK-B', 'LLY', 'JPM', 'XOM', 'UNH', 'PG', 'COST']
SECTORS = ['Tech', 'Tech', 'Cars', 'Fin', 'Pharma', 'Fin', 'Oil', 'Health', 'Consumer', 'Retail']
INDEX = ['VTI']
ALL_TICKERS = STOCKS + INDEX

# Simulation Range
END_DATE = "2024-06-07"
HIST_YEARS = 7
SIM_YEARS = 5

# Directory Locations
BASE_DIR = r"C:\Users\vragu\OneDrive\Desktop\Proj\CovMatForecast"
DATA_DIR = "data"
STOCK_PRICES_FILE = "prices_10.pk"
STOCK_RETURNS_FILE = "returns_10.pk"
FED_FUNDS_HIST_FILE = "fed_funds.csv"
STOCK_XRETS_FILE = "xrets_10.pk"
STOCK_EXP_XRETS = "exp_xrets_10.csv"
PORT_RET_FILE = "port_rets_{method}.csv"

# Expected Return parameters
EQ_RISK_PREM = 0.03

# Cov matrix methodology
ANN_FACTOR = 252
COV_METHOD = 'garch'  # ['robust', 'hist', 'garch']
COV_WINDOW = ANN_FACTOR * 2
VOL_TGT = 0.10

# Cov_matrix - robust optimization
CORR_SAME_SECTOR = 0.7
CORR_DIFF_SECTOR = 0.4
SHRINK_COEF = 0.5

# GARCH parameters
GARCH_ST_WIN = 21
GARCH_WEIGHTS = {'ON': 0.1, 'ST': 0.4, 'LT': 0.5}
