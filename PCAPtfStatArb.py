# Data-related Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

# General Imports
from datetime import datetime, timedelta
import os
from pytz import timezone
from pandas.tseries.offsets import BDay
import warnings
warnings.simplefilter(action = "ignore", category = FutureWarning)

# Statistical methods and models
from sklearn.decomposition import PCA
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm

# Display output
from prettytable import PrettyTable
import json
import tqdm


SPY_SECTOR_ETF = ["XLC", "XLY", "XLP", "XLE", "XLF", "XLV", "XLI", "XLB", "XLRE", "XLK", "XLU"]

class PCAPtfStatArb:
    
    benchmark_sym = "SPY"

    def __init__(self, tickers, start_date, end_date, initial_cash = 10e5):
        self.tickers = tickers
        tz = timezone('EST')
        self.start_date = start_date.replace(tzinfo = tz)
        self.end_date = end_date.replace(tzinfo = tz)
        self.initial_cash = initial_cash
        self.get_market_data()
    
    def get_market_data(self):
        folder = 'ProjectData'
        if not os.path.exists(folder):
            os.mkdir(folder)
        
        holder = {}
        data_tickers = self.tickers
        if self.benchmark_sym not in self.tickers:
            data_tickers += [self.benchmark_sym]
      
        yesterday = datetime.now() - timedelta(days = 1)
        max_range = yesterday - timedelta(days = 10 * 365)

        for ticker in data_tickers:
            file_name = f'{folder}/{ticker}.csv'
            df = yf.Ticker(ticker).history(start = max_range.strftime("%Y-%m-%d"), end = yesterday.strftime("%Y-%m-%d"))
            df.to_csv(file_name)
            df = pd.read_csv(file_name, index_col=0)
            df.index = pd.to_datetime(df.index, utc=True)
            df.index = df.index.tz_convert('US/Eastern')
            holder[ticker] = df
        
        self.benchmark_data = holder.pop(self.benchmark_sym)
        self.benchmark_returns = pd.DataFrame(self.benchmark_data.Close.pct_change().dropna(), columns = ['returns'])
        self.close_df, self.open_df, self.return_df, self.std_returns_df = self.generalize_data(holder)
        self.calculate_beta()

        
    def generalize_data(self, holder):
        # Generate dfs with close prices, returns, and standardized returns
        close_df = pd.concat(list(map(lambda x: x.Close, holder.values())), axis = 1)
        close_df.columns = list(holder.keys())
        open_df = pd.concat(list(map(lambda x: x.Open, holder.values())), axis = 1)
        open_df.columns = list(holder.keys())
        return_df = close_df.pct_change(axis = 0).dropna()
        std_returns_df = (return_df - return_df.rolling(252, axis = 0).mean())/ return_df.rolling(252, axis = 0).std()
        std_returns_df.dropna(inplace = True)
        return close_df, open_df, return_df, std_returns_df
    
    def calculate_beta(self):
        self.beta_df = pd.DataFrame()     
        for stock in self.return_df.columns:
            self.beta_df[stock] = self.return_df[stock].rolling(22).cov(self.benchmark_returns) / self.return_df[stock].rolling(22).var()
        
        # Beta calculation based on the previous close when opening the position
        self.beta_df.shift(1)

    def get_residuals(self, data, N = 1):
        pca = PCA(n_components = N)
        pca.fit(data)
        index = pca.components_[0]
        mm = [sm.OLS(s, index).fit() for s in data.values]
        res = list(map(lambda x: x.resid.T, mm))
        return res

    def z_score(self, data):
        scaler = StandardScaler()
        return scaler.fit_transform(data)
    
    def simulation(self, max_position, min_data_points, beta_hedge = False):
        '''
        Simulation function
    
        Assume trades will be opened during daily open and closed during daily close
        '''
        entry = {}
        pnls = []
        betas = []
        dates = []
        orders = {}
        slippage = 0.00025
        commission_fee = 0.005
        beta_limit = 0
    
    
        if max_position >= self.close_df.shape[1] / 2:
            print("There's too many positions within the portfolio")
            return
    
        cash_value = self.initial_cash
        for i in range(self.close_df.shape[0] + 1):
            if i < min_data_points:
                continue
    
            if i != self.close_df.shape[0] and self.close_df.index[i] < start_date:
                continue
    
            # Get Mark-to-Market Value of different positions after previous close
            pnl = 0
            for stock, position_info in entry.items():
                price, size = position_info.values()
                if stock != "hedge":
                    pnl += (price - self.close_df.iloc[i - 1, stock]) * size
                else:
                    pnl += (price - self.benchmark_data.Close[i - 1]) * size
                pnl -= commission_fee * abs(size)
    
            cash_value += pnl
            pnls.append(pnl)
    
            orders[self.close_df.index[i - 1]] = entry
    
            if i >= self.close_df.shape[0] or self.close_df.index[i] > end_date:
                break
    
            section_data = self.close_df.iloc[i - min_data_points:i, :]
            if section_data.isna().sum().sum() > 0:
                continue
    
            # Getting residuals and z-scores based on previous close
            residuals = self.get_residuals(section_data, 2)
            residuals = np.stack(residuals, axis = 0)
            z_scores = self.z_score(residuals)
            zs = dict(enumerate(z_scores[-1, :], 1))
    
    
            # Opening the position according to open price of current time
            
            entry = {}
            idx_long = (np.argsort([zs[j] for j in zs])[:max_position])
            idx_short = (np.argsort([zs[j] for j in zs])[-max_position:])
    
            beta = 0
            
            for long, short in zip(idx_long, idx_short):
                long_entry_price = self.open_df.iloc[i, long] * (1 + slippage)
                short_entry_price = self.open_df.iloc[i, short] * (1 - slippage)
                entry[long] = {'price': long_entry_price, "size": np.round(cash_value / (max_position * long_entry_price))}
                entry[short] = {'price': short_entry_price, "size": - np.round(cash_value / (max_position * short_entry_price))}
                beta += (self.beta_df.iloc[i - 1, long] - self.beta_df.iloc[i - 1, short]) / max_position
    
            
            # Beta Hedge
            if beta_hedge and abs(beta) > beta_limit:
                abs_difference = abs(beta) - beta_limit
                hedge_direction = -1 if beta > 0 else 1
                entry_price = self.benchmark_data.Open[i] * (1 + slippage) if hedge_direction > 0 else self.benchmark_data.Open[i] * (1 - slippage)
                entry["hedge"] = {"price": entry_price, "size": np.round(abs_difference * cash_value / entry_price) * hedge_direction}
                beta += abs_difference * hedge_direction
            
            betas.append(beta)
            dates.append(self.close_df.index[i])
    
        
        # Consolidate pnl data to df
        pd.options.display.float_format = "{:.3f}".format
        df = pd.DataFrame(pnls[1:], columns = ['daily_pnl'])
        df['ptf_value'] = self.initial_cash + df.daily_pnl.cumsum()
        df["ptf_beta"] = betas
        df.loc[-1] = [0, self.initial_cash, 0]
        df.index = df.index + 1
        df.sort_index(inplace = True)
        df.index = orders.keys()
    
        df['benchmark_close'] = self.benchmark_data.Close.loc[df.index]
        df['benchmark_val'] = df.benchmark_close / df.benchmark_close[0] * self.initial_cash
        df['ptf_return'] = df.ptf_value.pct_change()
        df['benchmark_return'] = df.benchmark_val.pct_change()
        
    
        return cash_value, df, orders        
    
    def store_results(self, df, orders):

        result_storage = "BacktestResults"
        if not os.path.exists(result_storage):
            os.mkdir(result_storage)
        
        orders_json = os.path.join(result_storage, "orders.json")
        with open(orders_json, "w") as f:
            json.dumps(orders)

        simulation_csv = os.path.join(result_storage, "ptf.csv")
        df.to_csv(simulation_csv, index = True)

    
    def analyze_results(self, df, orders):
        
        graph_storage = "Graphs"
        if not os.path.exists(graph_storage):
            os.mkdir(graph_storage)

        # Equity Curve
        df['benchmark_close'] = self.benchmark_data.Close[(self.benchmark_data.index >= self.start_date - BDay(1)) & (self.benchmark_data.index <= self.end_date)]
        df['benchmark_val'] = df.benchmark_close / df.benchmark_close[0] * self.initial_cash
        df['ptf_return'] = df.ptf_value.pct_change()
        df['benchmark_return'] = df.benchmark_val.pct_change()
        plt.plot(df.ptf_value, label = "Strategy")
        plt.plot(df.benchmark_val, label = "Benchmark")
        plt.legend()
        plt.savefig(os.path.join(graph_storage, "EquityCurveVSBenchmark.png"))

        
        df['ptf_return'] = df.ptf_value.pct_change().fillna(0)


        # Graph of Portfolio Beta
        plt.plot(df.ptf_beta)
        plt.title('Portfolio Beta')
        plt.xlabel("Time")
        plt.ylabel("Beta")
        plt.savefig(os.path.join(graph_storage, "PortfolioBetaGraph.png"))

    def backtest_statistics(self, df):

        ptf_returns = df.ptf_return.dropna()
        ptf_return = (df.ptf_value[-1] - df.ptf_value[0]) / df.ptf_value[0]
        avg_return = ptf_returns.mean()
        std_daily_return = ptf_returns.std()
        volatility = std_daily_return * np.sqrt(252)

        confidence_level = 0.95
        z_score_var = norm.ppf(confidence_level)
        VaR_percentage = - z_score_var * std_daily_return

        benchmark_returns = df.benchmark_return.dropna()
        model = sm.OLS(ptf_returns, sm.add_constant(benchmark_returns)).fit()
        alpha, beta = model.params

        sharpe = avg_return / std_daily_return * np.sqrt(252)

        window = 252
        roll_max = df.ptf_value.rolling(window, min_periods = 1).max()
        df['max_drawdown'] = df.ptf_value / roll_max - 1
        max_drawdown = -df.max_drawdown.min()

        statistics = {
            "ptf_return": ptf_return,
            "avg_return": avg_return,
            "volatility": volatility,
            "var": VaR_percentage,
            "maxdrawdown": max_drawdown,
            "alpha": alpha,
            "beta": beta,
            "sharpe": sharpe
        }
        return df, statistics
    
    def create_table(self):
        # Display in table
        table = PrettyTable()
        table.field_names = ["Test Name", "Portfolio Return (%)", "Average Return (%)", "Volatility (%)", "VaR (%)", "MDD(%)","Alpha (%)", "Beta", "Sharpe Ratio"]
        return table
    
    def add_data(self, table, data, test_name = "Backtest"):
        # Multiply all values except sharpe and alpha
        data = np.array(data)
        data[:-2] *= 100
        print(data)
        table.add_row([test_name] + list(data))
        return table
    
    def display_table(self, table):
        print(table)
    
    def run_simulation(self, max_pos, min_pos):
        pass

    def cartesian(self, arrays, out = None):
        arrays = [np.asarray(x) for x in arrays]
        dtype = arrays[0].dtype
    
        n = np.prod([x.size for x in arrays])
        if out is None:
            out = np.zeros([n, len(arrays)], dtype = dtype)
    
        m = int(n / arrays[0].size)
        out[:, 0] = np.repeat(arrays[0], m)
        if arrays[1:]:
            self.cartesian(arrays[1:], out = out[0:m, 1:])
            for j in range(1, arrays[0].size):
                out[j * m:(j+1) * m, 1:] = out[0:m, 1:]
        
        return out
    
    # Grid Search Optimization
    def optimize(self, start_date, end_date):
        max_pos_poss = list(range(1, 6))
        min_points_poss = list(range(10, 252, 10))
        all_possibilities = self.cartesian([max_pos_poss, min_points_poss])
        test_results = []
        
        for max_pos, min_point in tqdm.tqdm(all_possibilities):
            end_cash_value, final_df, orders = self.simulation(max_pos, min_point, True)
            final_df, statistics = self.backtest_statistics(final_df)
            statistics = pd.DataFrame.from_dict(statistics, orient = "index").T
            test_results.append(statistics)
        
        optim_res = pd.concat(test_results)
        optim_res[["max_pos", "min_points"]] = all_possibilities


if __name__ == "__main__":
    start_date = datetime(2019, 1, 1)
    end_date = datetime(2024, 1, 1)
    algo = PCAPtfStatArb(SPY_SECTOR_ETF, start_date, end_date)

    max_position = 3
    min_data_points = 170
    algo.run_simulation(max_position, min_data_points)
