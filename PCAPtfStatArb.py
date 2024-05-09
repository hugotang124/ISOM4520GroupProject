# Data-related Imports
import numpy as np
import pandas as pd
import yfinance as yf

# General Imports
from datetime import datetime, timedelta
import os
from pytz import timezone
import argparse
import warnings
warnings.simplefilter(action = "ignore", category = FutureWarning)

# Statistical methods and models
from sklearn.decomposition import PCA
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm, skew, kurtosis

# Display output
from prettytable import PrettyTable
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns
import json
import tqdm

from concurrent.futures import ThreadPoolExecutor, as_completed


SPY_SECTOR_ETF = ["XLC", "XLY", "XLP", "XLE", "XLF", "XLV", "XLI", "XLB", "XLRE", "XLK", "XLU"]
BACKTEST_TABLE_FIELDS = ["Test Name", "Portfolio Return (%)", "Average Return (%)",
                         "Volatility (%)", "VaR (%)", "Expected Shortfall (%)" ,
                         "MDD(%)", "Win Rate (%)", "Long Win Rate (%)", "Short Win Rate (%)",
                         "Alpha (%)", "Beta", "Skewness", "Kurtosis", "Sharpe Ratio"]

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
        print("Getting market data")
        folder = 'ProjectData'
        if not os.path.exists(folder):
            os.mkdir(folder)
        
        holder = {}
        data_tickers = self.tickers.copy()
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

        
        print("Handling market data")
        
        self.benchmark_data = holder.pop(self.benchmark_sym)
        self.close_df, self.open_df, self.return_df, self.std_returns_df = self.generalize_data(holder)

        self.benchmark_data = self.benchmark_data.loc[self.close_df.index]
        self.benchmark_data['returns'] = self.benchmark_data.Close.pct_change()
        self.benchmark_returns = self.benchmark_data['returns'].loc[self.return_df.index]
        self.calculate_beta()

        
    def generalize_data(self, holder):
        # Generate dfs with close prices, returns, and standardized returns
        close_df = pd.concat(list(map(lambda x: x.Close, holder.values())), axis = 1)
        close_df.columns = list(holder.keys())
        open_df = pd.concat(list(map(lambda x: x.Open, holder.values())), axis = 1)
        open_df.columns = list(holder.keys())
        return_df = close_df.pct_change(axis = 0)
        std_returns_df = (return_df - return_df.rolling(252, axis = 0).mean())/ return_df.rolling(252, axis = 0).std()
        return close_df, open_df, return_df, std_returns_df
    
    def calculate_beta(self):
        # Get the beta of each ticker at every moment of time

        print("Calculating individual stock beta")
        self.beta_df = pd.DataFrame() 
        for stock in self.return_df.columns:
            self.beta_df[stock] = self.return_df[stock].rolling(22).cov(self.benchmark_returns) / self.benchmark_returns.rolling(22).var()
        self.beta_df.shift(1)

    def get_residuals(self, data, N = 1):
        # Get residuals of the PCA transformed dataset, assume stationarity

        pca = PCA(n_components = N)
        pca.fit(data)
        index = pca.components_[0]
        mm = [sm.OLS(s, index).fit() for s in data.values]
        res = list(map(lambda x: x.resid.T, mm))
        return res

    def z_score(self, data):
        # Standardisation of residuals
        scaler = StandardScaler()
        return scaler.fit_transform(data)
    
    def simulation(self, max_position, min_data_points, beta_limit = 10, beta_hedge = False):
        # Simulating trades, assume it will be opened during daily open and closed during daily close

        entry = {}
        pnls = []
        betas = []
        dates = []
        orders = {}

        # Assumptions 
        slippage = 0.00025
        commission_fee = 0.005

        beta_limit = beta_limit / 100
    
        if max_position >= self.close_df.shape[1] / 2:
            print("There's too many positions within the portfolio")
            return
    
        cash_value = self.initial_cash
        for i in range(self.close_df.shape[0] + 1):
            if i != self.close_df.shape[0] and self.close_df.index[i] < self.start_date:
                continue
    
            # Get Mark-to-Market Value of different positions after previous close
            daily_pnl = 0
            for stock, position_info in entry.items():
                price, size = position_info.values()
                if stock != self.benchmark_sym:
                    stock_col = self.tickers.index(stock)
                    trade_pnl = (price - self.close_df.iloc[i - 1, stock_col]) * size
                else:
                    trade_pnl = (price - self.benchmark_data.Close[i - 1]) * size
                
                trade_pnl -= 2 * commission_fee * abs(size)
                position_info["pnl"] = trade_pnl
                daily_pnl += trade_pnl
                
            cash_value += daily_pnl
            pnls.append(daily_pnl)
    
            orders[str(self.close_df.index[i - 1])] = entry
    
            if i >= self.close_df.shape[0]:
                break

            if self.close_df.index[i] > self.end_date:
                break
    
            section_data = self.close_df.iloc[i - min_data_points:i, :]

            # Ensure the start point is when all stocks in the basket is available to trade
            if section_data.isna().sum().sum() > 0:
                continue
    
            # Getting residuals and z-scores based on previous close
            residuals = self.get_residuals(section_data, 1)
            residuals = np.stack(residuals, axis = 0)
            z_scores = self.z_score(residuals)
            zs = dict(enumerate(z_scores[-1, :], 2))
    
    
            # Opening the position according to open price of current time
            
            # Filter the stocks to get n largest z-scores and n lowest z-scores
            entry = {}
            idx_long = (np.argsort([zs[j] for j in zs])[:max_position])
            idx_short = (np.argsort([zs[j] for j in zs])[-max_position:])
    
            beta = 0
            
            # Enter the position
            for long, short in zip(idx_long, idx_short):
                long_entry_price = self.open_df.iloc[i, long] * (1 + slippage)
                short_entry_price = self.open_df.iloc[i, short] * (1 - slippage)
                entry[self.tickers[long]] = {'price': long_entry_price, "size": np.round(cash_value / (max_position * long_entry_price))}
                entry[self.tickers[short]] = {'price': short_entry_price, "size": - np.round(cash_value / (max_position * short_entry_price))}
                beta += (self.beta_df.iloc[i - 1, long] - self.beta_df.iloc[i - 1, short]) / (2 * max_position)
    
            # Beta Hedge
            if beta_hedge and abs(beta) > beta_limit:
                abs_difference = abs(beta) - beta_limit
                hedge_direction = -1 if beta > 0 else 1
                entry_price = self.benchmark_data.Open[i] * (1 + slippage) if hedge_direction > 0 else self.benchmark_data.Open[i] * (1 - slippage)
                entry[self.benchmark_sym] = {"price": entry_price, "size": np.round(abs_difference * cash_value / entry_price) * hedge_direction}
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
        df.index = pd.to_datetime(list(orders.keys()))

        # Creating columns for evaluation purposes
        df['benchmark_close'] = self.benchmark_data.Close.loc[df.index]
        df['benchmark_val'] = df.benchmark_close / df.benchmark_close[0] * self.initial_cash
        df['ptf_return'] = df.ptf_value.pct_change()
        df['benchmark_return'] = df.benchmark_val.pct_change()

        previous_peaks = df.ptf_value.cummax()
        df['max_drawdown'] = (df.ptf_value - previous_peaks) / previous_peaks
    
        return cash_value, df, orders        
    
    def store_results(self, df, orders, table):

        print("Storing backtest results")

        result_storage = "BacktestResults"
        if not os.path.exists(result_storage):
            os.mkdir(result_storage)
        
        # Summary statistics
        summary = os.path.join(result_storage, "summary.txt")
        with open(summary, "w") as f:
            f.write(table.get_string())
        
        # Orders in this simulation
        orders_json = os.path.join(result_storage, "orders.json")
        with open(orders_json, "w") as f:
            json.dump(orders, f)

        # Data of portfolio vs benchmark
        simulation_csv = os.path.join(result_storage, "ptf.csv")
        df.to_csv(simulation_csv, index = True)

    
    def plot_graphs(self, df, statistics):

        print("Plotting graphs")
        
        graph_storage = "Graphs"
        if not os.path.exists(graph_storage):
            os.mkdir(graph_storage)

        # Equity Curve
        equity_curve = plt.figure(figsize = (10, 10))
        plt.plot(df.ptf_value, label = "Strategy")
        plt.plot(df.benchmark_val, label = "Benchmark")
        plt.title("Equity Curve")
        plt.ylabel("Value")
        plt.xlabel("Time")
        plt.legend()
        equity_curve.savefig(os.path.join(graph_storage, "EquityCurveVSBenchmark.png"))

        # Portfolio Beta
        beta_curve = plt.figure(figsize = (10, 10))
        plt.plot(df.ptf_beta)
        plt.title('Portfolio Beta')
        plt.xlabel("Time")
        plt.ylabel("Beta")
        plt.savefig(os.path.join(graph_storage, "PortfolioBetaGraph.png"))

        # Max Drawdown Curve
        mdd_curve = plt.figure(figsize = (10, 10))
        plt.plot(df.max_drawdown)
        plt.title("Max Drawdown Curve")
        plt.title('Portfolio Max Drawdown')
        plt.xlabel("Time")
        plt.ylabel("Drawdown")
        plt.savefig(os.path.join(graph_storage, "PortfolioMDDGraph.png"))

        # Returns distribution plot with VaR line
        ptf_returns = df.ptf_return.dropna()
        return_dist = plt.figure(figsize = (10, 10))
        mu = ptf_returns.mean()
        std = ptf_returns.std()
        plt.hist(df.ptf_return.dropna(), bins = 50)
        plt.axvline(statistics['var'], color = 'r', label = "VaR")
        plt.title("Return Distribution")
        plt.xlabel("Returns")
        plt.legend()
        plt.savefig(os.path.join(graph_storage, "ReturnDistributionGraph.png"))

        # Returns Q-Q plot
        qq_plot = plt.figure(figsize = (10, 10))
        sm.qqplot(df.ptf_return.dropna() * 100, line = "45")
        plt.title("Q-Q Plot of Returns against Normal Distribution")
        plt.savefig(os.path.join(graph_storage, "ReturnQQPlot.png"))




    def backtest_statistics(self, df, orders):

        ptf_return = (df.ptf_value[-1] - df.ptf_value[0]) / df.ptf_value[0]

        df_b = df.copy().dropna()
        ptf_returns = df_b.ptf_return
        avg_return = ptf_returns.mean()
        
        std_daily_return = ptf_returns.std()
        volatility = std_daily_return * np.sqrt(252)

        confidence_level = 0.95
        z_score_var = norm.ppf(confidence_level)
        VaR_percentage = - z_score_var * std_daily_return
        

        benchmark_returns = df_b.benchmark_return
        model = sm.OLS(ptf_returns, sm.add_constant(benchmark_returns)).fit()
        alpha, beta = model.params

        sharpe = avg_return / std_daily_return * np.sqrt(252)

        max_drawdown = -df_b.max_drawdown.min()

        orders_df  = pd.DataFrame.from_dict({(i,j): orders[i][j] 
                           for i in orders.keys() 
                           for j in orders[i].keys()},
                       orient='index')
        orders_df.index = orders_df.index.set_names(["time", "symbol"])
        orders_df.reset_index(inplace = True)
    
        orders_df['direction'] = np.where(orders_df['size'] > 0, 1, -1)
        orders_df['size'] = np.abs(orders_df['size'])
        orders_df["order_return"] = orders_df.pnl / (orders_df.price * orders_df.size * orders_df.direction)
        orders_df = orders_df[orders_df.symbol != "SPY"]
        win_rate = orders_df[orders_df.order_return > 0].shape[0] / orders_df.shape[0]
        long_orders = orders_df[orders_df.direction == 1]
        long_win_rate = long_orders[long_orders.order_return > 0].shape[0] / long_orders.shape[0]
        short_orders = orders_df[orders_df.direction == -1]
        short_win_rate = short_orders[short_orders.order_return > 0].shape[0] / short_orders.shape[0]


        skewness_val = skew(df_b.ptf_return)

        kurtosis_val = kurtosis(df_b.ptf_return)

        expected_shortfall = df_b[df_b.ptf_return < VaR_percentage].ptf_return.mean()

        statistics = {
            "ptf_return": ptf_return,
            "avg_return": avg_return,
            "volatility": volatility,
            "var": VaR_percentage,
            "expexted_shortfall": expected_shortfall,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "long_win_rate": long_win_rate,
            "short_win_rate": short_win_rate,
            "alpha": alpha,
            "beta": beta,
            "skewness": skewness_val,
            "kurtosis": kurtosis_val,
            "sharpe": sharpe
        }
        return df, statistics
    
    def create_table(self, fields):
        # Display in table
        table = PrettyTable()
        table.field_names = fields
        return table
    
    def run_simulation(self, max_pos, min_datapoints, beta_limit = 10, beta_hedge = False):
        print(f"Running simulation on max_pos = {max_pos}, min_datapoints = {min_datapoints}, beta_limit = {beta_limit / 100}, beta_hedge = {beta_hedge}")
        end_cash_value, df, orders = self.simulation(max_pos, min_datapoints, beta_limit, beta_hedge)
        df, statistics = self.backtest_statistics(df, orders)
        table = self.create_table([f"Case_{max_pos}_{min_datapoints}_{beta_limit}_{beta_hedge}", "Stats"])
        stats = np.array(list(statistics.values()))
        stats[:-4] *= 100
        stats = np.round(stats, 3)
        for row_name, stat in zip(BACKTEST_TABLE_FIELDS[1:], list(stats)):
            table.add_row([row_name, stat], divider = True)
        
        stats_df = pd.DataFrame(stats, index = BACKTEST_TABLE_FIELDS[1:])
        stats_df.columns = ['Stats']
        stats_df.index.name = f"Case_{max_pos}_{min_datapoints}_{beta_hedge}"

        self.plot_graphs(df, statistics)
        self.store_results(df, orders, table)

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

    test_results = []
    
    def optimize(self, arrays, optim_columns):
        all_possibilities = self.cartesian(arrays)

        print("Starting optimisation")

        for possibility in tqdm.tqdm(all_possibilities):
            end_cash_value, final_df, orders = self.simulation(*possibility)
            final_df, statistics = self.backtest_statistics(final_df, orders)
            statistics = pd.DataFrame.from_dict(statistics, orient = "index").T
            self.test_results.append(statistics)
        
        print("Finished optimisation")
        
        optim_res = pd.concat(self.test_results)
        optim_res[optim_columns] = all_possibilities
        optim_res['backtest_name'] = optim_res.apply(lambda x: "Case_" + "_".join([str(x[column]) for column in optim_columns]), axis = 1)
        optim_res.set_index("backtest_name", inplace = True)
        optim_res.sort_values(by = "sharpe", ascending = False, inplace = True)
        return optim_res.drop(columns = optim_columns)
    
    def run_optimize(self, hedge = True):
        max_pos_poss = list(range(1, 6))
        min_points_poss = list(range(10, 252, 10))
        optim_arrays = [max_pos_poss, min_points_poss]
        optim_columns = ["max_pos", "min_points"]
        if hedge:
            beta_limit_choices = list(range(0, 55, 5))
            optim_arrays.append(beta_limit_choices)
            optim_columns += ["beta_limit", "hedge"]

        optim_res = self.optimize(optim_arrays, optim_columns)
        optim_res.to_csv("OptimizationResults.csv", index = True)
        best_params = optim_res.index[0].split("_")[1:]
        print(tabulate(optim_res, headers = optim_res.columns))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog = "Script that run backtesting and optimization with PCA Strategy")
    parser.add_argument("-s", "--start-date", action = "store", default = "20220101", help = "Start date")
    parser.add_argument("-e", "--end-date", action = "store", default = "20240101", help = "End date")
    parser.add_argument("-o", "--optimise", action = "store_true", default = False, help = "Optimization or not")
    parser.add_argument("-b", "--backtest", action = "store_true", default = False, help = "Backtest or not")
    parser.add_argument("-t", "--tickers", action = "store", default = "", help = "Tickers, SPY Sector ETFs if not defined")
    parser.add_argument("-m", "--max-pos", action = "store", default = 0, type = int, help = "Max positions if backtest")
    parser.add_argument("-p", "--min-points", action = "store", default = 0, type = int, help = "Min datapoints for PCA if backtest")
    parser.add_argument("--beta-limit", action = "store", default = 0, type = int, help = "Beta limit multiplied by 100 for backtest")
    parser.add_argument("--hedge", action = "store_true", default = False, help = "Hedge included in backtest or optimisation")

    args = parser.parse_args()
    start_date = datetime.strptime(args.start_date, "%Y%m%d")
    end_date = datetime.strptime(args.end_date, "%Y%m%d")
    if start_date > end_date:
        start_date, end_date = end_date, start_date

    tickers = SPY_SECTOR_ETF if args.tickers else args.tickers.split(",")

    algo = PCAPtfStatArb(SPY_SECTOR_ETF, start_date, end_date)
    
    if args.backtest:
        if (args.max_pos <= 0) and (not args.min_points <= 0):
            print("Please define correct params for max_pos and min_points")
        algo.run_simulation(args.max_pos, args.min_points, args.beta_limit, args.hedge)
    elif args.optimise:
        algo.run_optimize(args.hedge)

    
    
