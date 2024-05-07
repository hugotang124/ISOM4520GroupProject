import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
from scipy.stats import linregress
from sklearn.decomposition import PCA
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
import json

class PCAPtfStatArb:
    
    max_data_days_coverage = 365
    benchmark_sym = "SPY"

    def __init__(self, tickers, start_date, end_date):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.get_market_data()
    
    def get_market_data(self):
        folder = 'ProjectData'
        if not os.path.exists(folder):
            os.mkdir(folder)
        
        holder = {}
        if self.benchmark_sym not in tickers:
            tickers += [self.benchmark_sym]

        for ticker in self.tickers:
            file_name = f'{folder}/{ticker}.csv'
            real_start_date = start_date - timedelta(days = self.max_data_days_coverage)
            df = yf.Ticker(ticker).history(start = real_start_date.strftime("%Y-%m-%d"), end = end_date.strftime("%Y-%m-%d"))
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
    

    def simulation(self, max_position, initial_cash, min_data_points):
        '''
        Simulation function
    
        Assume trades will be opened during daily open and closed during daily close
        '''

        open_simu = self.open_df
        close_simu = self.close_df


        entry = {}
        pnls = []
        orders = {}
        betas = []
    
        if open_simu.shape != close_simu.shape:
            return
    
        if max_position >= close_simu.shape[1] / 2:
            print("There's too many positions within the portfolio")
            return
    
        cash_value = initial_cash
        for i in range(close_simu.shape[0] + 1):
            if i < min_data_points:
                continue
    
            # Getting residuals and z-scores based on previous close
            residuals = self.get_residuals(close_simu.iloc[i - min_data_points:i, :], 2)
            residuals = np.stack(residuals, axis = 0)
            z_scores = self.z_score(residuals)
            zs = dict(enumerate(z_scores[-1, :], 1))
    
            # Get Mark-to-Market Value of different positions after previous close
            pnl = 0
            for stock, position_info in entry.items():
                price, size = position_info.values()
                pnl += (price - close_simu.iloc[i - 1, stock]) * size
            cash_value += pnl
            pnls.append(pnl)
            
            orders[close_simu.index[i - 1]] = entry
            
            if i >= close_simu.shape[0]:
                break
    
            # Opening the position according to open price of current time
            
            entry = {}
            idx_long = (np.argsort([zs[j] for j in zs])[:max_position])
            idx_short = (np.argsort([zs[j] for j in zs])[-max_position:])

            beta = 0
            
            for long, short in zip(idx_long, idx_short):
                long_entry_price = open_simu.iloc[i, long]
                short_entry_price = open_simu.iloc[i, short]
                entry[long] = {'price': long_entry_price, "size": np.round(cash_value / (max_position * long_entry_price))}
                entry[short] = {'price': short_entry_price, "size": - np.round(cash_value / (max_position * short_entry_price))}
                beta += (self.beta_df.iloc[i, long] + self.beta_df.iloc[1, short]) / max_position

            betas.append(beta) 
    
        
        # Consolidate pnl and beta data to df
        pnls = [0] * (min_data_points - 1) + pnls
        betas = [0] * (min_data_points - 1) + betas
        pd.options.display.float_format = "{:.3f}".format
        df = pd.DataFrame(pnls, columns = ['daily_pnl'])
        df['ptf_value'] = initial_cash + df.daily_pnl.cumsum()
        df['ptf_beta'] = betas
        df.index = close_simu.index


        if self.store:
            self.store_results(df, orders)
        
        self.analyze_results(df, orders)
        
    
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
        
        df['ptf_return'] = df.ptf_value.pct_change().fillna(0)


        


        
        # Graph of Portfolio Beta
        plt.plot(df.ptf_beta)
        plt.title('Portfolio Beta')
        plt.xlabel("Time")
        plt.ylabel("Beta")
        plt.savefig(os.path.join(graph_storage, "PortfolioBetaGraph.png"))
        







if __name__ == "__main__":
    end_date = datetime.now() - timedelta(days = 1)
    start_date = end_date - timedelta(days = 5 * 365)
    tickers = ["XLC", "XLY", "XLP", "XLE", "XLF", "XLV", "XLI", "XLB", "XLRE", "XLK", "XLU", "SPY"]
    algo = PCAPtfStatArb(tickers, start_date, end_date)

    max_position = 3
    initial_cash = 10e6
    min_data_points = 170
    algo.simulation(max_position, initial_cash, min_data_points)
