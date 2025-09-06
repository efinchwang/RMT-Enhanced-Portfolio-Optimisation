import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from strategy import StaticPCAStrategy

class Backtester:
    def __init__(self, strategy, data):
        self.strategy = strategy
        self.data = data
        self.assets = self.data.columns.tolist()
        self.results = pd.DataFrame(index = data.index)
        for asset in self.assets:
            self.results[asset] = 0.0
        self.results["daily_return"] = 0.0

    def run(self):
        for t in range(self.strategy.lookback, len(self.data)):
            past_data = self.data.iloc[: t]
            prev_prices = self.data.iloc[t - 1]
            current_prices = self.data.iloc[t]

            weights = self.strategy.generate_signal(past_data = past_data)
            
            asset_returns = (current_prices.values - prev_prices.values) / prev_prices.values
            daily_return = np.dot(weights, asset_returns)
            
            self.results.loc[self.results.index[t], [asset for asset in self.assets]] = weights
            self.results.loc[self.results.index[t], "daily_return"] = daily_return
        return
    
    def calculate_performance_metrics(self):
        equity_curve = (1 + self.results["daily_return"]).cumprod()
        final_equity = equity_curve.iloc[-1]

        num_years = len(equity_curve) / 252
        annualised_return = final_equity ** (1 / num_years) - 1

        annualised_volatility = self.results["daily_return"].std() * np.sqrt(252)

        sharpe_ratio = annualised_return / annualised_volatility

        running_max = equity_curve.cummax()
        drawdown = (equity_curve - running_max) / running_max
        max_drawdown = drawdown.min()

        performance = {
        "Equity Curve": equity_curve,
        "Annualised Return": annualised_return,
        "Annualised Volatility": annualised_volatility,
        "Sharpe Ratio": sharpe_ratio,
        "Drawdown": drawdown,
        "Maximum Drawdown": max_drawdown
        }

        return performance
    
    def plot_results(self) -> None:
        performance = self.calculate_performance_metrics()
        
        equity_curve = performance["Equity Curve"]

        plt.figure()
        plt.plot(equity_curve.index, equity_curve.values, label = "Strategy Equity Curve", color = "blue")
        plt.title("Strategy Equity Curve")
        plt.xlabel("Date")
        plt.ylabel("Cumulative Return")
        plt.legend()
        plt.grid(True)
        plt.show()

        drawdown = performance["Drawdown"]

        plt.figure()
        plt.plot(drawdown.index, drawdown.values * 100, color = "red")
        plt.fill_between(drawdown.index, drawdown.values * 100, 0, color="red", alpha=0.3)
        plt.title("Strategy Drawdown")
        plt.xlabel("Date")
        plt.ylabel("Drawdown (%)")
        plt.grid(True)
        plt.show()
        return