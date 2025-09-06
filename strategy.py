import pandas as pd
import numpy as np
import yfinance as yf

class StaticPCAStrategy:
    def __init__(self, ticker_list: list, start: str, end: str, rolling_window: int):
        self.ticker_list = ticker_list
        self.start = start
        self.end = end
        self.lookback = rolling_window


    def import_data(self) -> pd.DataFrame:
        data = yf.download(self. ticker_list, start = self.start, end = self.end, auto_adjust= True, progress = False)["Close"]
        return data
    
    def compute_log_returns(self, data: pd.DataFrame) -> pd.DataFrame:
        log_returns = np.log(data) - np.log(data.shift(1))
        rolling_mean = log_returns.rolling(window = self.lookback).mean()
        centered_returns = log_returns - rolling_mean
        returns = centered_returns.dropna()
        
        return returns
    
    def compute_sample_cov_matrix(self, returns: pd.DataFrame) -> pd.DataFrame:
        X = returns.to_numpy(dtype = float, copy = False)
        X = X - X.mean(axis = 0, keepdims = True)

        T_prime = X.shape[0]

        sample_cov_matrix = (X.T @ X) / (T_prime - 1)
        
        return pd.DataFrame(sample_cov_matrix, index = returns.columns, columns = returns.columns)

    def compute_shrinkage_target(self, sample_cov_matrix: pd.DataFrame) -> pd.DataFrame:
        n = sample_cov_matrix.shape[0]
        variances = np.diag(sample_cov_matrix.values)  

        stddev_outer = np.sqrt(np.outer(variances, variances))
        corr = sample_cov_matrix.values / stddev_outer

        off_diag_mask = ~ np.eye(n, dtype = bool)
        avg_corr = float(np.mean(corr[off_diag_mask]))
 
        target_corr = np.full((n, n), avg_corr)
        np.fill_diagonal(target_corr, 1)
        target_vals = target_corr * stddev_outer

        return pd.DataFrame(target_vals, index = sample_cov_matrix.index, columns = sample_cov_matrix.columns)
    
    def compute_shrunk_cov_matrix(self, returns: pd.DataFrame) -> pd.DataFrame:
        sample_cov_matrix = self.compute_sample_cov_matrix(returns)
        target_matrix = self.compute_shrinkage_target(sample_cov_matrix)

        X = returns.to_numpy(dtype = float, copy = False)
        X = X - X.mean(axis = 0, keepdims = True)

        X_prod = X[:, :, None] * X[:, None, :]
        numerator_matrix = np.mean((X_prod - sample_cov_matrix.values[None, :, :]) ** 2, axis = 0)
        numerator_total = numerator_matrix.sum()

        denominator_total = ((sample_cov_matrix.values - target_matrix.values) ** 2).sum()

        if denominator_total == 0.0:
            delta = 0.0
            print("This means the sample covariance matrix perfectly matched the target matrix.")
        else:
            delta = min(1, max(0, numerator_total / denominator_total))

        sigma_hat = delta * target_matrix.values + (1 - delta) * sample_cov_matrix.values

        return pd.DataFrame(sigma_hat, index = sample_cov_matrix.index, columns = sample_cov_matrix.columns)
    
    def eigendecomposition(self, shrunk_cov_matrix: pd.DataFrame):
        shrunk_cov_matrix_array = shrunk_cov_matrix.to_numpy(dtype = float, copy = False)
        eigenvalues, eigenvectors = np.linalg.eigh(shrunk_cov_matrix_array)

        descending_order_indices = eigenvalues.argsort()[:: -1]   
        eigenvalues = eigenvalues[descending_order_indices]
        eigenvectors = eigenvectors[:, descending_order_indices]

        sigma = np.median(eigenvalues)

        Q = shrunk_cov_matrix.shape[0] / self.lookback
        lambda_min = sigma * (1 - np.sqrt(Q)) ** 2
        lambda_max = sigma * (1 + np.sqrt(Q)) ** 2

        signal_mask = eigenvalues > lambda_max

        V_k = eigenvectors[:, signal_mask]
        Lambda_k = np.diag(eigenvalues[signal_mask])

        return V_k, Lambda_k, eigenvalues
    
    def reconstruct_stabilized_cov(self, V_k: np.ndarray, Lambda_k: np.ndarray, eigenvalues: np.ndarray) -> np.ndarray:
        n = V_k.shape[0]
        k = V_k.shape[1]
        Sigma_signal = V_k @ Lambda_k @ V_k.T
        P_noise = np.eye(n) - V_k @ V_k.T

        lambda_noise = np.mean(eigenvalues[k:]) if len(eigenvalues) > k else 0.0
        Sigma_final = Sigma_signal+ lambda_noise * P_noise

        return Sigma_final
    
    def compute_gmv_weights(self, Sigma_final: pd.DataFrame) -> np.ndarray:
        inv_Sigma_final = np.linalg.inv(Sigma_final)
        ones = np.ones(inv_Sigma_final.shape[0])
        weights = inv_Sigma_final @ ones / (ones.T @ inv_Sigma_final @ ones)

        return weights
    
    def generate_signal(self, past_data: pd.DataFrame):
        log_data = self.compute_log_returns(past_data)

        if len(log_data) < self.lookback:
            num_assets = past_data.shape[1]
            return np.zeros(num_assets)
        
        window_data = log_data.iloc[- self.lookback: ]
        shrunk_cov_matrix = self.compute_shrunk_cov_matrix(window_data)
        V_k, Lambda_k, eigenvalues = self.eigendecomposition(shrunk_cov_matrix)
        Sigma_final = self.reconstruct_stabilized_cov(V_k, Lambda_k, eigenvalues)
        weights = self.compute_gmv_weights(Sigma_final)

        return weights