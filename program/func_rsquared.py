import pandas as pd
import numpy as np
import statsmodels.api as sm
from constants import WINDOW  # Assuming WINDOW is defined for rolling calculations

# Function to calculate daily returns
def calculate_daily_returns(prices):
    return prices.pct_change().dropna()

# Calculate R-Squared Value
def calculate_r_squared(series_1, series_2):
    model = sm.OLS(series_1, sm.add_constant(series_2)).fit()
    return model.rsquared

# Store R-Squared Results
def store_r_squared_results(df_market_prices):
    markets = df_market_prices.columns.to_list()
    r_squared_pairs = []

    # Calculate daily returns for each market
    df_returns = df_market_prices.apply(calculate_daily_returns)

    # Ensure we have enough data points after calculating returns
    if len(df_returns) < WINDOW:
        print("Not enough data points for the specified WINDOW size.")
        return "Not enough data points"

    # Find R-squared value for each pair
    for base_market in markets[:-1]:
        for quote_market in markets[markets.index(base_market) + 1:]:
            # Use rolling window to calculate R-squared value over each WINDOW
            for start in range(len(df_returns) - WINDOW + 1):
                windowed_base = df_returns[base_market].iloc[start:start + WINDOW]
                windowed_quote = df_returns[quote_market].iloc[start:start + WINDOW]
                
                r_squared = calculate_r_squared(windowed_base, windowed_quote)
                
                # Store the result
                r_squared_pairs.append({
                    "base_market": base_market,
                    "quote_market": quote_market,
                    "R_squared": r_squared
                })

    # Create DataFrame from results
    df_r_squared = pd.DataFrame(r_squared_pairs)

    # Save DataFrame to CSV
    df_r_squared.to_csv("r_squared_pairs.csv", index=False)
    print("R-squared pairs successfully saved")
    return "saved"

# Example usage with a DataFrame `df_market_prices` containing daily prices for each token
# store_r_squared_results(df_market_prices)
