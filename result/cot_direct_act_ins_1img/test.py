
# response = "We need to chose that:\n[BEGIN](2): take pot 1 from desk 2[END]. This is correct action.\n And the next is [BEGIN](2): go to desk 1[END]."

# import re

# def refine_action(response):
#     # Split the response by "**Response:**" to separate the header from the actions
#     parts = response.split('**Response:**')
#     actions = parts[1] if len(parts) > 1 else response
#     # Regular expression to find all matches of the pattern "optional leading characters (number): some action"
#     # matches = re.findall(r"[-\s]*\(\d+\): ([^\n]+)", actions)
#     matches = re.findall(r"\[BEGIN\]\((\d+)\): ([^\[]+)\[END\]", actions)
#     if not matches:
#         return "No action"
#     # Extract and return the first action if any matches are found
#     first_action = matches[0][1].strip() if matches else "No action"
#     return first_action


# action = refine_action(response)
# print(action)

from scipy.stats import norm
import numpy as np

# Given data
S0 = 1800  # Current price of gold
K = 1850  # Strike price
r = 0.01  # Risk-free interest rate
T = 0.5  # Time to expiration in years
sigma = 0.20  # Volatility

# Calculating d1 and d2
d1 = (np.log(S0 / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
d2 = d1 - sigma * np.sqrt(T)

# Calculating the call option price
C = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


# Hypothetical daily closing prices of gold for one week
prices = [1800, 1810, 1800, 1790, 1795]

# Calculate daily returns
daily_returns = [np.log(prices[i] / prices[i-1]) for i in range(1, len(prices))]

# Compute the standard deviation of daily returns (weekly volatility)
weekly_volatility = np.std(daily_returns)

# Annualize the volatility
annualized_volatility = weekly_volatility * np.sqrt(52)  # Using 52 to represent the number of weeks in a year

weekly_volatility, annualized_volatility