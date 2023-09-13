# %%
import yfinance as yf
import pandas as pd
import numpy as np
from pandas_datareader import data as web
import matplotlib.pyplot as plt
import talib
from datetime import datetime
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns

# %%
yf.pdr_override()


def read_stocks():
    return ["AAPL", "AMZN", "GOOGL", "QCOM", "ICLN"]


# Time
startday = "2023-01-01"
enddate = datetime.today().strftime("%Y-%m-%d")


# %%
# loop

df = pd.DataFrame()

for stock in read_stocks():
    df[stock] = web.get_data_yahoo(
        stock, start=startday, end=enddate)["Adj Close"]


# %%
# plot
plt.figure(figsize=(8, 6))

for c in df.columns.values:
    plt.plot(df[c], label=c)


plt.grid()
plt.title("Multiple Stocks")
plt.xlabel("Date")
plt.ylabel("Close $USD")
plt.legend(df.columns.values, loc="upper left")
plt.show()

# %%
# change to %
returns = df.pct_change()
returns

# %%
# annualized covariance matrix

cov_matrix_annual = returns.cov()*252
cov_matrix_annual

# %%
# portfolio variance
weight = np.array([0.2, 0.2, 0.2, 0.2, 0.2])  # 權重假設每個均分

port_variance = np.dot(weight.T, np.dot(cov_matrix_annual, weight))
port_variance  # 計算銓重配置投資組合的方差

# %%
# 投資組合的波動率，這個值表示了投資組合的預期年化波動程度
port_volatlity = np.sqrt(port_variance)
port_volatlity

# %%
# 平均收益率乘以其在投資組合中的權重
portfolioSimpleAnnualReturn = np.sum(returns.mean()*weight) * 252
portfolioSimpleAnnualReturn

# %%
percent_var = str(round(port_variance, 2) * 100) + "%"
percent_vols = str(round(port_volatlity, 2) * 100) + "%"
percent_ret = str(round(portfolioSimpleAnnualReturn, 2) * 100) + '%'

print("Expected annual return :" + percent_ret)
print("Annual Volatility(risk) :" + percent_vols)
print("Annual variance :" + percent_var)

# %%
plt.scatter(port_volatlity, portfolioSimpleAnnualReturn,
            marker="o", color="green", label="Investment")
plt.title("Invest")
plt.xlabel("Port_volatlity")
plt.ylabel("Annual Return")
plt.legend()
plt.show()

# %%
mu = expected_returns.mean_historical_return(df)
S = risk_models.sample_cov(df)


# %%
ef = EfficientFrontier(mu, S)
weight = ef.max_sharpe()
cleaned_weight = ef.clean_weights()
print(cleaned_weight)
ef.portfolio_performance(verbose=True)
