# %% [markdown]
# # EMiF Project 2: Has the structure of risk changed since COVID-19?
# **Course:** Empirical Methods in Finance  
# **Objective:** Analyze the non-linear structure of risk in financial markets pre- and post-COVID-19 using GARCH, Markov-Switching, and Quantile Regressions.

# %%
# ==========================================================
# CONFIGURATION & IMPORTS
# ==========================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from arch import arch_model
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from statsmodels.regression.quantile_regression import QuantReg
import warnings

plt.style.use('seaborn-v0_8-whitegrid')
warnings.filterwarnings('ignore')
print("Libraries imported successfully.")

# %% [markdown]
# ## Part 1: Data Preparation & Regime Splitting
# We load the dataset and compute daily log returns. To evaluate the structural change, we split the sample around the COVID-19 market crash (March 23, 2020).

# %%
# ==========================================================
# PART 1: DATA PREPARATION
# ==========================================================
# Load data 
try:
    df = pd.read_excel('Data.xlsx')
except FileNotFoundError:
    print("Error: 'Data.xlsx' not found. Please ensure the file is in the same directory.")

df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Calculate daily log returns in percentage
returns = np.log(df / df.shift(1)).dropna() * 100

# Define the COVID-19 structural break date
covid_break = '2020-03-23'

ret_pre = returns.loc[:covid_break]
ret_post = returns.loc[covid_break:]

print(f"Pre-COVID sample:  {ret_pre.index[0].date()} to {ret_pre.index[-1].date()} ({len(ret_pre)} obs)")
print(f"Post-COVID sample: {ret_post.index[0].date()} to {ret_post.index[-1].date()} ({len(ret_post)} obs)")

# %% [markdown]
# ## Part 2: Volatility Structure (Class 9 - GARCH)
# **Question:** Has volatility persistence changed post-COVID?  
# We estimate a GARCH(1,1) model. The sum of alpha (shock) and beta (persistence) indicates the degree of volatility clustering in the market.

# %%
# ==========================================================
# PART 2: VOLATILITY STRUCTURE
# ==========================================================
def estimate_garch(series, name):
    # Estimate a GARCH(1,1) model
    model = arch_model(series, vol='Garch', p=1, q=1, rescale=False)
    fit = model.fit(disp='off')
    alpha = fit.params['alpha[1]']
    beta = fit.params['beta[1]']
    print(f"[{name}] Alpha (Shock): {alpha:.4f} | Beta (Persistence): {beta:.4f} | Sum: {alpha+beta:.4f}")
    return fit

print("--- S&P 500 GARCH(1,1) Pre vs Post COVID ---")
garch_pre = estimate_garch(ret_pre['S&P500'], "PRE-COVID ")
garch_post = estimate_garch(ret_post['S&P500'], "POST-COVID")

# %% [markdown]
# ## Part 3: Regime Probabilities (Class 12 - Markov Switching)
# **Question:** Are markets spending more time in the 'High Risk' regime?  
# By modeling the absolute variance with a 2-state Markov-Switching model, we can filter the probability of being in a high-volatility state over time.

# %%
# ==========================================================
# PART 3: REGIME PROBABILITIES
# ==========================================================
# Modeling the absolute variance of the S&P 500 to identify risk regimes
y_ms = returns['S&P500'].abs()
ms_model = MarkovRegression(y_ms, k_regimes=2, trend='c', switching_variance=True).fit(disp=False)
print("Markov-Switching Model successfully fitted to S&P 500 absolute returns.")

# %% [markdown]
# ## Part 4: Tail Risk Dynamics (Class 13 - Quantile Regression)
# **Question:** How does Credit Risk (US High Yield) transmit to Equities (S&P500)?  
# We compare the mean behavior (OLS) with the behavior during extreme market stress (5% Quantile). A change in the tail-risk beta indicates a shift in the non-linear structure of risk.

# %%
# ==========================================================
# PART 4: TAIL RISK DYNAMICS 
# ==========================================================
def run_risk_regressions(data, period_name):
    # Y = Equity Market, X = US High Yield Bonds
    Y = data['S&P500']
    X = sm.add_constant(data['US HY Bonds'])
    
    # 1. Classical Linear Model (Mean behavior - OLS with HAC errors)
    ols = sm.OLS(Y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 5})
    
    # 2. Non-Linear Model (Extreme Tail Risk - 5% Quantile)
    quant_mod = QuantReg(Y, X)
    quant_res = quant_mod.fit(q=0.05)
    
    print(f"\n--- {period_name} ---")
    print(f"OLS Beta (Mean behavior) : {ols.params['US HY Bonds']:.4f} (p-val: {ols.pvalues['US HY Bonds']:.4f})")
    print(f"Q(0.05) Beta (Tail Risk) : {quant_res.params['US HY Bonds']:.4f} (p-val: {quant_res.pvalues['US HY Bonds']:.4f})")
    
    return ols, quant_res

ols_pre, q_pre = run_risk_regressions(ret_pre, "PRE-COVID")
ols_post, q_post = run_risk_regressions(ret_post, "POST-COVID")

# %% [markdown]
# ## Part 5: Professional Visualizations
# Visual synthesis of the structural change in market regimes and tail risk sensitivities.

# %%
# ==========================================================
# PART 5: PROFESSIONAL VISUALIZATIONS
# ==========================================================
fig, ax = plt.subplots(2, 1, figsize=(12, 10))

# Chart 1: Markov Regimes 
smoothed_probs = ms_model.smoothed_marginal_probabilities[1] # Probability of High Volatility Regime
ax[0].plot(returns.index, returns['S&P500'], color='lightgrey', label='S&P 500 Returns')
ax[0].set_ylabel("Returns (%)")

ax2 = ax[0].twinx()
ax2.plot(returns.index, smoothed_probs, color='red', alpha=0.6, label='Prob(High Risk Regime)')
ax2.set_ylabel("Probability")

ax[0].set_title("Market Returns and Probability of High-Risk Regime (Markov Switching)", fontweight='bold')
ax[0].axvline(pd.to_datetime(covid_break), color='black', linestyle='--', linewidth=2, label='COVID-19 Break')
ax[0].legend(loc='upper left')
ax2.legend(loc='upper right')

# Chart 2: Tail Risk Structure (Class 13)
labels = ['Pre-COVID (Mean)', 'Pre-COVID (5% Tail Risk)', 'Post-COVID (Mean)', 'Post-COVID (5% Tail Risk)']
betas = [ols_pre.params['US HY Bonds'], q_pre.params['US HY Bonds'], 
         ols_post.params['US HY Bonds'], q_post.params['US HY Bonds']]
colors = ['skyblue', 'darkblue', 'lightcoral', 'darkred']

sns.barplot(x=labels, y=betas, palette=colors, ax=ax[1])
ax[1].set_title("Non-Linear Risk Structure: S&P 500 Sensitivity to High Yield Bonds (Class 13)", fontweight='bold')
ax[1].set_ylabel("Beta Coefficient")
ax[1].axhline(0, color='black', linewidth=1)

plt.tight_layout()
plt.show()

# %%
