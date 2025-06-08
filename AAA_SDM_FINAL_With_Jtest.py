import sqlite3
import pandas as pd
import numpy as np
from libpysal.weights.util import full2W
from scipy import sparse
from spreg import GM_Combo
from scipy.stats import norm
import os

# === 1. Load data ===
conn = sqlite3.connect("eu_trade.db")
panel_df = pd.read_sql_query("SELECT * FROM panel_data", conn)
w_df = pd.read_sql_query("SELECT * FROM W_Average", conn)
conn.close()

# === 2. Prepare spatial weights matrix ===
w_df.set_index('index', inplace=True)
w_df = w_df.T
unique_countries = panel_df['Country'].unique()
w_df = w_df.loc[unique_countries, unique_countries]
W_matrix = w_df.values
w = full2W(W_matrix)
w.transform = 'r'

# === 3. Panel sorting and lag ===
panel_df['time'] = pd.to_datetime(panel_df['time'])
panel_df.sort_values(by=['Country', 'time'], inplace=True)

# === 4. Create lags ===
panel_df['y_lag'] = panel_df.groupby('Country')['y'].shift(1)
panel_df['x1_lag'] = panel_df.groupby('Country')['x1'].shift(1)  # <-- NEW
panel_df['x2_lag'] = panel_df.groupby('Country')['x2'].shift(1)  # <-- NEW
panel_df.dropna(subset=['y_lag', 'x1_lag', 'x2_lag'], inplace=True)

# === 5. Create spatial lags (within-year) ===
panel_df['W_y'] = 0.0
panel_df['W_x1_lag'] = 0.0
panel_df['W_x2_lag'] = 0.0

for t in panel_df['time'].unique():
    temp = panel_df[panel_df['time'] == t]
    idx = temp.index
    temp_sorted = temp.set_index('Country').loc[unique_countries]

    panel_df.loc[idx, 'W_y'] = w.sparse @ temp_sorted['y'].values
    panel_df.loc[idx, 'W_x1_lag'] = w.sparse @ temp_sorted['x1_lag'].values  # <-- NEW
    panel_df.loc[idx, 'W_x2_lag'] = w.sparse @ temp_sorted['x2_lag'].values  # <-- NEW

# === 6. Two-way demeaning function ===
def two_way_demean(df, entity_col, time_col, var_cols):
    overall_mean = df[var_cols].mean()
    entity_means = df.groupby(entity_col)[var_cols].transform('mean')
    time_means = df.groupby(time_col)[var_cols].transform('mean')
    return df[var_cols] - entity_means - time_means + overall_mean

# === 7. Apply two-way demeaning ===
vars_to_demean = ['y', 'y_lag', 'x1_lag', 'x2_lag', 'W_y', 'W_x1_lag', 'W_x2_lag']
demeaned_vars = two_way_demean(panel_df, 'Country', 'time', vars_to_demean)
panel_df[demeaned_vars.columns] = demeaned_vars

# === 8. Prepare matrices ===
X = panel_df[['y_lag', 'x1_lag', 'x2_lag', 'W_x1_lag', 'W_x2_lag']].values.astype(float)
y = panel_df['y'].values.reshape(-1, 1)

n_regions = len(unique_countries)
n_obs = len(panel_df)
T = n_obs // n_regions
W_big = sparse.kron(sparse.eye(T), w.sparse, format='csr')

# === 9. Estimate model ===
model = GM_Combo(
    y=y, 
    x=X, 
    w=W_big,
    name_y='y',
    name_x=['y_lag', 'x1_lag', 'x2_lag', 'W_x1_lag', 'W_x2_lag']
)

# === 9b. Hansen J-test for overidentification ===
# spreg.GM_Combo has j_statistic attribute with j value and degrees of freedom

from scipy.stats import chi2
import numpy as np


n = model.u.shape[0]
q = model.z.shape[1]
k = model.betas.shape[0]

g = (model.z.T @ model.u) / n

# Approximate weighting matrix as identity
W = np.eye(q)

J_stat = n * (g.T @ W @ g).item()
df = q - k

p_value = 1 - chi2.cdf(J_stat, df)

print(f"\n--- Hansen J-Test (Overidentification Test) ---")
print(f"J-statistic (approx): {J_stat:.4f}")
print(f"Degrees of freedom: {df}")
print(f"P-value (approx): {p_value:.4f}")

if p_value < 0.05:
    print("Reject the null hypothesis: instruments may be invalid (model misspecified).")
else:
    print("Fail to reject the null: instruments are valid.")

from esda.moran import Moran
import numpy as np

moran_stats = []

for t in panel_df['time'].unique():
    positions = np.where(panel_df['time'] == t)[0]
    res_t = model.u[positions].flatten()
    mi = Moran(res_t, w)
    moran_stats.append({'time': t, 'I': mi.I, 'p_sim': mi.p_sim})

moran_df = pd.DataFrame(moran_stats)
print("\n--- Moran's I Test on Residuals by Time Period ---")
print(moran_df)


# Optionally, print average Moran's I and proportion significant:
avg_I = moran_df['I'].mean()
prop_sig = (moran_df['p_sim'] < 0.05).mean()

print(f"\nAverage Moran's I across periods: {avg_I:.4f}")
print(f"Proportion of periods with significant spatial autocorrelation: {prop_sig:.2%}")


# === 10. Pseudo log-likelihood, AIC, BIC ===
def pseudo_loglik(model, k=None):
    residuals = model.u
    n = residuals.shape[0]
    sigma2 = (residuals ** 2).mean()
    pll = -0.5 * n * (np.log(2 * np.pi * sigma2) + 1)
    if k is None:
        k = len(model.betas)
    aic = -2 * pll + 2 * k
    bic = -2 * pll + k * np.log(n)
    return pll, aic, bic

pseudo_ll, pseudo_aic, pseudo_bic = pseudo_loglik(model)
print(f"\n--- Model Fit Statistics ---")
print(f"Pseudo Log-Likelihood: {pseudo_ll:.2f}")
print(f"Pseudo AIC: {pseudo_aic:.2f}")
print(f"Pseudo BIC: {pseudo_bic:.2f}")

# === 11. Output coefficient table ===
n_vars = 5
coefs = model.betas[:n_vars].flatten()
se = np.sqrt(np.diag(model.vm)[:n_vars])
t_stats = coefs / se
p_values = 2 * (1 - norm.cdf(np.abs(t_stats)))

def significance_stars(p):
    if p < 0.01: return '***'
    elif p < 0.05: return '**'
    elif p < 0.1: return '*'
    else: return ''

stars = [significance_stars(p) for p in p_values]

coef_table = pd.DataFrame({
    'Variable': ['y_lag', 'x1_lag', 'x2_lag', 'W_x1_lag', 'W_x2_lag'],
    'Coefficient': [f"{c:.4f}{s}" for c, s in zip(coefs, stars)],
    'Std. Error': [f"{e:.4f}" for e in se],
    't-Statistic': [f"{t:.2f}" for t in t_stats],
    'p-Value': [f"{p:.4f}" for p in p_values]
})

# Export LaTeX
model_name_short = "SDM_Lagged_X"
latex_file = f"{model_name_short}_coefs.tex"
with open(latex_file, "w") as f:
    f.write(coef_table.to_latex(index=False, caption="Results", label="tab:sdm_lagged", escape=False))

# === 12. Summary stats ===
summary = pd.DataFrame([{
    "Model": model.name_h,
    "LogLik": round(pseudo_ll, 2),
    "AIC": round(pseudo_aic, 2),
    "BIC": round(pseudo_bic, 2)
}])
summary_file = "model_comparison_summary.csv"
summary.to_csv(summary_file, mode='a', index=False, header=not os.path.exists(summary_file))

# === 13. Print summary ===
print(model.summary)

# === 14. Calculate Direct, Indirect, and Total Effects for SDM ===

import numpy as np

# Extract parameters
# The order of variables is ['y_lag', 'x1_lag', 'x2_lag', 'W_x1_lag', 'W_x2_lag']
beta = model.betas.flatten()

# Spatial autoregressive parameter (rho)
rho = beta[0]

# Coefficients on direct covariates (non-spatial)
beta_x = beta[1:3]       # [x1_lag, x2_lag]

# Coefficients on spatially lagged covariates
theta_x = beta[3:5]      # [W_x1_lag, W_x2_lag]

# Identity matrix of size n_regions
I_n = np.eye(n_regions)

# Spatial weights matrix (numpy array)
W = W_matrix

# Compute the spatial multiplier matrix: (I - rho * W)^(-1)
A_inv = np.linalg.inv(I_n - rho * W)

# Initialize dictionaries to hold effects
direct_effects = {}
indirect_effects = {}
total_effects = {}

variables = ['x1_lag', 'x2_lag']

for i, var in enumerate(variables):
    # Total effect matrix for variable i
    S = A_inv @ (beta_x[i] * I_n + theta_x[i] * W)
    
    # Direct effect = average of diagonal elements (mean direct impact)
    direct = np.mean(np.diag(S))
    
    # Indirect effect = average of row sums minus diagonal (spillover impact)
    indirect = np.mean(np.sum(S, axis=1) - np.diag(S))
    
    # Total effect = direct + indirect
    total = direct + indirect
    
    direct_effects[var] = direct
    indirect_effects[var] = indirect
    total_effects[var] = total

# Print results
print("\n--- Spatial Durbin Model Effects ---")
print(f"{'Variable':<10} {'Direct':>10} {'Indirect':>10} {'Total':>10}")
for var in variables:
    print(f"{var:<10} {direct_effects[var]:10.4f} {indirect_effects[var]:10.4f} {total_effects[var]:10.4f}")
