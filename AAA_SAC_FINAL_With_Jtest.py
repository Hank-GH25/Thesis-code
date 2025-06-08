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
panel_df['x1_lag'] = panel_df.groupby('Country')['x1'].shift(1)
panel_df['x2_lag'] = panel_df.groupby('Country')['x2'].shift(1)
panel_df.dropna(subset=['y_lag', 'x1_lag', 'x2_lag'], inplace=True)

# === 5. Create spatial lag of y ===
panel_df['W_y'] = 0.0
for t in panel_df['time'].unique():
    temp = panel_df[panel_df['time'] == t]
    idx = temp.index
    temp_sorted = temp.set_index('Country').loc[unique_countries]
    panel_df.loc[idx, 'W_y'] = w.sparse @ temp_sorted['y'].values

# === 6. Two-way demeaning function ===
def two_way_demean(df, entity_col, time_col, var_cols):
    overall_mean = df[var_cols].mean()
    entity_means = df.groupby(entity_col)[var_cols].transform('mean')
    time_means = df.groupby(time_col)[var_cols].transform('mean')
    return df[var_cols] - entity_means - time_means + overall_mean

# === 7. Apply two-way demeaning ===
vars_to_demean = ['y', 'y_lag', 'x1_lag', 'x2_lag', 'W_y']
demeaned_vars = two_way_demean(panel_df, 'Country', 'time', vars_to_demean)
panel_df[demeaned_vars.columns] = demeaned_vars

# === 8. Prepare matrices ===
X = panel_df[['y_lag', 'x1_lag', 'x2_lag']].values.astype(float)
y = panel_df['y'].values.reshape(-1, 1)

n_regions = len(unique_countries)
n_obs = len(panel_df)
T = n_obs // n_regions
W_big = sparse.kron(sparse.eye(T), w.sparse, format='csr')

# === 9. Estimate SAC model ===
model = GM_Combo(
    y=y, 
    x=X, 
    w=W_big,
    name_y='y',
    name_x=['y_lag', 'x1_lag', 'x2_lag']
)

# === J-Test (Hansen) for Overidentifying Restrictions ===
from scipy.stats import chi2

# Residuals and instruments
residuals = model.u  # (n x 1)
Z = model.z  # (n x m)
n = Z.shape[0]
k = X.shape[1]
m = Z.shape[1]

# Compute the optimal GMM weighting matrix manually
Zu = Z * residuals  # Element-wise multiply each column of Z by residuals
S = (Zu.T @ Zu) / n  # Covariance matrix of moment conditions
W_opt = np.linalg.inv(S)

# J-statistic
moment = Z.T @ residuals  # Shape: (m, 1)
J_stat = float(moment.T @ W_opt @ moment)
df = m - k
J_pval = 1 - chi2.cdf(J_stat, df)

print("\n--- Hansen J-Test of Overidentifying Restrictions ---")
print(f"J-statistic: {J_stat:.4f}")
print(f"Degrees of freedom: {df}")
print(f"p-value: {J_pval:.4f}")



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
n_vars = 3
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
    'Variable': ['y_lag', 'x1_lag', 'x2_lag'],
    'Coefficient': [f"{c:.4f}{s}" for c, s in zip(coefs, stars)],
    'Std. Error': [f"{e:.4f}" for e in se],
    't-Statistic': [f"{t:.2f}" for t in t_stats],
    'p-Value': [f"{p:.4f}" for p in p_values]
})

# Export LaTeX
model_name_short = "SAC_Model"
latex_file = f"{model_name_short}_coefs.tex"
with open(latex_file, "w") as f:
    f.write(coef_table.to_latex(index=False, caption="Results", label="tab:sac_model", escape=False))

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


# === 14. Moran's I (Panel-Averaged) on Residuals ===
from esda.moran import Moran

# Add residuals to the panel data
panel_df['residuals'] = model.u.flatten()

# Store Moran's I statistics
morans_i_values = []
p_values = []

for t in panel_df['time'].unique():
    temp = panel_df[panel_df['time'] == t]
    
    # Make sure data is sorted by 'Country' to align with W
    temp_sorted = temp.set_index('Country').loc[unique_countries]
    residuals_t = temp_sorted['residuals'].values

    # Compute Moran's I for this time point
    moran = Moran(residuals_t, w)
    morans_i_values.append(moran.I)
    p_values.append(moran.p_norm)

# Average Moran's I and p-value
average_moran_i = np.mean(morans_i_values)
average_p_value = np.mean(p_values)

print("\n--- Average Moran's I on Residuals (Panel) ---")
print(f"Average Moran's I: {average_moran_i:.4f}")
print(f"Average p-value (normal approximation): {average_p_value:.4f}")

# === 15. Compute Direct and Indirect Effects for SAC Model ===

import scipy.sparse.linalg as splinalg

# Extract spatial lag parameter (rho) from model.betas (last two are rho and lam)
rho = model.betas[-2][0]

# Extract spatial weights matrix for one cross-section (w.sparse is n_regions x n_regions)
W_sparse = w.sparse

n = W_sparse.shape[0]

# Compute spatial multiplier matrix S = (I - rho * W)^-1
I_n = sparse.eye(n)
S = splinalg.inv(I_n - rho * W_sparse).toarray()

# Extract coefficients for regressors (excluding spatial parameters)
beta = coefs  # [y_lag, x1_lag, x2_lag]

# Variables of interest for effects (excluding y_lag)
exog_vars = ['x1_lag', 'x2_lag']
exog_betas = beta[1:]  # coefficients for x1_lag and x2_lag

direct_effects = []
indirect_effects = []
total_effects = []

for i, b in enumerate(exog_betas):
    S_beta = S * b
    direct = np.mean(np.diag(S_beta))
    total = np.mean(S_beta.sum(axis=1))
    indirect = total - direct
    direct_effects.append(direct)
    indirect_effects.append(indirect)
    total_effects.append(total)

effects_table = pd.DataFrame({
    'Variable': exog_vars,
    'Direct Effect': direct_effects,
    'Indirect Effect': indirect_effects,
    'Total Effect': total_effects
})

print("\n--- Direct and Indirect Effects ---")
print(effects_table.round(4).to_string(index=False))
