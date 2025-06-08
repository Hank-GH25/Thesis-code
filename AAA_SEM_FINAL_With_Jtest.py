import sqlite3
import pandas as pd
import numpy as np
from libpysal.weights.util import full2W
from scipy import sparse
from spreg import GM_Error
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

# === 5. Two-way demeaning function ===
def two_way_demean(df, entity_col, time_col, var_cols):
    overall_mean = df[var_cols].mean()
    entity_means = df.groupby(entity_col)[var_cols].transform('mean')
    time_means = df.groupby(time_col)[var_cols].transform('mean')
    return df[var_cols] - entity_means - time_means + overall_mean

# === 6. Apply two-way demeaning ===
vars_to_demean = ['y', 'y_lag', 'x1_lag', 'x2_lag']
demeaned_vars = two_way_demean(panel_df, 'Country', 'time', vars_to_demean)
panel_df[demeaned_vars.columns] = demeaned_vars

# === 7. Prepare matrices ===
X = panel_df[['y_lag', 'x1_lag', 'x2_lag']].values.astype(float)
y = panel_df['y'].values.reshape(-1, 1)

n_regions = len(unique_countries)
n_obs = len(panel_df)
T = n_obs // n_regions
W_big = sparse.kron(sparse.eye(T), w.sparse, format='csr')

# === 8. Estimate SEM model ===
model = GM_Error(
    y=y, 
    x=X, 
    w=W_big,
    name_y='y',
    name_x=['y_lag', 'x1_lag', 'x2_lag']
)

# === 9. Pseudo log-likelihood, AIC, BIC ===
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

# === 10. Output coefficient table ===
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
model_name_short = "SEM_Lagged_X"
latex_file = f"{model_name_short}_coefs.tex"
with open(latex_file, "w") as f:
    f.write(coef_table.to_latex(index=False, caption="Results", label="tab:sem_lagged", escape=False))

# === 11. Summary stats ===
summary = pd.DataFrame([{
    "Model": model.name_w,
    "LogLik": round(pseudo_ll, 2),
    "AIC": round(pseudo_aic, 2),
    "BIC": round(pseudo_bic, 2)
}])
summary_file = "model_comparison_summary.csv"
summary.to_csv(summary_file, mode='a', index=False, header=not os.path.exists(summary_file))

# === 12. Print summary ===
print(model.summary)

from scipy.stats import chi2

# === 13. Hansen's J-Test (Manual Instrument Construction) ===

# 1. Residuals
u = model.u  # shape (n_obs, 1)
n_obs = u.shape[0]

# 2. Reconstruct instrument matrix Z
# Standard GMM instruments: X and spatially lagged X
WX = W_big @ X
Z = np.hstack((X, WX))  # shape: (n_obs, 2k) if X has k columns

# 3. Compute the weighting matrix (HAC-consistent covariance of moments)
# Use robust 2SLS-style estimator
moment_matrix = Z.T @ Z
W = np.linalg.inv(moment_matrix)

# 4. Compute sample moments
g_bar = Z.T @ u / n_obs  # shape (2k, 1)

# 5. Compute J-statistic
J_stat = float(n_obs * (g_bar.T @ W @ g_bar))

# 6. Degrees of freedom = #instruments - #parameters
df = Z.shape[1] - model.k  # model.k = number of regressors

# 7. P-value from chi-square distribution
if df > 0:
    p_value = 1 - chi2.cdf(J_stat, df)
else:
    p_value = np.nan  # Not valid if exactly identified

print(f"\n--- Hansen's J-Test ---")
print(f"J-statistic: {J_stat:.4f}")
print(f"Degrees of freedom: {df}")
print(f"P-value: {p_value if not np.isnan(p_value) else 'NA (exactly identified)'}")



from esda.moran import Moran

# === 14. Moran's I Test on Residuals ===
# Reshape residuals by time to reconstruct panel (n_regions x T)
resids = model.u.reshape((T, n_regions)).T  # Shape: (regions, time)

# Average residuals across time for simplicity (alternative: loop over time)
avg_resid = resids.mean(axis=1)
moran_test = Moran(avg_resid, w)

print(f"\n--- Moran's I Test on Residuals ---")
print(f"Moran's I: {moran_test.I:.4f}")
print(f"Expected I under null: {moran_test.EI:.4f}")
#z_score = moran_test.z.item() if hasattr(moran_test.z, "item") else moran_test.z[0]
#p_value = moran_test.p_norm.item() if hasattr(moran_test.p_norm, "item") else moran_test.p_norm[0]

#print(f"Z-score: {z_score:.4f}")
#print(f"P-value: {p_value:.4f}")

# === 15. Direct and Indirect Effects for Spatial Error Model ===

# Extract coefficients and lambda (spatial error parameter)
beta = model.betas.flatten()[:n_vars]  # coefficients on regressors
lambda_se = model.lambd[0] if hasattr(model, 'lambd') else None

print("\n--- Spatial Error Model Effects ---")
print(f"Spatial error parameter (lambda): {lambda_se:.4f}" if lambda_se is not None else "Lambda parameter not available.")

variables = ['y_lag', 'x1_lag', 'x2_lag']

print(f"\n{'Variable':<10} {'Direct Effect':>15} {'Indirect Effect':>20}")
for var, coef in zip(variables, beta):
    print(f"{var:<10} {coef:15.4f} {'N/A (SEM no indirect effects)':>20}")

