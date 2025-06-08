import sqlite3
import pandas as pd
import numpy as np
from libpysal.weights.util import full2W
from scipy import sparse
from spreg import GM_Error_Het  # <- CHANGED
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

# === 3. Panel sorting and lagging ===
panel_df['time'] = pd.to_datetime(panel_df['time'])
panel_df.sort_values(by=['Country', 'time'], inplace=True)

panel_df['y_lag'] = panel_df.groupby('Country')['y'].shift(1)
panel_df['x1_lag'] = panel_df.groupby('Country')['x1'].shift(1)
panel_df['x2_lag'] = panel_df.groupby('Country')['x2'].shift(1)
panel_df.dropna(subset=['y_lag', 'x1_lag', 'x2_lag'], inplace=True)

# === 4. Create spatial lags (for x1_lag and x2_lag only) ===
panel_df['W_x1_lag'] = 0.0
panel_df['W_x2_lag'] = 0.0

for t in panel_df['time'].unique():
    temp = panel_df[panel_df['time'] == t]
    idx = temp.index
    temp_sorted = temp.set_index('Country').loc[unique_countries]

    panel_df.loc[idx, 'W_x1_lag'] = w.sparse @ temp_sorted['x1_lag'].values
    panel_df.loc[idx, 'W_x2_lag'] = w.sparse @ temp_sorted['x2_lag'].values

# === 5. Two-way demeaning ===
def two_way_demean(df, entity_col, time_col, var_cols):
    overall_mean = df[var_cols].mean()
    entity_means = df.groupby(entity_col)[var_cols].transform('mean')
    time_means = df.groupby(time_col)[var_cols].transform('mean')
    return df[var_cols] - entity_means - time_means + overall_mean

vars_to_demean = ['y', 'y_lag', 'x1_lag', 'x2_lag', 'W_x1_lag', 'W_x2_lag']
demeaned_vars = two_way_demean(panel_df, 'Country', 'time', vars_to_demean)
panel_df[demeaned_vars.columns] = demeaned_vars

# === 6. Prepare matrices ===
X = panel_df[['y_lag', 'x1_lag', 'x2_lag', 'W_x1_lag', 'W_x2_lag']].values.astype(float)
y = panel_df['y'].values.reshape(-1, 1)

n_regions = len(unique_countries)
n_obs = len(panel_df)
T = n_obs // n_regions
W_big = sparse.kron(sparse.eye(T), w.sparse, format='csr')

# === 7. Estimate SDEM model ===
model = GM_Error_Het(
    y=y,
    x=X,
    w=W_big,
    name_y='y',
    name_x=['y_lag', 'x1_lag', 'x2_lag', 'W_x1_lag', 'W_x2_lag']
)

# === 8. Fit statistics ===
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

# === 9. Coefficient table ===
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
model_name_short = "SDEM_Lagged_X"
latex_file = f"{model_name_short}_coefs.tex"
with open(latex_file, "w") as f:
    f.write(coef_table.to_latex(index=False, caption="SDEM Results", label="tab:sdem_lagged", escape=False))

# === 10. Model summary ===
summary = pd.DataFrame([{
    "Model": model.name_w,
    "LogLik": round(pseudo_ll, 2),
    "AIC": round(pseudo_aic, 2),
    "BIC": round(pseudo_bic, 2)
}])
summary_file = "model_comparison_summary.csv"
summary.to_csv(summary_file, mode='a', index=False, header=not os.path.exists(summary_file))

# === 11. Print summary ===
print(model.summary)



from sklearn.linear_model import LinearRegression
from scipy.stats import chi2

# === 12. Approximate Hansen's J-Test (Instruments = Regressors) ===

# Assumes no endogenous regressors; using X as instruments
u = model.u.flatten()
Z_manual = X  # Treat regressors as instruments

# Regression: residuals on instruments
reg = LinearRegression(fit_intercept=False).fit(Z_manual, u)
R2 = reg.score(Z_manual, u)

n = X.shape[0]
k = X.shape[1]
df_j = Z_manual.shape[1] - k  # Degrees of freedom = overidentifying restrictions (likely 0)

J_stat = n * R2
p_val = 1 - chi2.cdf(J_stat, df_j) if df_j > 0 else np.nan

print("\n--- Approximate Hansen's J-Test ---")
print(f"J-statistic: {J_stat:.4f}")
print(f"Degrees of freedom: {df_j}")
print(f"p-value: {p_val if not np.isnan(p_val) else 'NA (no overidentifying restrictions)'}")

# === 13. Direct, Indirect, and Total Effects for SDEM ===

# Indices for the variables in the model
# You have: ['y_lag', 'x1_lag', 'x2_lag', 'W_x1_lag', 'W_x2_lag']
# Direct effects: x1_lag (idx 1), x2_lag (idx 2)
# Indirect effects: W_x1_lag (idx 3), W_x2_lag (idx 4)

direct_coefs = coefs[1:3]
indirect_coefs = coefs[3:5]

effects_df = pd.DataFrame({
    'Variable': ['x1_lag', 'x2_lag'],
    'Direct Effect': direct_coefs,
    'Indirect Effect': indirect_coefs,
    'Total Effect': direct_coefs + indirect_coefs
})

print("\n--- Direct, Indirect, and Total Effects (SDEM) ---")
print(effects_df.round(4).to_string(index=False))



from esda.moran import Moran

print("\n--- Moran’s I on Residuals by Time ---")
panel_df['residual'] = model.u

moran_results = []
for t in sorted(panel_df['time'].unique()):
    try:
        sub_df = panel_df[panel_df['time'] == t]
        resids = sub_df.set_index('Country').loc[unique_countries]['residual'].values

        moran = Moran(resids, w)

        moran_results.append({
            'time': t,
            'Moran_I': moran.I,
            'Expected_I': moran.EI,
            'Z': moran.z_norm,
            'p': moran.p_norm
        })

        print(f"{t.date()} | I = {moran.I:.4f}, Z = {moran.z_norm:.2f}, p = {moran.p_norm:.4f}")
    except Exception as e:
        print(f"{t.date()} | Moran's I failed: {e}")

# === Compute averages across time ===
if moran_results:
    moran_df = pd.DataFrame(moran_results)

    avg_I = moran_df['Moran_I'].mean()
    avg_EI = moran_df['Expected_I'].mean()
    avg_Z = moran_df['Z'].mean()
    avg_p = moran_df['p'].mean()

    print("\n--- Average Moran’s I Summary ---")
    print(f"Average I: {avg_I:.4f}")
    print(f"Average Expected I: {avg_EI:.4f}")
    print(f"Average Z-score: {avg_Z:.2f}")
    print(f"Average p-value: {avg_p:.4f}")

    # Optionally export
    # moran_df.to_csv("moran_by_time.csv", index=False)

