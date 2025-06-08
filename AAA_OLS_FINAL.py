import sqlite3
import pandas as pd
import numpy as np
import statsmodels.api as sm
import os

# === 1. Load data ===
conn = sqlite3.connect("eu_trade.db")
panel_df = pd.read_sql_query("SELECT * FROM panel_data", conn)
conn.close()

# === 2. Panel sorting and lag creation ===
panel_df['time'] = pd.to_datetime(panel_df['time'])
panel_df.sort_values(by=['Country', 'time'], inplace=True)

panel_df['y_lag'] = panel_df.groupby('Country')['y'].shift(1)
panel_df['x1_lag'] = panel_df.groupby('Country')['x1'].shift(1)
panel_df['x2_lag'] = panel_df.groupby('Country')['x2'].shift(1)

panel_df.dropna(subset=['y_lag', 'x1_lag', 'x2_lag'], inplace=True)

# === 3. Add year column for two-way FE ===
panel_df['year'] = panel_df['time'].dt.year

# === 4. Two-way demeaning function (Country and Year Fixed Effects) ===
def two_way_demean(df, entity_col, time_col, var_cols):
    overall_means = df[var_cols].mean()
    entity_means = df.groupby(entity_col)[var_cols].transform('mean')
    time_means = df.groupby(time_col)[var_cols].transform('mean')
    return df[var_cols] - entity_means - time_means + overall_means

vars_to_demean = ['y', 'y_lag', 'x1_lag', 'x2_lag']
demeaned = two_way_demean(panel_df, 'Country', 'year', vars_to_demean)
panel_df[demeaned.columns] = demeaned

# === 5. Prepare regression matrices ===
X = panel_df[['y_lag', 'x1_lag', 'x2_lag']].values.astype(float)
X = sm.add_constant(X)  # Adds intercept; won't be identified but statsmodels requires it
y = panel_df['y'].values

# === 6. Estimate OLS ===
ols_model = sm.OLS(y, X).fit()

# === 7. Model fit statistics ===
n = len(y)
k = X.shape[1]
sigma2 = np.mean(ols_model.resid ** 2)
loglik = -0.5 * n * (np.log(2 * np.pi * sigma2) + 1)
aic = -2 * loglik + 2 * k
bic = -2 * loglik + k * np.log(n)

print(f"\n--- Model Fit Statistics ---")
print(f"Log-Likelihood: {loglik:.2f}")
print(f"AIC: {aic:.2f}")
print(f"BIC: {bic:.2f}")

# === 8. Coefficient Table ===
coef_names = ['const', 'y_lag', 'x1_lag', 'x2_lag']
coefs = ols_model.params
se = ols_model.bse
t_stats = ols_model.tvalues
p_values = ols_model.pvalues

def significance_stars(p):
    if p < 0.01: return '***'
    elif p < 0.05: return '**'
    elif p < 0.1: return '*'
    else: return ''

stars = [significance_stars(p) for p in p_values]

coef_table = pd.DataFrame({
    'Variable': coef_names,
    'Coefficient': [f"{c:.4f}{s}" for c, s in zip(coefs, stars)],
    'Std. Error': [f"{e:.4f}" for e in se],
    't-Statistic': [f"{t:.2f}" for t in t_stats],
    'p-Value': [f"{p:.4f}" for p in p_values]
})

# === 9. Export LaTeX ===
model_name_short = "OLS_TwoWayDemeaned"
latex_file = f"{model_name_short}_coefs.tex"
with open(latex_file, "w") as f:
    f.write(coef_table.to_latex(index=False, caption="Two-Way Fixed Effects Regression Results", label="tab:ols_two_way", escape=False))

# === 10. Save summary ===
summary = pd.DataFrame([{
    "Model": "OLS_TwoWayFE",
    "LogLik": round(loglik, 2),
    "AIC": round(aic, 2),
    "BIC": round(bic, 2)
}])
summary_file = "model_comparison_summary.csv"
summary.to_csv(summary_file, mode='a', index=False, header=not os.path.exists(summary_file))

# === 11. Print summary ===
print(ols_model.summary())
