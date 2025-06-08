import pandas as pd

# Load your data
df = pd.read_csv(r'C:\Users\Henry\Documents\Econ Docs\Thesis\Thesis Code\panel_data.csv')

# Select the variables of interest
variables = ['y', 'x1', 'x2']

# Calculate descriptive statistics
summary_stats = df[variables].agg(
    ['count', 'mean', 'median', 'min', 'max', 'std', 'var']
).T  # Transpose for clarity

# Optional: round for easier reading
summary_stats = summary_stats.round(2)

# Print the summary
print(summary_stats)

# Optionally, export to LaTeX for Overleaf
summary_stats.to_latex('summary_statistics.tex')
