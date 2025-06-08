import time
from ecbdata import ecbdata

# Define time period
START_PERIOD = "2014-01"
END_PERIOD = "2024-12"

# Define Austria's country code
country = "Austria"
code = "AT"

# Define the data series keys for HICP, Unemployment, and GDP Growth
series_keys = {
    "HICP": "ICP.M.{country_code}.N.000000.4.INX",
    "Unemployment": "LFSI.M.{country_code}.S.UNEHRT.TOTAL0.15_74.T",
    "GDP_Growth": "MNA.Q.N.{country_code}.W2.S1.S1.B.B1GQ._Z._Z._Z.EUR.V.N",
}
# Loop through each indicator and fetch data
for indicator, key_template in series_keys.items():
    series_key = key_template.format(country_code=code)

    print(f"\n🔍 Fetching {indicator} data for {country}...")
    try:
        df = ecbdata.get_series(series_key, start=START_PERIOD, end=END_PERIOD)

        if df is not None and not df.empty:
            df = df.reset_index()
            
            # Print the column names
            print(f"📊 Column names for {indicator} in {country}: {df.columns.tolist()}")
        else:
            print(f"⚠️ No data available for {indicator} in {country}.")
    except Exception as e:
        print(f"❌ Error fetching {indicator} data for {country}: {e}")

    time.sleep(2)  # Enforce delay to avoid request blocks
