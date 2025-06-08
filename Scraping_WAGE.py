import pandas as pd
import time
from ecbdata import ecbdata

# Define time period
START_PERIOD = "2014-01"
END_PERIOD = "2024-12"

# Define the countries and their respective ISO 3166-1 alpha-2 country codes
countries = {
    'Austria': 'AT', 'Belgium': 'BE', 'Bulgaria': 'BG', 'Cyprus': 'CY',
    'Czechia': 'CZ', 'Germany': 'DE', 'Denmark': 'DK', 'Estonia': 'EE',
    'Spain': 'ES', 'Finland': 'FI', 'France': 'FR',
    'Greece': 'GR', 'Croatia': 'HR', 'Hungary': 'HU', 'Ireland': 'IE',
    'Italy': 'IT', 'Lithuania': 'LT', 'Luxembourg': 'LU', 'Latvia': 'LV',
    'Malta': 'MT', 'Netherlands': 'NL', 'Poland': 'PL', 'Portugal': 'PT',
    'Romania': 'RO', 'Sweden': 'SE', 'Slovenia': 'SI', 'Slovakia': 'SK'
}

# Define the data series keys for HICP, Unemployment, and GDP Growth
series_keys = {
    #"HICP": "ICP.M.{country_code}.N.000000.4.INX",
    #"Unemployment": "LFSI.M.{country_code}.S.UNEHRT.TOTAL0.15_74.T",
    #"GDP_Growth": "MNA.Q.N.{country_code}.W2.S1.S1.B.B1GQ._Z._Z._Z.EUR.V.N",
    #"M3": "BSI.M.{country_code}.Y.V.M30.X.I.U2.2300.Z01.A",
    #"Wages": "MNA.Q.N.{country_code}.W2.S1.S1.D.D11._Z.BTE._Z.XDC.V.N",
    'Wages': "MNA.Q.N.{country_code}.W2.S1.S1._Z.ULC_PS._Z._T._Z.IX.D.N"
}

# Initialize a list to store data
data_list = []

# Loop through each country and fetch the data
for country, code in countries.items():
    for indicator, key_template in series_keys.items():
        series_key = key_template.format(country_code=code)

        print(f"🔍 Fetching {indicator} data for {country}...")
        try:
            df = ecbdata.get_series(series_key, start=START_PERIOD, end=END_PERIOD)

            if df is not None and not df.empty:
                df = df.reset_index()

                # Select only TIME_PERIOD and the last column (value)
                df = df[['TIME_PERIOD', 'OBS_VALUE']]  
                df.columns = ['TIME_PERIOD', 'OBS_VALUE']

                # Add metadata columns
                df.insert(0, 'Country', country)
                df.insert(1, 'Indicator', indicator)

                data_list.append(df)
            else:
                print(f"⚠️ No data available for {indicator} in {country}.")
        except Exception as e:
            print(f"❌ Error fetching {indicator} data for {country}: {e}")

        time.sleep(2)  # Enforce a 2-second delay to avoid being blocked

# Combine all data into a single DataFrame
if data_list:
    final_df = pd.concat(data_list, ignore_index=True)
    final_df.to_csv('LCI_data_2014_2024.csv', index=False)
    print("✅ Data has been saved to 'LCI_data_2014_2024.csv'.")
else:
    print("❌ No data was retrieved.")
