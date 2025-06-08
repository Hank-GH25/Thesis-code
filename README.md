# Thesis-code
Code used in Bachelors Thesis on spatial dependence in inflation within the EU
The files with the prefix AAA and then the model name are the final versions of each model ran.
The scraper files scraped the data from the European Central Bank website.
The transformation and deleting years\renaming columns files were used next to properly format the data, and the iputing file was also used in this step to impute the unit labor cost data to monthly.
The SQL database file was then used to upload all of the gathered and formatted data to an SQL database as seperate tables.
The trade flow data was manually formatted to individual spreadsheets in the folder Country_trade_files, in this repository it is just the csv files that are in the format country_name_tot.csv
Using this folder the Spatial weight matrix creation file was used that saved each matrix as a table in the SQL database, and the W_average file was then used to create one average matrix from these time period specific matrices.
The panel data file was also used to format the seperate SQL tables for each variable into one table in panel data format so that it could be used in the model code.
All the other files were used for intermittent checks to determine whether the code had worked as intended.
