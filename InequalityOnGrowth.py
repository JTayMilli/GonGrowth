# %%
#Long Run affect of Income Inequality on Economic Growth
import pandas as pd

import numpy as np
from numpy import arange
from sklearn.linear_model import LinearRegression

# Import matplotlib and seaborn libraries to visualize the data
import matplotlib.pyplot as plt 
import seaborn as sns


from sklearn import datasets, linear_model
import statsmodels.api as sm
from scipy import stats
import pyreadr
from mpl_toolkits.mplot3d import Axes3D

import country_converter as coco
cc = coco.CountryConverter()

# %%
CountryList = ['Albania', 'Algeria', 'Argentina', 'Armenia', 'Australia',
       'Austria', 'Azerbaijan', 'Bangladesh', 'Barbados', 'Belarus',
       'Benin', 'Bosnia and Herzegovina', 'Brazil', 'Bulgaria',
       'Burkina Faso', 'Burundi', 'Cambodia', 'Cameroon', 'Canada',
       'Central African Republic', 'Chile', 'Colombia',
       'Comoros', 'Costa Rica', 'Croatia', 'Cyprus', 'Denmark',
       'Djibouti', 'Dominica', 'Dominican Republic', 'Ecuador',
       'El Salvador', 'Estonia', 'Ethiopia', 'France', 'Gabon', 'Georgia',
       'Germany', 'Ghana', 'Greece', 'Guatemala', 'Guinea',
       'Guinea-Bissau', 'Honduras', 'Hungary', 'Iceland', 'India',
       'Kenya', 'Kuwait', 'Latvia', 'Lebanon', 'Lesotho', 'Lithuania',
       'Luxembourg', 'Madagascar', 'Malawi', 'Malaysia', 'Mali', 'Malta',
       'Mauritania', 'Mauritius', 'Mexico', 'Mongolia', 'Montenegro',
       'Morocco', 'Mozambique', 'Nepal', 'Netherlands', 'New Zealand',
       'Nicaragua', 'Nigeria', 'Norway', 'Oman', 'Pakistan', 'Panama',
       'Paraguay', 'Peru', 'Philippines', 'Poland', 'Portugal', 'Qatar',
       'Romania', 'Rwanda', 'Saudi Arabia', 'Serbia', 'Singapore',
       'Tajikistan', 'Thailand', 'Togo', 'Trinidad and Tobago', 'Tunisia',
       'Turkmenistan', 'Uganda', 'Ukraine', 'United Arab Emirates',
       'Uruguay', 'Zambia', 'Zimbabwe'] #China, United States, Russia

# %%
TWWI = pd.read_csv("DOT/TWWI.csv").replace(0, np.nan).dropna()
TWWI = TWWI.merge(( TWWI.groupby('Country')[["Year","TWWI"]]
                   .rolling(5, min_periods=1, on='Year', closed = "right")
                   .mean().reset_index().drop(columns = "level_1")
                   .rename(columns = {'TWWI':'TWWIavg'}) ), on = ["Country", "Year"], how = "left")#.fillna(0) 
TWWI = TWWI.merge(( TWWI.groupby('Country')[["Year","dpTWWI"]]
                   .rolling(5, min_periods=1, on='Year', closed = "right")
                   .mean().reset_index().drop(columns = "level_1")
                   .rename(columns = {'dpTWWI':'dpTWWIavg'}) ), on = ["Country", "Year"], how = "left")#.fillna(0) 
TWWI = TWWI.drop_duplicates(subset = ['Country', 'Year'], keep = 'last')
TWWI = TWWI.reset_index(drop=True)
####TRADE WEIGHTED WORLD INCOME!!!!!

# %%
#TWWI[TWWI["Country"] == 'Germany']

# %%
tfp = pd.read_csv("tfp-at-constant-national-prices-20111.csv")
tfp = tfp.rename(columns = {"Entity" : "Country"})
tfp = tfp.replace({'United States' : 'USA'}).rename(columns = {"Total factor productivity index (using national accounts) (2017=1)" : "TFP"})
tfp["TFP"] = tfp["TFP"]*100

# %%
tfp["Country"] = cc.convert(names = tfp['Country'], to = "name_short")

# %%
pwt = pd.read_excel("pwt100.xlsx", sheet_name = "Data")[["country", "year", "ctfp", "cwtfp", "rtfpna", "rwtfpna"]]

#ctfp	TFP level at current PPPs (USA=1)
#cwtfp	Welfare-relevant TFP levels at current PPPs (USA=1)
#rtfpna	TFP at constant national prices (2017=1)
#rwtfpna	Welfare-relevant TFP at constant national prices (2017=1)

# %%
pwt = (pwt
    .rename(columns = {"country" : "Country", "year" : "Year", "rtfpna" : "TFP", "ctfp" : "CTFP", "cwtfp" : "CWTFP", "rwtfpna" : "WTFP"})
    .replace({'United States' : 'USA'}))

pwt[["CTFP", "CWTFP", "TFP", "WTFP"]]= pwt[["CTFP", "CWTFP", "TFP", "WTFP"]] * 100

##Removing USA as value always == 1
v = pwt[pwt["Country"] == "USA"]
v[["ctfp", "cwtfp"]] = np.nan
pwt[pwt["Country"] == "USA"] = v
pwt["Country"] = cc.convert(names = pwt['Country'], to = "name_short")
pwt[pwt["Country"] == "USA"].head(10)

# %%
WID_Data = pd.read_excel("WID_Data.xlsx", sheet_name = "Data")
WID_Data[WID_Data['Year'] == 2015].head(20)

# %%
WID_G = WID_Data.iloc[:, 0:250]
WID_G = WID_G[WID_G.Percentile == 'pall'].drop(['Percentile'], axis = 1)

# %%
WID_G1 = WID_G
WID_G1 = WID_G1.melt(id_vars=["Year"], 
        var_name="Country", 
        value_name="Gini")

# %%
WID_G2 = WID_G1
WID_G2['Country'] = WID_G2['Country'].str.split('\n').str[3]
WID_G2 = WID_G2.replace({'Russian Federation' : 'Russia'})
WID_G2['Gini'] = WID_G2["Gini"]########################################################

# %%
WID_G2['Gini'].describe()

# %%
#tfp_gini = pd.concat([tfp, WID_G2], axis=1, join="inner")
WID_G2["Country"] = cc.convert(names = WID_G2['Country'], to = "name_short")
tfp_gini = pwt.merge(WID_G2, on = ['Country', 'Year'], how="outer")
tfp_gini = tfp_gini[tfp_gini["Country"] != "South Africa"] ## Why is SA in the dataset twice

# %%
swiid = pd.read_csv("swiid9_5\swiid9_5_summary.csv").replace(0,np.nan)

# %%
swiid = swiid.rename(columns = {'country' : 'Country', 'year': 'Year'})
swiid['Year'] = swiid['Year'].astype(int)
swiid.abs_red = swiid.gini_mkt - swiid.gini_disp
tfp_gini = tfp_gini.merge(swiid, on=['Country', 'Year'], how='outer')
tfp_gini

# %%
### Choosing Measure of gini
tfp_gini['WIID'] = tfp_gini['Gini']
tfp_gini['Gini'] = tfp_gini['gini_mkt']

# %%
"""
## IV
TGall['Gini'] = TGall['Gini'].replace(0, np.nan)
X = TGall[["lnGDPPC", "Gini"]].dropna(axis=0)
model = linear_model.LinearRegression()
reg = model.fit(X[['lnGDPPC']], X['Gini'])
print(model.coef_[0])
"""

# %%
# Normalize Gini Coefficient
#tfp_gini["NGini"] = np.nan
tfp_gini["-Gini"] = np.nan
tfp_gini["dGini"] = np.nan
tfp_gini["d5Gini"] = np.nan
tfp_gini["dpGini"] = np.nan
tfp_gini["d5pGini"] = np.nan
tfp_gini["dNGini"] = np.nan
tfp_gini["dpGinim5"] = np.nan
tfp_gini["Ginim5"] = np.nan

"""
# Find the changes tfp and gini
tfp_gini["dTFP"] = np.nan
tfp_gini["dpTFP"] = np.nan
tfp_gini["dpTFPm1"] = np.nan
tfp_gini["dpTFPm2"] = np.nan
tfp_gini["dpTFPavg"] = np.nan
tfp_gini["-dpTFP^2"] = np.nan
tfp_gini["dpTFP^2"] = np.nan
tfp_gini["dpTFP^3"] = np.nan
tfp_gini["log(dpTFP)"] = np.nan

tfp_gini["dCTFP"] = np.nan
tfp_gini["dpCTFP"] = np.nan
tfp_gini["dCTFPm1"] = np.nan
tfp_gini["dCTFPm2"] = np.nan
tfp_gini["dCTFPavg"] = np.nan
tfp_gini["-dpCTFP^2"] = np.nan
tfp_gini["dpCTFP^2"] = np.nan
tfp_gini["dpCTFP^3"] = np.nan

tfp_gini["dWTFP"] = np.nan
tfp_gini["dWTFPm1"] = np.nan
tfp_gini["dWTFPm2"] = np.nan
tfp_gini["dWTFPavg"] = np.nan
tfp_gini["-dWTFP^2"] = np.nan
tfp_gini["dWTFP^2"] = np.nan
tfp_gini["dWTFP^3"] = np.nan
"""


for x in tfp_gini["Country"].unique():
    temp = tfp_gini[tfp_gini["Country"] == x]

    #b = float(temp.loc[temp["Year"] == 2017, "Gini"])
    #temp["NGini"] = temp["Gini"].div(b) * 100

    temp["-Gini"] = temp["Gini"] - temp["Gini"].min()

    temp.dGini = temp['Gini'].diff(periods=1)
    temp.d5Gini = temp['Gini'].diff(periods=5)
    temp.dpGini = temp['dGini'].div(temp['Gini']) * 100
    temp.d5pGini = temp['d5Gini'].div(temp['Gini']) * 100
    #temp.dNGini = temp['NGini'].diff(periods=1)
    temp["dpGinim5"] = temp.dpGini.shift(+5)
    temp.Ginim5 = temp.Gini.shift(+5)

    """
    temp["dTFP"] = temp['TFP'].diff(periods=1)
    temp["dpTFP"] = temp['dTFP'].div(temp['TFP'])*100
    temp["dpTFPm1"] = temp.dpTFP.shift(+1)
    temp["dpTFPm2"] = temp.dpTFPm1.shift(+1)
    temp["dpTFPavg"] = (temp["dpTFP"] + temp["dpTFPm1"] + temp["dpTFPm2"]).div(3)
    temp["-dpTFP^2"] = temp.dTFP*abs(temp.dTFP)
    temp["dpTFP^2"] = temp.dpTFP ** 2
    temp["dpTFP^3"] = temp.dpTFP ** 3
    temp["log(dpTFP)"] = np.log(temp["dpTFP"])

    temp["dCTFP"] = temp['CTFP'].diff(periods=1)
    temp["dpCTFP"] = temp['dCTFP'].div(temp['CTFP'])*100
    temp["dCTFPm1"] = temp.dCTFP.shift(+1)
    temp["dCTFPm2"] = temp.dCTFPm1.shift(+1)
    temp["dCTFPavg"] = (temp["dCTFP"] + temp["dCTFPm1"] + temp["dCTFPm2"]).div(3)
    temp["-dpCTFP^2"] = temp.dCTFP*abs(temp.dCTFP)
    temp["dpCTFP^2"] = temp.dpCTFP ** 2 
    temp["dpCTFP^3"] = temp.dpCTFP ** 3

    temp["dWTFP"] = temp['WTFP'].diff(periods=1)
    temp["dWTFPm1"] = temp.dWTFP.shift(+1)
    temp["dWTFPm2"] = temp.dWTFPm1.shift(+1)
    temp["dWTFPavg"] = (temp["dWTFP"] + temp["dWTFPm1"] + temp["dWTFPm2"]).div(3)
    temp["-dWTFP^2"] = temp.dWTFP*abs(temp.dWTFP)
    temp["dWTFP^2"] = temp.dWTFP ** 2
    temp["dWTFP^3"] = temp.dWTFP ** 3
    """
    
    tfp_gini[tfp_gini["Country"] == x] = temp

# %%
##Add GDP per capita before removing years
GDPpc_Data = (pd.read_excel("mpd2020.xlsx", sheet_name = "Full data")
                .drop(columns = ["countrycode"])
                .rename(columns = {"gdppc" : "GDPPC", "country" : "Country", "year" : "Year"})
                .replace({'United States' : 'USA'}))
GDPpc_Data['GDP'] = GDPpc_Data['GDPPC'] * GDPpc_Data['pop']

##ONLY if using relative GDPPC###

GDPpc_Data["Country"] = cc.convert(names = GDPpc_Data['Country'], to = "name_short")

GDPpc_Data = GDPpc_Data[GDPpc_Data.Country.isin(CountryList)]
GDPpc_Data['RelGDPPC']  = GDPpc_Data['GDPPC'] / GDPpc_Data.groupby(['Year'])['GDPPC'].transform('max')

TGall = tfp_gini.merge(GDPpc_Data, on = ['Country', 'Year'], how = 'outer')
TGall = TGall[TGall.GDPPC != 0]

TGall["dGDPPC"] = np.nan
TGall["d5GDPPC"] = np.nan
TGall["dpGDPPC"] = np.nan
TGall["d5pGDPPC"] = np.nan
TGall["dpGDPPC^2"] = np.nan
TGall["lnGDPPC"] = np.nan

##Differnece in GDPpc_Data
for x in TGall["Country"].unique():
    temp = TGall[TGall["Country"] == x]

    temp["dGDPPC"] = temp['GDPPC'].diff(periods=1)
    temp["d5GDPPC"] = temp['GDPPC'].diff(periods=5)
    #temp["dpGDPPC"] = temp["dGDPPC"].div(temp['GDPPC'])*100
    #temp["d5pGDPPC"] = temp["d5GDPPC"].div(temp['GDPPC'])*100
    temp["dpGDPPC^2"] = temp["dpGDPPC"] ** 2
    temp["lnGDPPC"] = np.log(temp["GDPPC"])
    temp["dpGDPPC"] = temp['GDPPC'].diff(periods=1)
    temp["d5pGDPPC"] = temp['lnGDPPC'].diff(periods=5)

    TGall[TGall["Country"] == x] = temp

# %%
TGall = TGall[TGall.Country.isin(CountryList)]
TGall['RellnGDPPC']  = TGall ['lnGDPPC']/ TGall.groupby(['Year'])['lnGDPPC'].transform('max')
TGall['RellnGDPPCm5'] = TGall.groupby('Country')['RellnGDPPC'].shift(5)
TGall['GDPPCm5'] = TGall.groupby('Country')['GDPPC'].shift(5)

# %%
TGall = TGall.sort_values(by = ["Country", "Year"])

# %%
TGall["Gini2m5"] = TGall["Ginim5"]**2
TGall["lnGG"] = TGall["lnGDPPC"].mul(TGall['Gini'])
TGall["lnGGm5"] = TGall.groupby('Country')["lnGG"].shift(5)
TGall["lnGDPPCm5"] = TGall.groupby('Country')["lnGDPPC"].shift(5)

TGall['RellnGGm5'] = TGall['RellnGDPPCm5'] * TGall["Ginim5"]
TGall['RellnGG2m5'] = TGall['RellnGDPPCm5'] * TGall["Gini2m5"]
TGall = TGall.replace([np.nan, -np.inf], 0)

# %%
TGall['Gini'] = TGall['Gini'].replace(0, np.nan)

# %%
TGall = TGall.merge(TWWI, on = ["Country", "Year"], how = "left")

# %% [markdown]
# https://ourworldindata.org/taxation 

# %%
tmitr = (pd.read_csv("top-income-tax-rates-piketty.csv")
        .rename(columns = {"Entity" : "Country", "Top marginal income tax rate (WIR (2018))": "TMITR"})
        .drop(columns = "Code")
        .replace({'United States' : 'USA'}))
tmitr["Country"] = cc.convert(names = tmitr['Country'], to = "name_short")
TGall = TGall.merge(tmitr, on = ['Country', 'Year'], how = 'left')

# %%
oil = (pd.read_csv("API_TX.VAL.FUEL.ZS.UN_DS2_en_csv_v2_5996827.csv") #World Bank, % merchendise oil
        #.rename(columns = {"Entity" : "Country", "Top marginal income tax rate (WIR (2018))": "TMITR"})
        #.drop(columns = "Code")
        .replace({'United States' : 'USA', 'Country Name': 'Country'}))
oil = oil.drop(['Indicator Code', 'Indicator Name', 'Country Code'], axis = 1)
oil = pd.melt(oil, id_vars=['Country Name'], value_vars= oil.columns[1:64])
oil = oil.rename(columns={"Country Name": "Country", "variable": "Year", "value": "Oil"})
oil['Year'] = oil['Year'].astype(int)
oil['Oil'] = ( oil.groupby('Country', group_keys=False)['Oil']
                         .apply(lambda x: x.interpolate(method = 'linear', limit_area='inside')) ) # Interpolation of missing values in oil, linear

oil = oil.merge(GDPpc_Data, how = 'left', on = ['Country', 'Year'])

oil['OdivGDP'] = oil.Oil.div(oil.GDP) #oil exports as % gdp

OilW = ( oil.groupby('Country')[["Year",'OdivGDP']]
        .rolling(5, min_periods=1, on='Year', closed = "left")
        .mean().reset_index().drop(columns = "level_1")
        .rename(columns = {'OdivGDP':'OilW'}).fillna(0) ) ##calculating oil averge over previous 5 years
oil = oil.merge(OilW, on= ['Year','Country'], how = "outer")

oil["Country"] = cc.convert(names = oil['Country'], to = "name_short")

TGall = TGall.merge(oil, on = ['Country', 'Year'], how = 'left' )

# %%
TGall = TGall.sort_values(by = ["Country", "Year"])

# %%
Pr = ( pd.read_excel("statistic_id262858_opec-oil-price-annually-1960-2024.xlsx", sheet_name="Data", skiprows = 4)
      .drop(columns = "Unnamed: 0")
      .rename(columns = {"Unnamed: 1": "Year", "Unnamed: 2": "Price"}) )
Pr.drop(Pr.tail(1).index,inplace=True)
Pr['Year'] = Pr['Year'].astype(int)
Pr['lnP'] = np.log(Pr['Price'])
Pr['dlnP'] = Pr["lnP"].diff(1) ##calculating oil over previous 5 years
TGall = TGall.merge(Pr, on = ['Year'], how = 'left')
TGall['OilP'] = TGall['Oil'].mul(TGall['Price'])
TGall['OlnP'] = TGall['Oil'].mul(TGall['lnP'])

# %%
Pr

# %%
TGall = TGall[TGall.Country !=  'not found']

# %%
save = TGall

# %%
TGall = save

# %%
TGall['OilWP'] = TGall['OilW'] * TGall['dlnP']

OPS = ( TGall[["Year", 'Country', 'OilWP']].groupby('Country')[["Year",'OilWP']]
        .rolling(5, min_periods=1, on='Year', closed = "right")
        .mean().reset_index().drop(columns = "level_1")
        .rename(columns = {'OilWP':'OPS'}) ) ##calculating oil averge over previous 5 years
OPS = OPS[OPS.Country != 'not found']
OPS
TGall = TGall.merge(OPS, on= ['Year','Country'], how = "left")

# %%
TGall["lnGDPPC2"] = TGall["lnGDPPC"] ** 2
TGall['lnGDPPC2m5'] = TGall['lnGDPPCm5'] ** 2

# %%
## IV
TGall = TGall[TGall.Year >= 1960] # remove years before 1960
TGall = TGall[TGall.Year <= 2018]
TGall['Gini'] = TGall['Gini'].replace(0, np.nan)
X = TGall[["Year", "Country", "dpTWWI", "TWWI", "Gini", "OilWP", "lnGDPPC"]].replace(0, np.nan).dropna(axis=0)
X = X.groupby('Country').filter(lambda x: len(x) >= 5) #remove countries with less then 4 data points
X = X.groupby('Year').filter(lambda x: len(x) >= 10) #remove years with less then 10 entries
X = X[X.Country.isin(CountryList)] #make sure that the country is in our list of countries
X = X[["dpTWWI", "TWWI", "Gini", "OilWP", "lnGDPPC"]]
X = X[(np.abs(stats.zscore(X)) < 3).all(axis=1)]
X['B0'] = 1
model = linear_model.LinearRegression(fit_intercept=False)
reg = model.fit(X[["dpTWWI", 'TWWI', 'OilWP', 'lnGDPPC', 'B0']], X['Gini'])
print(model.coef_)
## the beta on gdppc is 100 times higher with the other variables included
AdjGDP = model.coef_[3]
#TGall['predIVGini'] = reg.predict(TGall['lnGDPPC'].values.reshape(-1, 1))
TGall['IVGini'] = TGall['Gini'] - (TGall['lnGDPPC'].mul(AdjGDP))
#TGall['IVGini'] = TGall['Gini'] - TGall['predIVGini']

TGall['IVGini'] = TGall['IVGini'].replace(0, np.nan)
TGall['IVGinim5'] = TGall.groupby('Country')["IVGini"].shift(5)

TGall["IVlnGG"] = TGall["lnGDPPC"].mul(TGall['IVGini'])
TGall["IVlnGGm5"] = TGall.groupby('Country')["IVlnGG"].shift(5)

TGall['IVGini2m5'] = TGall['IVGinim5'].mul(TGall['IVGinim5'])
TGall['IVlnGG2m5'] = TGall['IVGini2m5'].mul(TGall['lnGDPPCm5'])

TGall['IVGini3m5'] = TGall['IVGinim5'].mul(TGall['IVGinim5']).mul(TGall['IVGinim5'])
TGall['IVlnGG3m5'] = TGall['IVGini2m5'].mul(TGall['lnGDPPCm5']).mul(TGall['IVGinim5'])

TGall['IVRellnGGm5'] = TGall['RellnGDPPCm5'] * TGall["IVGinim5"]
TGall['IVRellnGG2m5'] = TGall['RellnGDPPCm5'] * TGall["IVGini2m5"]
TGall = TGall.rename(columns = {"RelGDPPC_x": "RelGDPPC"})
TGall['RelGDPPCm5'] = TGall.groupby('Country')['RelGDPPC'].shift(5)
TGall['IVRelGGm5'] = TGall['RelGDPPCm5'] * TGall["IVGinim5"]
TGall['IVRelGG2m5'] = TGall['RelGDPPCm5'] * TGall["IVGini2m5"]

#################### GDPPC 2
"""
TGall["IVlnGG2"] = TGall["lnGDPPC2"].mul(TGall['IVGini'])
TGall["IVlnG2Gm5"] = TGall["IVlnGG2"].shift(5)

TGall['IVlnG2G2m5'] = TGall['IVGini2m5'].mul(TGall['lnGDPPC2m5'])

TGall['IVlnG2G3m5'] = TGall['IVGini2m5'].mul(TGall['lnGDPPC2m5']).mul(TGall['IVGinim5'])
"""

# %%
### This is just a proof
"""
TGall['Gini'] = tfp_gini['Gini'].replace(0, np.nan)
temp = TGall[["lnGDPPC", "Gini", "TWWI"]].dropna(axis=0)
a = np.cov([temp['TWWI'], temp['Gini']])[0][1] / np.cov([temp['lnGDPPC'], temp['TWWI']])[0][1]
temp['Z'] = temp['Gini'] - (a * temp['lnGDPPC'])
b = np.cov([temp['Z'], temp['lnGDPPC']])[0][1] / np.cov([temp['Gini'], temp['Z']])[0][1]
b
"""

# %%
hdiData  = (pd.read_csv("HDItimeseries.csv")
            .iloc[:, 0:34]
            .drop(columns=['hdi_rank_2021']))
hdiData = pd.melt(hdiData, id_vars=["iso3", "country", "hdicode", "region"], var_name="Year", value_name="HDI")
hdiData.Year = hdiData.Year.str.split("_").str[1].astype(int)
hdiData["country"] = cc.convert(names = hdiData['country'], to = "name_short")
#hdiData["iso3"] = cc.convert(names = hdiData['iso3'], to = "name_short")
hdiData = hdiData.dropna().rename(columns={"country":"Country"})
hdiData

# %%
GNId = pd.read_excel("GNIclean.xlsx").drop(columns = ["Indicator Name", "Indicator Code"]).rename(columns = {"Country Name": "Country"})
GNId = pd.melt(GNId, id_vars=["Country", "Country Code"], var_name="Year", value_name="GNI")
GNId["Country"] = cc.convert(names = GNId['Country'], to = "name_short")
GNId = GNId.replace("not found",np.nan).dropna()
GNId.Year = GNId.Year.astype(int)
GNId

# %%
TGall.replace("not found",np.nan).dropna()
TGall = TGall.merge(hdiData, on = ['Country', 'Year'], how = 'left')
TGall.HDI = TGall.HDI.mul(100)
TGall = TGall.merge(GNId, on = ['Country', 'Year'], how = "left")

# %%
TGall["lnHDI"] = np.log(TGall["HDI"])
TGall["lnHDIm5"] = TGall.groupby('Country')["lnHDI"].shift(5)
TGall["GHDIm5"] = TGall["lnHDIm5"].mul(TGall["Ginim5"])
TGall["GHDI2m5"] = TGall["lnHDIm5"].mul(TGall["Gini2m5"])

# %% [markdown]
# https://ourworldindata.org/taxation
# 
# https://stats.oecd.org/viewhtml.aspx?datasetcode=TABLE_I7&lang=en 

# %%
TMITR = (pd.read_csv("TMITR.csv")
        .drop(columns = ["COU", "YEA", "Unit Code", "Unit", "PowerCode Code", "PowerCode", "Reference Period Code", "Reference Period", "Flag Codes", "Flags"]))
TMITR = (TMITR[TMITR["TAX"] == "TOP_TRATE"]
        .drop(columns = ["TAX", "Income Tax"])
        .rename(columns = {"Value" : "TMITR2"})
        .replace({'United States' : 'USA'}))
TMITR["Country"] = cc.convert(names = TMITR['Country'], to = "name_short")
TGall = TGall.merge(TMITR, on = ['Country', 'Year'], how = 'left')

# %%
nData = ( pd.read_csv("API_SP.POP.GROW_DS2_en_csv_v2_5995052.csv")
         .drop(["Country Code", "Indicator Name", "Indicator Code"], axis = 1)
         .rename(columns={"Country Name": "Country"})
)
nData = pd.melt(nData, id_vars=["Country"], var_name="Year", value_name="n")
nData.Year = nData.Year.astype(int)

# %%
nData["Country"] = cc.convert(names = nData['Country'], to = "name_short")
nData = nData.replace("not found",np.nan).dropna()
TGall = TGall.merge(nData, on=["Country", "Year"], how="left")

# %%
EXP = pd.read_excel("ExportClean.xlsx")
EXP["Country"] = cc.convert(names = EXP['Country'], to = "name_short")
EXP = EXP.melt(id_vars = ["Country"], var_name= "Year", value_name="Exports")
EXP

# %% [markdown]
# Do a test where we show that linear is not good

# %%
#pd.get_dummies(TGall["Country"])

# %%
TGall.Price.isna().sum() / len(TGall.Price)

# %%
## State Fixed Effects
#TGallFE = pd.concat([TGall, pd.get_dummies(TGall["Country"])], axis = 1)

# %%
#TGallFE.iloc[:,TGallFE.columns.get_loc("Albania"):].tail()

# %%
TGall = TGall[TGall.Country !=  'not found']

# %%
#intensity level 2 only , https://ucdp.uu.se/downloads/index.html#armedconflict
Conf = pd.read_excel("Dyadic_v23_1.xlsx")[['location', 'year', 'intensity_level']].rename(columns  = {'year': "Year", "location": "Country"})
Conf = Conf[Conf['intensity_level'] == 2].drop(columns  = "intensity_level")
Conf['Country'] = Conf['Country'].str.split(',')
Conf = Conf.explode('Country')
Conf.Country = cc.convert(names = Conf['Country'], to = "name_short")
Conf = Conf[Conf.Country !=  'not found']
Conf['Conflict'] = 1
#Conf['Adj5']  = 
Conf = Conf.sort_values(by = ["Country", "Year"])
#Conf['Adj5'] = Conf.groupby('Country')['Conflict'].shift(5)
Conf

# %%

TGall = TGall.merge(Conf, on = ["Year", "Country"], how = "left")

# %%
### Remove all years with conflict and 5 years before or after
TGall.Conflict = TGall.Conflict.fillna(0)
TGall["Con5"] = (TGall.groupby('Country')['Conflict']
                  .transform(lambda x: x.shift(5))
                  .fillna(0)
                  .astype(int))
TGall["Con-5"] = (TGall.groupby('Country')['Conflict']
                  .transform(lambda x: x.shift(-5))
                  .fillna(0)
                  .astype(int))
TGall = TGall[TGall['Conflict'] == 0]
TGall = TGall[TGall['Con5'] == 0]
TGall = TGall[TGall['Con-5'] == 0]
TGall = TGall.drop(columns = ["Conflict", "Con5", "Con-5"])

# %%
"""
def cd(country = "Canada"):
    return tfp_gini[tfp_gini['Country'] == country]
"""

# %%
"""
def all(country = "Canada"):
        return TGall[TGall['Country'] == country]
        """

# %%
def TGC(vars = ["TFP", "Gini", "-Gini", "dpTFP", "dpGini", "dNGini"], FE = False, Year = False):
    TGallD = TGall[vars]
    if Year == True:
        TGallD = pd.concat([TGall["Year"], TGallD], axis = 1)
    if FE == True:
        TGallD = pd.concat([TGallD, pd.get_dummies(TGall["Country"])], axis = 1)
    #if d == True:
    #    TGallD = pd.concat([TGall["dpGDPPC"], TGallD], axis = 1)
    #if 'dGini' in TGallD.columns:
    #    TGallD = pd.concat([TGallD[TGallD["dGini"] < -.01], TGallD[TGallD["dGini"] > .01]])
    TGallD = TGallD.replace(0, np.nan).dropna(axis = 0)
    #TGallD = TGallD[(np.abs(stats.zscore(TGallD[vars])) < 3).all(axis=1)]
    return TGallD

# %%
def TGCc(vars = ["TFP", "Gini", "NGini", "-Gini", "dTFP", "dGini", "dNGini", "dTFPavg"], country = "Canada"):
    TGallD = TGall[TGall['Country'] == country]
    TGallD = TGallD[vars].dropna()
    #if 'dGini' in TGallD.columns:
    #    TGallD = pd.concat([TGallD[TGallD["dGini"] < -.01], TGallD[TGallD["dGini"] > .01]])
    TGallD = TGallD[(np.abs(stats.zscore(TGallD)) < 3).all(axis=1)]
    return TGallD

# %%
TGall = TGall[TGall.Year >  1959]

# %%
# TGCc(["CTFP", "TFP", "dCTFP", "dTFP", "dpCTFP", "dpTFP"], country = "Canada").tail(15).to_excel("dTFP_Canada.xlsx")

# %%
print(len(WID_G2))
#Need to compare country names
print(len(tfp_gini))
print(len(swiid))

# %%
TGall

# %%
temp = TGall[TGall.Country == ""]
temp = temp[['IVGinim5', "lnGDPPCm5", "lnGDPPC"]].replace(0,np.nan).dropna()
x = temp['IVGinim5']
y = temp['lnGDPPCm5']
z = temp['lnGDPPC'] - temp['lnGDPPCm5']

fig = plt.figure(figsize = (10,10))
ax = plt.axes(projection='3d')
ax.grid()

ax.scatter(x, y, z, c = 'b')
ax.set_title('% Change GDPpc')

# Set axes label
ax.set_xlabel('Gini', labelpad=20)
ax.set_ylabel('GDPpc', labelpad=20)
ax.set_zlabel('% Change GDPpc', labelpad=20)

plt.show()

sns.lineplot(x = x, y = z)

# %%
X = TGall[["Year", "Country", "IVlnGGm5","IVGinim5", "lnGDPPC", "lnGDPPCm5", 'IVGini2m5', 'IVlnGG2m5', 'IVGini3m5', 'IVlnGG3m5', "n", "OPS", "TWWIavg"]].replace(0,np.nan).dropna()
X = X[(np.abs(stats.zscore(X.drop(columns = ["Year", "Country"]))) < 3).all(axis=1)]
X.to_csv("tansferdata.csv")

# %%
X = TGall[["Year", "Country", "IVlnGGm5","IVGinim5", "lnGDPPC", "lnGDPPCm5", 'IVGini2m5', 'IVlnGG2m5', "n", "OPS", "TWWIavg", "dpTWWIavg"]].replace(0,np.nan).dropna()
X = X[(np.abs(stats.zscore(X.drop(columns = ["Year", "Country"]))) < 3).all(axis=1)]
X.describe()

# %%
X = TGall[["Year", "Country", "IVlnGGm5", "Ginim5", "IVGinim5", "lnGDPPC", "lnGDPPCm5", 'IVGini2m5', 'IVlnGG2m5']].replace(0,np.nan).dropna()
X = X[(np.abs(stats.zscore(X.drop(columns = ["Year", "Country"]))) < 3).all(axis=1)]
X['pdGDPPC'] = X['lnGDPPC'] - X['lnGDPPCm5']
X.describe()

# %%
X = TGall[["Year", "Country", "IVlnGGm5","IVGinim5", "lnGDPPC", "lnGDPPCm5", 'IVGini2m5', 'IVlnGG2m5', "n", "OPS", "TWWIavg", "dpTWWIavg", 'RelGDPPCm5', 'IVRelGGm5', 'IVRelGG2m5', 'd5pGDPPC', "abs_red"]].replace(0,np.nan).dropna()
#X = TGall[["Year", "Country", "IVlnGGm5","IVGinim5", "d5pGDPPC", "lnGDPPCm5", 'IVGini2m5', 'IVlnGG2m5', "n", "OPS", "TWWIavg", "dpTWWIavg"]].replace(0,np.nan).dropna()
X = X[(np.abs(stats.zscore(X.drop(columns = ["Year", "Country"]))) < 3).all(axis=1)]
X = X.groupby('Country').filter(lambda x: len(x) >= 5) #remove countries with less then 4 data points
X = X.groupby('Year').filter(lambda x: len(x) >= 10) #remove years with less then 10 entries
X = X[X.Country.isin(CountryList)] #make sure that the country is in our list of countries
X.describe()

# %%
TGall = TGall.rename(columns = {'GDPPC_y': 'GDPPC'})
TGall.columns

# %%
#X = TGall[["Year", "Country", "IVlnGGm5","IVGinim5", "lnGDPPC", "lnGDPPCm5", 'IVGini2m5', 'IVlnGG2m5', 'IVGini3m5', 'IVlnGG3m5', "IVlnG2Gm5", "IVlnG2G2m5", "IVlnG2G3m5", 'lnGDPPC2m5', "n", "OPS", "TWWIavg"]].replace(0,np.nan).dropna()
#X = TGall[["Year", "Country", "IVlnGGm5","IVGinim5", "d5pGDPPC", "lnGDPPCm5", 'IVGini2m5', 'IVlnGG2m5', 'IVGini3m5', 'IVlnGG3m5', "n", "OPS", "TWWIavg", "dpTWWIavg"]].replace(0,np.nan).dropna()
X = TGall[["Year", "Country", "IVlnGGm5","IVGinim5", "lnGDPPC", "lnGDPPCm5", 'IVGini2m5', 'IVlnGG2m5', "n", "OPS", "TWWIavg", "dpTWWIavg", "abs_red"]].replace(0,np.nan).dropna()
#X = TGall[["Year", "Country", "IVlnGGm5","IVGinim5", "lnGDPPC", "lnGDPPCm5", 'IVGini2m5', 'IVlnGG2m5', "n", "OPS", "TWWIavg", "dpTWWIavg", 'RelGDPPCm5', 'IVRelGGm5', 'IVRelGG2m5']].replace(0,np.nan).dropna()
X = X[(np.abs(stats.zscore(X.drop(columns = ["Year", "Country"]))) < 3).all(axis=1)]
X = X.groupby('Country').filter(lambda x: len(x) >= 5) #remove countries with less then 4 data points
X = X.groupby('Year').filter(lambda x: len(x) >= 10) #remove years with less then 10 entries
X = X[X.Country.isin(CountryList)] #make sure that the country is in our list of countries
X.to_csv("tansferdata.csv")
dummYear = pd.get_dummies(X.Year).fillna(0)
dummCountry = pd.get_dummies(X.Country).fillna(0)
X2 = pd.concat([X.drop(columns = ['Year', 'Country']), dummYear, dummCountry ], axis=1)# dummy years + country
X2.columns = X2.columns.astype(str)
MAINmodel = linear_model.LinearRegression(fit_intercept=False)
X3 = X2.drop(columns = ['lnGDPPC'])
X3["B0"] = 1
reg = MAINmodel.fit(X3, X2['lnGDPPC'])
ission = pd.DataFrame(columns = X3.columns)
ission.loc[len(ission)] = reg.coef_
Arti = pd.DataFrame(columns = X.columns)
Arti['IVGinim5'] = range(1,100) ###This goes first!
Arti['Country'] = 'Gabon'
Arti['Year'] = 1990
Arti['lnGDPPCm5'] = 7
#Arti['GDPPCm5'] = np.exp(Arti['lnGDPPCm5'][0])
#Arti['lnGDPPC2m5'] = Arti['lnGDPPCm5'] ** 2
Arti['n'] = X.n.mean()
Arti['OPS'] = X.OPS.mean()
Arti['abs_red'] = X.abs_red.mean()
Arti['TWWIavg'] = X.TWWIavg.mean()
Arti['dpTWWIavg'] = X.dpTWWIavg.mean()
Arti['IVGini2m5'] = Arti['IVGinim5'] ** 2
#Arti['IVGini3m5'] = Arti['IVGinim5'] ** 3
Arti['IVlnGGm5'] = Arti['IVGinim5'] * Arti['lnGDPPCm5']
Arti['IVlnGG2m5'] = Arti['IVGini2m5'] * Arti['lnGDPPCm5']
#Arti['IVlnGG3m5'] = Arti['IVGini3m5'] * Arti['lnGDPPCm5']
"""
Arti['RelGDPPCm5'] = Arti['GDPPCm5'][0] / TGall[TGall.Year == Arti.Year[0]].GDPPCm5.max()
Arti['IVRelGGm5'] = Arti['RelGDPPCm5'] * Arti['IVGinim5']
Arti['IVRelGG2m5'] = Arti['RelGDPPCm5'] * Arti['IVGini2m5']
Arti['RellnGDPPCm5'] = Arti['lnGDPPCm5'][0] / X[X.Year == Arti.Year[0] - 5].lnGDPPC.max()
Arti['IVRellnGGm5'] = Arti['RellnGDPPCm5'] * Arti['IVGinim5']
Arti['IVRellnGG2m5'] = Arti['RellnGDPPCm5'] * Arti['IVGini2m5']
Arti['IVlnG2Gm5'] = Arti['IVGinim5'] * Arti['lnGDPPC2m5']
Arti['IVlnG2G2m5'] = Arti['IVGini2m5'] * Arti['lnGDPPC2m5']
Arti['IVlnG2G3m5'] = Arti['IVGini3m5'] * Arti['lnGDPPC2m5']
"""
Arti['pred'] = ission.B0[0] + ission[str(Arti['Year'][0])][0] + ission[Arti['Country'][0]][0] #str(Arti['Year'][0]) gets fixed effect for that year
for column in Arti.drop(columns = ["Year", "Country", "lnGDPPC", "pred"]): #, 'GDPPCm5'
    i = column
    Arti['pred'] = Arti['pred'] + (Arti[i]  * ission[i][0])
    
Arti['pred'] = Arti['pred'] - Arti['lnGDPPCm5']
Arti['pred'] = (Arti['pred']) * 100
Arti['pred'] = Arti['pred']# - Arti['pred'][int(X.IVGinim5.mean())] # make mean = 0
"""
Arti['pred'] = ( 5.319 + (Arti['IVGinim5'] * -2.107e-01) 
                #- meandelta
                #+ (Arti['Year'] * 3.901e-01)
                + (Arti['IVGini2m5'] * 3.420e-03)
                - (Arti['lnGDPPCm5'] * 3.430e-01) + (Arti['IVlnGGm5'] * 2.509e-02)
                + (Arti['IVlnGG2m5'] * -3.962e-04) + (Arti['n'] * 4.158e-03)
                + (Arti['OPS'] * 4.390e+03) + (Arti['TWWIavg'] * -5.826e-15) )
"""
ax = sns.lineplot(data=Arti[['IVGinim5', 'pred']], x = 'IVGinim5', y = 'pred', color = "Black")
ax.set_xticks(range(0,101, 2))
#ax.set_yticks(range(-20, 111, 10))
ax.axvline(x = X.IVGinim5.min(),    # Line on x = 2
        ymin = 100, # Bottom of the plot
        ymax = -100,
        color = 'red') # Top of the plot
ax.axvline(x = X.IVGinim5.max(),    # Line on x = 2
        ymin = 100, # Bottom of the plot
        ymax = -100,
        color = 'red') # Top of the plot
ax.axvline(x = X.IVGinim5.mean(),    # Line on x = 2
        ymin = 100, # Bottom of the plot
        ymax = -100,
        color = 'purple') # Top of the plot
ax.axvline(x = X.IVGinim5.mean() - X.IVGinim5.std(),    # Line on x = 2
        ymin = 100, # Bottom of the plot
        ymax = -100,
        color = 'brown') # Top of the plot
ax.axvline(x = X.IVGinim5.mean() + X.IVGinim5.std(),    # Line on x = 2
        ymin = 100, # Bottom of the plot
        ymax = -100,
        color = 'brown') # Top of the plot
ax.axhline(y = 0, color = "black", linestyle = '--')   # Line on x = 2ymin = 100, # Bottom of the plot ymax = -100, color = 'purple') # Top of the plot
ax.set_ylim(Arti['pred'][int(X.IVGinim5.mean())]-15, Arti['pred'][int(X.IVGinim5.mean())]+15)
ax.set_xlim(X.IVGinim5.min(), X.IVGinim5.max())
ax.set_xlabel("Gini Coefficient")
ax.set_ylabel("% Change GDPPC (5 years)")
ax.set_title("purple: mean(Gini), brown: +- std(Gini)")
plt.grid()
plt.show()
## Create a column that has the max of each year for relative gdppc
##Answers don't make sense with IV that I created
## Answers don't make sense when whne taken to the the third degree

# %%
def mainReg(lnGDPPCm5):
    #X = TGall[["Year", "Country", "IVlnGGm5","IVGinim5", "lnGDPPC", "lnGDPPCm5", 'IVGini2m5', 'IVlnGG2m5', 'IVGini3m5', 'IVlnGG3m5', "IVlnG2Gm5", "IVlnG2G2m5", "IVlnG2G3m5", 'lnGDPPC2m5', "n", "OPS", "TWWIavg"]].replace(0,np.nan).dropna()
    #X = TGall[["Year", "Country", "IVlnGGm5","IVGinim5", "d5pGDPPC", "lnGDPPCm5", 'IVGini2m5', 'IVlnGG2m5', 'IVGini3m5', 'IVlnGG3m5', "n", "OPS", "TWWIavg", "dpTWWIavg"]].replace(0,np.nan).dropna()
    X = TGall[["Year", "Country", "IVlnGGm5","IVGinim5", "lnGDPPC", "lnGDPPCm5", 'IVGini2m5', 'IVlnGG2m5', "n", "OPS", "TWWIavg", "dpTWWIavg", "abs_red"]].replace(0,np.nan).dropna()
    #X = TGall[["Year", "Country", "IVlnGGm5","IVGinim5", "lnGDPPC", "lnGDPPCm5", 'IVGini2m5', 'IVlnGG2m5', "n", "OPS", "TWWIavg", "dpTWWIavg", 'RelGDPPCm5', 'IVRelGGm5', 'IVRelGG2m5']].replace(0,np.nan).dropna()

    X = X[(np.abs(stats.zscore(X.drop(columns = ["Year", "Country"]))) < 3).all(axis=1)]
    X = X.groupby('Country').filter(lambda x: len(x) >= 5) #remove countries with less then 4 data points
    X = X.groupby('Year').filter(lambda x: len(x) >= 10) #remove years with less then 10 entries
    X = X[X.Country.isin(CountryList)] #make sure that the country is in our list of countries
    X.to_csv("tansferdata.csv")
    dummYear = pd.get_dummies(X.Year).fillna(0)
    dummCountry = pd.get_dummies(X.Country).fillna(0)
    X2 = pd.concat([X.drop(columns = ['Year', 'Country']), dummYear, dummCountry ], axis=1)# dummy years + country
    X2.columns = X2.columns.astype(str)
    MAINmodel = linear_model.LinearRegression(fit_intercept=False)
    X3 = X2.drop(columns = ['lnGDPPC'])
    X3["B0"] = 1
    reg = MAINmodel.fit(X3, X2['lnGDPPC'])
    ission = pd.DataFrame(columns = X3.columns)
    ission.loc[len(ission)] = reg.coef_

    Arti = pd.DataFrame(columns = X.columns)

    Arti['IVGinim5'] = range(1,100) ###This goes first!

    Arti['Country'] = 'Gabon'
    Arti['Year'] = 1990
    Arti['lnGDPPCm5'] = lnGDPPCm5
    #Arti['GDPPCm5'] = np.exp(Arti['lnGDPPCm5'][0])

    #Arti['lnGDPPC2m5'] = Arti['lnGDPPCm5'] ** 2
    Arti['n'] = X.n.mean()
    Arti['OPS'] = X.OPS.mean()
    Arti['abs_red'] = X.abs_red.mean()
    Arti['TWWIavg'] = X.TWWIavg.mean()
    Arti['dpTWWIavg'] = X.dpTWWIavg.mean()

    Arti['IVGini2m5'] = Arti['IVGinim5'] ** 2
    #Arti['IVGini3m5'] = Arti['IVGinim5'] ** 3

    Arti['IVlnGGm5'] = Arti['IVGinim5'] * Arti['lnGDPPCm5']
    Arti['IVlnGG2m5'] = Arti['IVGini2m5'] * Arti['lnGDPPCm5']
    #Arti['IVlnGG3m5'] = Arti['IVGini3m5'] * Arti['lnGDPPCm5']

    """
    Arti['RelGDPPCm5'] = Arti['GDPPCm5'][0] / TGall[TGall.Year == Arti.Year[0]].GDPPCm5.max()
    Arti['IVRelGGm5'] = Arti['RelGDPPCm5'] * Arti['IVGinim5']
    Arti['IVRelGG2m5'] = Arti['RelGDPPCm5'] * Arti['IVGini2m5']


    Arti['RellnGDPPCm5'] = Arti['lnGDPPCm5'][0] / X[X.Year == Arti.Year[0] - 5].lnGDPPC.max()
    Arti['IVRellnGGm5'] = Arti['RellnGDPPCm5'] * Arti['IVGinim5']
    Arti['IVRellnGG2m5'] = Arti['RellnGDPPCm5'] * Arti['IVGini2m5']

    Arti['IVlnG2Gm5'] = Arti['IVGinim5'] * Arti['lnGDPPC2m5']
    Arti['IVlnG2G2m5'] = Arti['IVGini2m5'] * Arti['lnGDPPC2m5']
    Arti['IVlnG2G3m5'] = Arti['IVGini3m5'] * Arti['lnGDPPC2m5']
    """

    Arti['pred'] = ission.B0[0] + ission[str(Arti['Year'][0])][0] + ission[Arti['Country'][0]][0] #str(Arti['Year'][0]) gets fixed effect for that year
    for column in Arti.drop(columns = ["Year", "Country", "lnGDPPC", "pred"]): #, 'GDPPCm5'
        i = column
        Arti['pred'] = Arti['pred'] + (Arti[i]  * ission[i][0])
        
    Arti['pred'] = Arti['pred'] - Arti['lnGDPPCm5']
    Arti['pred'] = (Arti['pred']) * 100
    Arti['pred'] = Arti['pred']# - Arti['pred'][int(X.IVGinim5.mean())] # make mean = 0
    """
    Arti['pred'] = ( 5.319 + (Arti['IVGinim5'] * -2.107e-01) 
                    #- meandelta
                    #+ (Arti['Year'] * 3.901e-01)
                    + (Arti['IVGini2m5'] * 3.420e-03)
                    - (Arti['lnGDPPCm5'] * 3.430e-01) + (Arti['IVlnGGm5'] * 2.509e-02)
                    + (Arti['IVlnGG2m5'] * -3.962e-04) + (Arti['n'] * 4.158e-03)
                    + (Arti['OPS'] * 4.390e+03) + (Arti['TWWIavg'] * -5.826e-15) )
    """

    ax = sns.lineplot(data=Arti[['IVGinim5', 'pred']], x = 'IVGinim5', y = 'pred', color = "Black")
    ax.set_xticks(range(0,101, 2))
    #ax.set_yticks(range(-20, 111, 10))
    ax.axvline(x = X.IVGinim5.min(),    # Line on x = 2
            ymin = 100, # Bottom of the plot
            ymax = -100,
            color = 'red') # Top of the plot
    ax.axvline(x = X.IVGinim5.max(),    # Line on x = 2
            ymin = 100, # Bottom of the plot
            ymax = -100,
            color = 'red') # Top of the plot
    ax.axvline(x = X.IVGinim5.mean(),    # Line on x = 2
            ymin = 100, # Bottom of the plot
            ymax = -100,
            color = 'purple') # Top of the plot
    ax.axvline(x = X.IVGinim5.mean() - X.IVGinim5.std(),    # Line on x = 2
            ymin = 100, # Bottom of the plot
            ymax = -100,
            color = 'brown') # Top of the plot
    ax.axvline(x = X.IVGinim5.mean() + X.IVGinim5.std(),    # Line on x = 2
            ymin = 100, # Bottom of the plot
            ymax = -100,
            color = 'brown') # Top of the plot
    ax.axhline(y = 0, color = "black", linestyle = '--')   # Line on x = 2ymin = 100, # Bottom of the plot ymax = -100, color = 'purple') # Top of the plot
    ax.set_ylim(Arti['pred'][int(X.IVGinim5.mean())]-15, Arti['pred'][int(X.IVGinim5.mean())]+15)
    ax.set_xlim(X.IVGinim5.min(), X.IVGinim5.max())
    ax.set_xlabel("Gini Coefficient")
    ax.set_ylabel("% Change GDPPC (5 years)")
    ax.set_title("purple: mean(Gini), brown: +- std(Gini)")
    plt.grid()
    #plt.show()

    ## Create a column that has the max of each year for relative gdppc

    ##Answers don't make sense with IV that I created
    ## Answers don't make sense when whne taken to the the third degree
    return Arti, X

# %%
size = 1
Arti, X = mainReg(6)
predList = [np.array(Arti['pred'])]
for i in range(6+size,12+size,size):
    Arti, X = mainReg(i)
    predList.append(np.array(Arti['pred']))

predList
ax = Axes3D(fig)

x = range(1,99+1)
y = range(1,7+1)

data = predList
#maxT = data.index(max(data))
#maxR = data[maxT].index(max(data[maxT]))
data = np.array(data)
#print(data)

hf = plt.figure()
ha = hf.add_subplot(projection='3d')

X, Y = np.meshgrid(x, y)  # `plot_surface` expects `x` and `y` data to be 2D
ha.plot_surface(X, Y, data)

ha.set_xlabel('Gini')
ha.set_ylabel('ln(GDPpc)')
ha.set_zlabel('% Change GDPpc (5 years)')
ha.set_yticklabels(range(6,12+1))
ha.set_title('% Change GDPpc (5 years)')


# %%
sns.histplot((X[X.Year == 2005]).IVGinim5, binwidth=1)

# %%
Arti

# %%
ission

# %%
len(X.Country.unique())

# %%
X = TGall[["Year", "Country", "GHDIm5","Ginim5", "lnHDI", "lnHDIm5", 'Gini2m5', 'GHDI2m5', "n"]].replace(0,np.nan).dropna()
X.to_csv("huuu.csv")

# %%
X = TGall[["Year", "Country", "gini_mkt", "gini_disp", "lnGDPPC", "n"]].replace(0,np.nan).dropna()
X = X[X["Country"]=="Brazil"]
X[["lnGDPPC", "gini_mkt", "gini_disp", "n"]].describe()

# %%
len(tfp_gini["Country"].unique())

# %%
men = TGall.replace(0,np.nan).describe()

# %%
wiid = TGall.WIID.replace(0,np.nan).mul(100).dropna()
gini_disp = TGall.gini_disp.replace(0,np.nan).dropna()
gini_mkt = TGall.gini_mkt.replace(0,np.nan).dropna()
#plt.style.use('dark_background')

plt.hist(wiid, 100)
plt.hist(gini_mkt, 100)
plt.hist(gini_disp, 100)

plt.xlabel("Gini")

# %%
cd("Serbia").tail()

# %%
tfp_giniCC = tfp_gini.dropna()
CC = TGall.dropna()

# %% [markdown]
# We could build a model for inequality without gdp or income

# %% [markdown]
# # More Variable

# %%
# Using pairplot we'll visualize the data for correlation
#sns.pairplot(tfp_gini, x_vars=['TFP', 'Gini'], 
#            y_vars='Country', size=20, aspect=.5, kind='scatter')
#plt.show()

# %%
sns.lmplot(data=TGall[TGall["Year"] >= 1980], x="Year", y="TFP", line_kws={'color': 'darkmagenta'}, scatter_kws = {'color' : 'mediumaquamarine'})#, size = 10)

# %%
sns.lmplot(data=TGC(["Gini"],Year = True), x="Year", y="Gini", line_kws={'color': 'darkmagenta'}, scatter_kws = {'color' : 'mediumaquamarine'})#, size = 10)

# %%
#sns.lmplot(data=tips, x="total_bill", y="tip", col="time", hue="smoker")
edc = TGall[["lnGDPPC", "Gini", "lnGDPPCm5"]].replace(0,np.nan).dropna(axis = 0)
#edc.tail(50)
edc["pChGDPpc"] = TGall["lnGDPPC"] - TGall["lnGDPPCm5"]
edc = edc[(edc["Gini"] <= -.01) | (edc["Gini"] >= .01)]
sns.lmplot(data=edc, x="Gini", y="pChGDPpc", line_kws={'color': 'green'}, scatter_kws = {'color' : 'indigo'})#, size = 10)

# %%
#sns.lmplot(data=TGC(["dpTFP", "dpGDPPC", "Gini"]), x="dpGDPPC", y="Gini", order = 2, line_kws={'color': 'purple'}, scatter_kws = {'color' : 'darkorange'})#, size = 10)

# %%
#TGC(["dpTFP", "lnGDPPC", "Gini"])

# %%
#sns.lmplot(data=TGC(["dpTFP", "dpGDPPC", "dpGini"]), x="dpGDPPC", y="dpGini", line_kws={'color': 'indigo'}, scatter_kws = {'color' : 'green'})#, size = 10)

# %%
#sns.lmplot(data=TGC(["TFP", "-Gini"]), y="TFP", x="-Gini", order = 2, line_kws={'color': 'firebrick'}, scatter_kws = {'color' : 'darkorange'})#, size = 10)


# %%
#sns.lmplot(data=TGC(["dpTFP", "-Gini"]), x="dpTFP", y="-Gini", order = 2, line_kws={'color': 'orange'}, scatter_kws = {'color' : 'purple'})#, size = 10)

# %%
sns.lmplot(data=TGall[TGall["Year"] == 2001], y="lnGDPPC", x="Gini", order = 2, line_kws={'color': 'orange'}, scatter_kws = {'color' : 'purple'})#, size = 10)

# %%
# sns.lmplot(data=cd(), x="TFP", y="Gini")

# %%
"""
from scipy import stats
penguins = sns.load_dataset("penguins")

sns.jointplot(data=cd(), x="TFP", y="Gini", kind = "reg")
"""

# %% [markdown]
# ## dTFP and dGini
# https://stackoverflow.com/questions/25579227/seaborn-lmplot-with-equation-and-r2-text

# %%
#def r2(x, y):
#    return stats.pearsonr(x, y)[0] ** 2

# r2(CC.dgdppc, CC.dGini)

# %%
#sns.regplot(data=tfp_gini, x="dTFP", y="Gini")

# %%
# mmm = sns.regplot(data=tfp_gini, x="dTFP", y="dGini", order=2)

# %% [markdown]
# # ML Regression

# %%
"""
from ctypes import sizeof


fig=plt.figure(figsize=(15, 15))
ax=fig.add_subplot(111,projection='3d')
n=100
ax.scatter(TGall["Gini"],TGall["dpTFP"],TGall["dpGDPPC"],color="red")
ax.set_xlabel("Gini")
ax.set_ylabel("dpTFP")
ax.set_zlabel("dpGDPPC")
plt.show()
"""

# %%
"""
# Predicting values:
# Function for predicting future values
def get_regression_predictions(input_features,intercept,slope):
    predicted_values = input_features*slope + intercept
    return predicted_values

# Checking various accuracy:
from sklearn.metrics import r2_score 
test_x = np.array(test[['dTFP']]) 
test_y = np.array(test[['dGini']]) 
test_y_ = regr.predict(test_x)
print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Mean sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y)** 2))
print("R2-score: %.2f" % r2_score(test_y_ , test_y) )
"""

# %% [markdown]
# # Machine learning 2
# https://towardsdatascience.com/multivariate-linear-regression-in-python-step-by-step-128c2b127171

# %%
"""
X = tfp_giniCC[["dTFP"]]
y = tfp_giniCC["dGini"]
X.head()
"""

# %%
#Initiate the theta values.
theta = np.array([0]*len(X.columns))

#number of training data
m = len(X)

def hypothesis(theta, X):
    return theta*X

#Define the cost function
def computeCost(X, y, theta):
    y1 = hypothesis(theta, X)
    y1=np.sum(y1, axis=1)
    return sum(np.sqrt((y1-y)**2))/(2*47)

# %%
#Predict Outputs
y_hat = hypothesis(theta, X)
y_hat = np.sum(y_hat, axis=1)
y

# %%
%matplotlib inline
import matplotlib.pyplot as plt
plt.figure()
plt.scatter(x = list(range(0, len(y))), y= y, color='blue')         
plt.scatter(x=list(range(0, len(y))), y=y_hat, color='darkred')
plt.show()

# %% [markdown]
# ## Simple Linear Regression

# %% [markdown]
# https://stackoverflow.com/questions/27928275/find-p-value-significance-in-scikit-learn-linearregression

# %%
X = TGC(["lnGDPPC", "Gini"])
model = LinearRegression()
reg = model.fit(X['lnGDPPC'].values.reshape(-1, 1), X['Gini'].values.reshape(-1, 1))
y_pred = reg.predict(X['lnGDPPC'].values.reshape(-1, 1))
y_pred

# %%
X = TGall[["IVlnGG","IVGinim5", "lnGDPPC", "lnGDPPCm5"]].replace(0,np.nan).dropna()
X2 = sm.add_constant(X.drop(columns = ["lnGDPPC"]))
est = sm.OLS(X.lnGDPPC, X2)
est2 = est.fit()
print(est2.summary())


# %%
X

# %%
X = TGall[["IVlnGG","IVGinim5", "lnGDPPC", "lnGDPPCm5", 'IVGini2m5', 'IVlnGG2m5']].replace(0,np.nan).dropna()
X2 = sm.add_constant(X.drop(columns = ["lnGDPPC"]))
est = sm.OLS(X.lnGDPPC, X2)
est2 = est.fit()
print(est2.summary())

# %%
X = TGall[["Year", "Country", "IVlnGGm5","IVGinim5", "lnGDPPC", "lnGDPPCm5", 'IVGini2m5', 'IVlnGG2m5', "n"]].replace(0,np.nan).dropna()
X = X[X.lnGDPPC > 10]
X.to_csv("tansferdata.csv")

# %%
#( len(TGall['OilP'].dropna()) - len(TGall.Gini.dropna()) )/ len(TGall.Gini.dropna()) # lose 32% of data when OilP is included

# %%
TGC(["IVlnGG"]).mean()

# %%
y = "lnGDPPC"
X = TGC(["Ginim5", "lnGGm5", "lnGDPPCm5", y], Year = True)

X2 = sm.add_constant(X.drop(columns = [y]))
est = sm.OLS(X[y], X2)
est2 = est.fit()
#print(est2.summary())
print(est2.predict())


# %%
y = "Gini"
X = TGC(["lnGDPPC", y])

X2 = sm.add_constant(X.drop(columns = [y]))
est = sm.OLS(X[y], X2)
est2 = est.fit()
#print(est2.summary())
est2.predict([50,70])

# %% [markdown]
# ### Extremly predictive, dTFP on Gini

# %% [markdown]
# run on several years avg

# %% [markdown]
# Taxes, democracy, news freedom as control variable

# %%
#TGall.tail().drop(columns = ["Year", "Country", ])
TGall["lnGDPPC"].mean()

# %%
#Average standard deviation per gini coefficient
std = 0
for x in tfp_gini["Country"].unique():
    temp = TGCc(["Gini"], country = x)
    std += temp["Gini"].std()

print(std / tfp_gini["Country"].unique().size)

# %%
tfp_gini.Gini.std()

# %%
# Avg std for dNGini
std = 0
for x in tfp_gini["Country"].unique():
    temp = TGCc(["dNGini"], country = x)
    std += temp["dNGini"].std()

print(std / tfp_gini["Country"].unique().size)

# %%
"""
X = TGC(["dpTFP", "dpGini", "GDPPC", "dGDPPC", "dpGDPPC"], FE = False, Year = True)
X2 = sm.add_constant(X.drop(columns = ("dpGini")))
est = sm.OLS(X["dpGini"], X2)
est2 = est.fit()
print(est2.summary())
"""

# %% [markdown]
# ## Ridge Regression

# %%

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

# %%
TGall.columns

# %%
rdf = TGall[["Year", "Country", "IVGinim5", "IVGini2m5", "lnGDPPC", "lnGDPPCm5", "abs_red", "n"]].replace(0,np.nan).dropna().reset_index(drop=True)
len(rdf.Country.unique())

# %%
#interaction variables
irdf = rdf.drop(["lnGDPPC", "Year", "Country"], axis = 1)

variables = irdf.columns
df_interaction = irdf
for i in range(len(variables)):
    for j in range(i+1, len(variables)):
        df_interaction[f'{variables[i]}*{variables[j]}'] = irdf[variables[i]] * irdf[variables[j]]
        
irdf = pd.concat([rdf[["Country", "lnGDPPC", "Year"]], df_interaction], axis=1).drop(["IVGinim5*IVGini2m5"], axis = 1)
irdf = pd.get_dummies(irdf, columns=['Country'], prefix='', prefix_sep='')
irdf["Intercept"] = 1
irdf

# %%
scaler = StandardScaler()
X = scaler.fit_transform(irdf.drop('lnGDPPC', axis=1))
y = irdf['lnGDPPC'].values

ridge = RidgeCV(alphas=np.logspace(-4, 4, 100), cv=10)
ridge.fit(X, y)

feature_names = list(irdf.drop('lnGDPPC', axis=1).columns)
ridge_coeff_df = pd.DataFrame(ridge.coef_, index=feature_names, columns=['Coefficient'])

ridge_coeff_df.head(30)

# %%
X_df = pd.DataFrame(X, columns = irdf.drop('lnGDPPC', axis=1).columns)
X_df.columns[0:20]

# %%
print(X_df.iloc[irdf[(irdf.Year ==  2018) & (irdf.Canada == 1)].index].values[0][2])
print(scaler.transform(irdf.iloc[irdf[(irdf.Year ==  2018) & (irdf.Canada == 1)].index].drop('lnGDPPC', axis = 1).values)[0][2])
#print(irdf.iloc[irdf[(irdf.Year ==  2018) & (irdf.Canada == 1)].index].values[0])

# %%
prediction = ridge.predict((scaler.transform(irdf.iloc[irdf[(irdf.Year ==  2018) & (irdf.Canada == 1)].index].drop('lnGDPPC', axis = 1).values)).reshape(1, -1))
print(prediction)

# %%
def mlCo(Country):
    df = irdf[(irdf.Year ==  2018) & (irdf[Country] == 1)].drop('lnGDPPC', axis = 1)
    pred = pd.DataFrame(columns=["Gini", "pred"])
    
    ## have to edit all vectors that are multiplied by Gini
    for i in range(1,101):
        g = df.IVGinim5.values[0]
        for r in df.columns:
            if "IVGini2m5" in r:
                df[r] = df[r].div(g**2).mul(i**2)
            elif "IVGinim5" in r:
                df[r] = df[r].div(g).mul(i)
        
        tdf = scaler.transform(df.values.reshape(1, -1))
        pred = pred.append({"Gini":i, "pred": (ridge.predict(tdf.reshape(1, -1))[0] - df.lnGDPPCm5.values[0])}, ignore_index = True)
    return pred
        

# %%
fake = mlCo("Benin")
sns.lineplot(x = "Gini", y = "pred", data = fake)


