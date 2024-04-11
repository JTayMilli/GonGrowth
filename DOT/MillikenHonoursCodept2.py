# %%

# import required module
import os
import pandas as pd
import numpy as np

import country_converter as coco
cc = coco.CountryConverter()

# %%
##Add GDP per capita before removing years
GDPpc_Data = (pd.read_excel("../mpd2020.xlsx", sheet_name = "Full data")
                .drop(columns = ["countrycode"])
                .rename(columns = {"gdppc" : "GDPPC", "country" : "Country", "year" : "Year"})
                .replace({'United States' : 'USA'}))
#GDPpc_Data.head()
GDPpc_Data['GDP'] = GDPpc_Data['GDPPC'] * GDPpc_Data['pop']
GDPpc_Data = GDPpc_Data.replace(0, np.nan).dropna()
GDPpc_Data['lnGDP'] = np.log(GDPpc_Data['GDP'])
GDPpc_Data['dlnGDP'] = GDPpc_Data.groupby('Country', group_keys=False)['lnGDP'].diff(1)
#GDPpc_Data["Country"] = cc.convert(names = GDPpc_Data['Country'], to = "name_short")
GDPpc_Data

# %%
GDPpc_Data = GDPpc_Data[['Year', 'Country', 'GDP', 'dlnGDP']].dropna()

# %%
TWWI = pd.DataFrame(columns=['Country', 'Year', 'TWWI', 'dpTWWI'])

# assign directory
directory = 'DOTdata'
 
# iterate over files in
# that directory
for filename in os.listdir(directory):
#for i in range(1,187):
    #f = f"DOTdata\Exports_to_Count ({i}).xlsx"
    f = os.path.join(directory, filename)
    # checking if it is a file
    if os.path.isfile(f):
        
        #---------------------------------------------------------------------------------------------
        Coun = ( pd.read_excel(f, skiprows = 3, nrows = 0, usecols = 'B')
                .columns[0] ) #gets country name only
        Coun = cc.convert(names = Coun, to = "name_short")

        temp = ( pd.read_excel(f, skiprows = 6)
                .drop(columns = ["Unnamed: 0"])
                .rename(columns= {'Unnamed: 1': 'Country'})
                .fillna(0) ) #gets data

        #Coun
        temp = temp[temp['Country']!= "Aruba, Kingdom of the Netherlands"]
        temp = temp[temp['Country']!= "Curaçao, Kingdom of the Netherlands"] #countries that don't matter
        temp = temp[temp['Country']!= "Sint Maarten, Kingdom of the Netherlands"]

        temp.Country = cc.convert(names = temp['Country'], to = "name_short")
        temp = temp[temp["Country"] != "not found"]
        temp = temp.melt(id_vars='Country', var_name="Year", value_name="ExpRec")
        temp.Year = temp.Year.astype(int)
        temp.ExpRec = temp.apply(lambda x: float(x['ExpRec'].split(" ")[0].replace(',','')) if type(x['ExpRec']) == str else x['ExpRec'], axis=1) ## 1.2 e issue solving

        CounGDP = GDPpc_Data[GDPpc_Data.Country == Coun]
        temp2 = temp.merge(CounGDP[['Year', 'GDP']], on ='Year', how = 'left').rename(columns = {'GDP': 'DomGDP'})
        temp2['w'] = temp2.apply(lambda x: np.nan if ( np.isnan(x['DomGDP']) or x['DomGDP'] == 0 ) else x['ExpRec'] / x['DomGDP'], axis=1)  ##calculate w per year # * 1e6 ??
        temp2 = temp2.sort_values(by = ["Country", "Year"])

        TW = ( temp2.groupby('Country')[["Year","w"]]
            .rolling(5, min_periods=5, on='Year', closed = "left")
            .mean().reset_index().drop(columns = "level_1")
            .rename(columns = {'w':'TW'}).fillna(0) ) ##calculating TW

        temp3 = temp2.merge(TW, on= ['Year','Country'], how = "outer").merge(GDPpc_Data[["Country", "Year", "GDP", "dlnGDP"]], on = ["Country", "Year"], how = "left")
        temp3['TWi'] = temp3.apply(lambda x: x['TW'] * x['GDP'], axis = 1)
        temp3['dpTWi'] = temp3.apply(lambda x: x['TW'] * x['dlnGDP'], axis = 1)
        #temp3.tail(50)

        temp4 = temp3.groupby('Year')[['TWi', 'dpTWi']].sum().reset_index().rename(columns = {"TWi": 'TWWI', "dpTWi": 'dpTWWI'})
        temp4['Country'] = Coun
        TWWI = pd.concat([TWWI, temp4])
#==============================================================================
temp3

# %%
TWWI.to_csv("TWWI.csv", index = False)


