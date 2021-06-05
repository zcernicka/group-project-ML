
import pandas as pd
import numpy as np

class DataCleaner(object):

    def __init__(self):
        pass


''' This script contains classes to clean the whole datasets '''

import pandas as pd 

class DataCleaner(object):
    def __init__ (self):
        pass
    
    def CreateBuurtTable(self,postal_code, buurt, wijk, gem):
        postal_code = postal_code.drop_duplicates(subset=['PC6'], keep='first')
        buurt_df = pd.merge(postal_code, buurt, left_on ='Buurt2018', right_on = 'buurtcode', how = 'left')
        buurt_df = pd.merge(buurt_df, wijk, left_on = 'Wijk2018', right_on = 'GWBcode8', how = 'left')
        buurt_df = pd.merge(buurt_df, gem, left_on = 'Gemeente2018', right_on = 'GWBcode8', how = 'left')
        buurt_df = buurt_df[['PC6', 'buurtcode', 'buurtname', 'GWBlabel_x', 'GWBlabel_y' ]]
        buurt_df['buurtcode'] = buurt_df['buurtcode'].astype(str)
        buurt_df = buurt_df.reset_index()
        buurt_df['postal_code'] = buurt_df['PC6'] 
        buurt_df['buurt']  = buurt_df['buurtname'] 
        buurt_df['wijk'] = buurt_df['GWBlabel_x'] 
        buurt_df['gemeente']  = buurt_df['GWBlabel_y']
        buurt_df = buurt_df.drop(['PC6', 'buurtname', 'index', 'GWBlabel_x', 'GWBlabel_y'], axis = 1)
        return buurt_df

    def MergeHousingBuurt(self, housing_data, postal_code):
        df_merge = pd.merge(housing_data, postal_code, left_on = 'postcode', right_on = 'postal_code', how = 'left')
        #df_merge = df_merge.drop([df_merge['']])
        return df_merge

    def clean_category(self,df):
        df['categorieObject']=df['categorieObject'].replace({'<{':'','}>':''}, regex=True)
        return df

    def clean_year(self, df):
        df['construction_year']=df['bouwjaar'].str.extract(r'([0-9]{4})')
        df = df.drop(['bouwjaar'], axis=1)
        df['house_age'] = 2020 - df['construction_year'].astype(int)
        df['house_age'] = np.where(df['house_age'] < 0, 0, df['house_age'])
        return df

    def clean_house_type(self, df):
        x = df
        df = df["soortWoning"].str.split(">", n=2, expand=True)
        df = df.replace({0: {'<{': '', '}>': '', '\(<{': '', '}>\)': '', '}': '', '\(': '', '\)': ''}}, regex=True)
        df = df.replace({1: {'<{': '', '}>': '', '\(<{': '', '}>\)': '', '}': '', '\(': '', '\)': ''}}, regex=True)
        df = df.replace({2: {'<{': '', '}>': '', '\(<{': '', '}>\)': '', '}': '', '\(': '', '\)': ''}}, regex=True)
        df = df.fillna(value="")
        df = df[0] + "," + df[1] + "," + df[2]
        df = [i[:-1] for i in df]
        newlist = []
        for l in range(len(df)):
            if "," in df[l][-1]:
                newlist.append(df[l][0:-1])
            else:
                newlist.append(df[l])
        x['house_type'] = newlist
        x = x.drop(['soortWoning'], axis=1)
        return x

    def clean_badkamers(self, df):
        median_badkamers = df['aantalBadkamers'].median()
        for row in range(len(df)):
            if df['aantalBadkamers'].isna().iloc[row] == True:
                df['aantalBadkamers'].iloc[row] = median_badkamers
        return df
    
    def clean_living_area(self, df):
        median_house = df['perceelOppervlakte'].median()
        #impute perceelOppervlakte of appartment with oppervlakte
        for row in range(len(df)):
            if df['perceelOppervlakte'].isna().iloc[row] == True and df['categorieObject'].iloc[row] == 'Appartement':
                df['perceelOppervlakte'].iloc[row] = df['oppervlakte'].iloc[row]
            elif df['perceelOppervlakte'].isna().iloc[row] == True and df['categorieObject'].iloc[row] == 'Woonhuis':
                df['perceelOppervlakte'].iloc[row] = median_house
        #clean the outliers in the oppervlakte
        df = df[df['oppervlakte'] != 1]
        return df

    def clean_aantal_badkamers(self, df):
        median_badkamers = df['aantalBadkamers'].median()
        for row in range(len(df)):
            if df['aantalBadkamers'].isna().iloc[row] == True:
                df['aantalBadkamers'].iloc[row] = median_badkamers
        return df

    def add_selling_days(self, df):
        df['selling_days'] = pd.to_datetime(df['datum_ondertekening']) - pd.to_datetime(df['publicatieDatum'])
        df['selling_days'] = df['selling_days'].dt.days.astype('int16')
        df['selling_days'] = np.where(df['selling_days'] < 0, 0, df['selling_days'])
        return df


    def clean_energy_label(self, df):
        df['energielabelKlasse'].fillna('no_label', inplace = True)
        for row in range(len(df)):
            if df['energielabelKlasse'].iloc[row] == 'no_label':
                df['energielabelKlasse'].iloc[row] = df['energielabelKlasse'].iloc[row]
            else:
                df['energielabelKlasse'].iloc[row] = df['energielabelKlasse'].iloc[row][0]

        return df

    def clean_selling_price (self, df):
        df = df.dropna(subset = ['koopPrijs'], axis = 0, inplace = False)
        return df
    
    def clean_cbs_address(self, df):
        df = df.dropna(subset = ['buurt', 'wijk', 'gemeente'])
        return df

    def clean_income(self, df):
        df = df[['NBH_code', '%_low_income_households', '%_below_social_minimum']]
        df = df[df['NBH_code'].str.contains('BU')]
        df['NBH_code'] = df['NBH_code'].astype(str)
        for a in range(len(df)):
            if df['NBH_code'].iloc[a][2:3] == '0' and df['NBH_code'].iloc[a][3:4] != '0':
                df['NBH_code'].iloc[a] = ''.join(df['NBH_code'].iloc[a].rsplit('BU0'))
            elif df['NBH_code'].iloc[a][2:4] == '00' and df['NBH_code'].iloc[a][4:5] != '0':
                df['NBH_code'].iloc[a] = ''.join(df['NBH_code'].iloc[a].rsplit('BU00'))
            elif df['NBH_code'].iloc[a][2:5] == '000':
                df['NBH_code'].iloc[a] = ''.join(df['NBH_code'].iloc[a].rsplit('BU000'))
            else:
                df['NBH_code'].iloc[a] = df['NBH_code'].iloc[a][2:]
        df['%_low_income_households'] = df['%_low_income_households'].str.extract(r'(\d)')
        df['%_below_social_minimum'] = df['%_below_social_minimum'].str.extract(r'(\d)')
        df = df.dropna(subset=['%_low_income_households', '%_below_social_minimum'])
        df['%_low_income_households'] = df['%_low_income_households'].astype(int)
        df['%_below_social_minimum'] = df['%_below_social_minimum'].astype(int)

        return df

    def clean_school(self, df):
        df = df[['WijkenEnBuurten', '0-4age_Binnen5Km_55', '4+age_Binnen10Km_67', '4+age_Binnen10Km_71',
                         '4+age_Binnen10Km_75']]
        df.rename(columns={"WijkenEnBuurten": "NBH_code"})
        df['0-4age_Binnen5Km_55'] = df['0-4age_Binnen5Km_55'].str.extract(r'(\d)')
        df['4+age_Binnen10Km_67'] = df['4+age_Binnen10Km_67'].str.extract(r'(\d)')
        df['4+age_Binnen10Km_71'] = df['4+age_Binnen10Km_71'].str.extract(r'(\d)')
        df['4+age_Binnen10Km_75'] = df['4+age_Binnen10Km_75'].str.extract(r'(\d)')
        df['0-4age_Binnen5Km_55'] = df['0-4age_Binnen5Km_55'].astype(float)
        df['4+age_Binnen10Km_67'] = df['4+age_Binnen10Km_67'].astype(float)
        df['4+age_Binnen10Km_71'] = df['4+age_Binnen10Km_71'].astype(float)
        df['4+age_Binnen10Km_75'] = df['4+age_Binnen10Km_75'].astype(float)
        df.rename(columns={"WijkenEnBuurten": "NBH_code"}, inplace=True)
        df['4+age_Binnen10Km'] = df[['4+age_Binnen10Km_67', '4+age_Binnen10Km_71', '4+age_Binnen10Km_75']].sum(axis=1)
        df = df.drop(['4+age_Binnen10Km_67', '4+age_Binnen10Km_71', '4+age_Binnen10Km_75'], axis=1)
        df = df[df['NBH_code'].str.contains('BU')]
        df['NBH_code'] = df['NBH_code'].astype(str)
        for a in range(len(df)):
            if df['NBH_code'].iloc[a][2:3] == '0' and df['NBH_code'].iloc[a][3:4] != '0':
                df['NBH_code'].iloc[a] = ''.join(df['NBH_code'].iloc[a].rsplit('BU0'))
            elif df['NBH_code'].iloc[a][2:4] == '00' and df['NBH_code'].iloc[a][4:5] != '0':
                df['NBH_code'].iloc[a] = ''.join(df['NBH_code'].iloc[a].rsplit('BU00'))
            elif df['NBH_code'].iloc[a][2:5] == '000':
                df['NBH_code'].iloc[a] = ''.join(df['NBH_code'].iloc[a].rsplit('BU000'))
            else:
                df['NBH_code'].iloc[a] = df['NBH_code'].iloc[a][2:]
        df = df.dropna(subset=['0-4age_Binnen5Km_55', '4+age_Binnen10Km'])
        return df

    def clean_density(self, df):
        df = df[['NBH_code', 'No_residents', 'Density']]
        df = df[df['NBH_code'].str.contains('BU')]
        df['NBH_code'] = df['NBH_code'].astype(str)
        for a in range(len(df)):
            if df['NBH_code'].iloc[a][2:3] == '0' and df['NBH_code'].iloc[a][3:4] != '0':
                df['NBH_code'].iloc[a] = ''.join(df['NBH_code'].iloc[a].rsplit('BU0'))
            elif df['NBH_code'].iloc[a][2:4] == '00' and df['NBH_code'].iloc[a][4:5] != '0':
                df['NBH_code'].iloc[a] = ''.join(df['NBH_code'].iloc[a].rsplit('BU00'))
            elif df['NBH_code'].iloc[a][2:5] == '000':
                df['NBH_code'].iloc[a] = ''.join(df['NBH_code'].iloc[a].rsplit('BU000'))
            else:
                df['NBH_code'].iloc[a] = df['NBH_code'].iloc[a][2:]
        df['Density'] = df['Density'].str.extract(r'(\d+)')
        df = df.dropna(subset=['No_residents', 'Density'])
        df['No_residents'] = df['No_residents'].astype(int)
        df['Density'] = df['Density'].astype(int)
        return df

    def clean_criminality(self, df):
        df = df[['NBH_code', 'Total_theft_from_house', 'Destruction', 'Violence_and_sexual_abuse']]
        df = df[df['NBH_code'].str.contains('BU')]
        df['NBH_code'] = df['NBH_code'].astype(str)
        for a in range(len(df)):
            if df['NBH_code'].iloc[a][2:3] == '0' and df['NBH_code'].iloc[a][3:4] != '0':
                df['NBH_code'].iloc[a] = ''.join(df['NBH_code'].iloc[a].rsplit('BU0'))
            elif df['NBH_code'].iloc[a][2:4] == '00' and df['NBH_code'].iloc[a][4:5] != '0':
                df['NBH_code'].iloc[a] = ''.join(df['NBH_code'].iloc[a].rsplit('BU00'))
            elif df['NBH_code'].iloc[a][2:5] == '000':
                df['NBH_code'].iloc[a] = ''.join(df['NBH_code'].iloc[a].rsplit('BU000'))
            else:
                df['NBH_code'].iloc[a] = df['NBH_code'].iloc[a][2:]
        df = df.dropna(subset=['Total_theft_from_house', 'Destruction', 'Violence_and_sexual_abuse'])
        df['Total_theft_from_house'] = df['Total_theft_from_house'].str.extract(r'(\d)')
        df['Destruction'] = df['Destruction'].str.extract(r'(\d)')
        df['Violence_and_sexual_abuse'] = df['Violence_and_sexual_abuse'].str.extract(r'(\d)')
        df = df.dropna(subset=['Total_theft_from_house', 'Destruction', 'Violence_and_sexual_abuse'])
        df['Total_theft_from_house'] = df['Total_theft_from_house'].astype(int)
        df['Destruction'] = df['Destruction'].astype(int)
        df['Violence_and_sexual_abuse'] = df['Violence_and_sexual_abuse'].astype(int)
   
        return df

    def clean_buurt(self, df):
        df.buurtcode = df.buurtcode.astype(str)
        df = df[['buurtcode', 'buurtname']]
        df.drop_duplicates(subset='buurtcode', keep='first', inplace=True)
        return df

    def merge_cbs_datasets(self, buurt, income, density, criminality, school):
        df = pd.merge(buurt, criminality, left_on='buurtcode', right_on='NBH_code', how='left')
        df = pd.merge(df, income, left_on='buurtcode', right_on='NBH_code', how='left')
        df = pd.merge(df, density, left_on='buurtcode', right_on='NBH_code', how='left')
        df = pd.merge(df, school, left_on='buurtcode', right_on='NBH_code', how='left')
        df = df.dropna()
        return df










