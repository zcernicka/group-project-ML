import pandas as pd
import numpy as np

class DataFeaturisation(object):

    def __init__(self):
        pass
    
    def DateToMonths (self, df):
        df['publicatieDatum'] = pd.to_datetime(df['publicatieDatum'])
        df['publicatieDatum'] = pd.DatetimeIndex(df['publicatieDatum']).month
        df['publicatieDatum'] = df['publicatieDatum'].astype(str)
        return df

    def AvgKoopPrijsCatPerBuurt (self,df):
        df_subset = df[['buurt', 'koopPrijs']]
        df_buurt_grouped = df_subset.groupby('buurt').mean().reset_index()
        buurt_classification = []
        for row in range(len(df_buurt_grouped)):
            if df_buurt_grouped['koopPrijs'].loc[row] <= 200000:
                brt_class = 'Low'
                buurt_classification.append(brt_class)
            elif df_buurt_grouped['koopPrijs'].loc[row] > 200000 and df_buurt_grouped['koopPrijs'].loc[row] <= 270000:
                brt_class = 'Moderate'
                buurt_classification.append(brt_class)
            elif df_buurt_grouped['koopPrijs'].loc[row] > 270000 and df_buurt_grouped['koopPrijs'].loc[row] <= 375000:
                brt_class = 'High'
                buurt_classification.append(brt_class)
            else:
                brt_class = 'Extremely High'
                buurt_classification.append(brt_class)
        df_buurt_grouped['buurt_classification'] = buurt_classification
        df_buurt_grouped = df_buurt_grouped.drop('koopPrijs', axis = 1 )
        df = pd.merge(df, df_buurt_grouped, left_on = 'buurt', right_on = 'buurt', how ='left')
        return df
    
    def PricePerSqM2 (self, df):
        df['price_sqm2'] = df['koopPrijs'] / df['oppervlakte']
        # must go to the very end, as KoopPrijs needed in the initial steps
        df = df.drop('koopPrijs', axis = 1)
        return df

    def dummy_coding(self, df):
        #choosing the right columns for further analysis
        df = df[['categorieObject','publicatieDatum', 'indTuin', 'perceelOppervlakte', 'energielabelKlasse', 
        'aantalKamers', 'aantalBadkamers', 'selling_days', 'house_age', 'buurt_classification', 
        'price_sqm2', 'Total_theft_from_house', 'Destruction', 'Violence_and_sexual_abuse', '%_low_income_households',
        '%_below_social_minimum', 'No_residents', 'Density', '0-4age_Binnen5Km_55', '4+age_Binnen10Km']]
        #perform the dummy coding 
        df['categorieObject'] = df['categorieObject'].replace(['Appartement', 'Woonhuis'], [0,1])
        df = pd.get_dummies(df, prefix = ['publicatieDatum','energielabelKlasse', 'buurt_classification'])
        df = df.reset_index()
        df = df.dropna()
        return df    
