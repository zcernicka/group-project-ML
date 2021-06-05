"""This script runs the whole pipeline."""
import sys
import os
from pathlib import Path
#sys.path.insert(1, r'C:/Users/Marselo/Documents/Github2/product3team9')
print(sys.path)
import pandas as pd
import json
import datetime
from funda_forecasting.load_data import DataLoader
from funda_forecasting.clean_data import DataCleaner
from funda_forecasting.featurize_data import DataFeaturisation
from funda_forecasting.hypertuning import DataPartitioner, Hypertuner_selling_days, Hypertuner_price_sqm2
from funda_forecasting.modelling import neural_network
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import pickle


def main():

    ## PREPPING
    # setting the run id
    run_id_start_time = datetime.datetime.now()

    print(f"Starting with run at time {run_id_start_time}")
    # read in config  #C:/Users/kerem/Documents/Master/Q 1/DB Mgmt/Artificial intelligence
    with open('./run/conf.json', 'r') as f:
        conf = json.load(f)

    run_folder = os.path.join(conf['base_folder'], 'run_' + run_id_start_time.strftime("%Y%m%d_%H%M"))

    for i in ['cleaned', 'log', 'processed', 'models', 'predictions']:
        Path(run_folder, i).mkdir(parents=True, exist_ok=True)
    # if the raw folder does not exist, stop and throw an error
    assert os.path.exists(os.path.join(conf['base_folder'], 'raw')), "I can't find the raw folder!"
    # make sure we have all folders where the output of the run
    # will be stored

    f = open('./run/conf.json', 'r')
    conf = json.load(f)
    basefolder = conf['base_folder']

    #log config for the fun
    with open(os.path.join(basefolder, 'log', 'run_config.json'), 'w') as f:
        json.dump(conf, f)

    reload_clean_data = False
    try:
        reload_clean_data = conf['loading_params']['reload_clean_data']
    except KeyError:
        pass
    
    if reload_clean_data:
        print("Attempting to reload previously cleaned data")
        try:
            # finding the latest run
            runs = [x for x in os.listdir(conf['base_folder']) if x.startswith('run')]
            runs.sort()
            previous_run = runs[-2]
            # copying over the cleaned data of the previous run
            shutil.copyfile(os.path.join(conf['base_folder'], previous_run, 'cleaned', 'buurt_df.csv'),
            os.path.join(conf['base_folder'], run_folder, 'cleaned', 'buurt_df.csv'))
            shutil.copyfile(os.path.join(conf['base_folder'], previous_run, 'cleaned', 'cleaned_criminality.csv'),
            os.path.join(conf['base_folder'], run_folder, 'cleaned', 'cleaned_criminality.csv'))
            shutil.copyfile(os.path.join(conf['base_folder'], previous_run, 'cleaned', 'cleaned_density.csv'),
            os.path.join(conf['base_folder'], run_folder, 'cleaned', 'cleaned_density.csv'))
            shutil.copyfile(os.path.join(conf['base_folder'], previous_run, 'cleaned', 'cleaned_income.csv'),
            os.path.join(conf['base_folder'], run_folder, 'cleaned', 'cleaned_income.csv'))
            shutil.copyfile(os.path.join(conf['base_folder'], previous_run, 'cleaned', 'clean_school.csv'),
            os.path.join(conf['base_folder'], run_folder, 'cleaned', 'clean_school.csv'))
            shutil.copyfile(os.path.join(conf['base_folder'], previous_run, 'cleaned', 'funda_with_buurt.csv'),
            os.path.join(conf['base_folder'], run_folder, 'cleaned', 'funda_with_buurt.csv'))
            shutil.copyfile(os.path.join(conf['base_folder'], previous_run, 'processed', 'funda_final.csv'),
            os.path.join(conf['base_folder'], run_folder, 'processed', 'funda_final.csv'))


      # loading the clean data of the previous run
            funda_final = pd.read_csv(os.path.join(run_folder, 'processed', "funda_final.csv"))

            print("previously cleaned data reloaded")
        except Exception as e:
            print(f'''reloading previously cleaned data failed with error {e}./n
            Falling back on regenerating clean data.
            ''')
            reload_clean_data = False

    if reload_clean_data is False:
        #Creating buurt table
        #Clean cbs_address
        postal_code = DataLoader(basefolder).load_data('raw/cbs_addresses/pc6hnr20180801_gwb-vs2.csv')
        buurt = DataLoader(basefolder).load_data('raw/cbs_addresses/buurt.csv')
        wijk = DataLoader(basefolder).load_data('raw/cbs_addresses/wijk.csv')
        gem = DataLoader(basefolder).load_data('raw/cbs_addresses/gem.csv')
        buurt_df = DataCleaner().CreateBuurtTable(postal_code, buurt, wijk, gem)
        buurt_df.to_csv(os.path.join(run_folder, "cleaned", "buurt_df.csv"))

        #Cleaning Funda Dataset
        housing_data = DataLoader(basefolder).load_data2('raw/housing_data.csv')
        housing_data = DataCleaner().clean_category(housing_data)
        housing_data = DataCleaner().clean_year(housing_data)  
        housing_data = DataCleaner().clean_house_type(housing_data)
        housing_data = DataCleaner().clean_badkamers(housing_data)
        housing_data = DataCleaner().clean_living_area(housing_data)
        housing_data = DataCleaner().clean_aantal_badkamers(housing_data)
        housing_data = DataCleaner().add_selling_days(housing_data)
        housing_data = DataCleaner().clean_energy_label(housing_data)
        housing_data = DataCleaner().clean_selling_price(housing_data)
    
        #merging funda with buurt
        funda_with_buurt = DataCleaner().MergeHousingBuurt(housing_data, buurt_df)
        funda_with_buurt = DataCleaner().clean_cbs_address(funda_with_buurt)
        funda_with_buurt.to_csv(os.path.join(run_folder, "cleaned", "funda_with_buurt.csv"), index = False)


        #cleaning cbs datasets
        cbs_income = DataLoader(basefolder).load_data('raw/cbs_datasets/cbs_income_social_security.csv')
        clean_income = DataCleaner().clean_income(cbs_income)
        clean_income.to_csv(os.path.join(run_folder, "cleaned", "clean_income.csv"), index = False)
        cbs_school = DataLoader(basefolder).load_data('raw/cbs_datasets/cbs_school.csv')
        clean_school = DataCleaner().clean_school(cbs_school)
        clean_school.to_csv(os.path.join(run_folder, "cleaned", "clean_school.csv"), index = False)
        cbs_density = DataLoader(basefolder).load_data('raw/cbs_datasets/cbs_sex_age_populationdensity.csv')
        clean_density = DataCleaner().clean_density(cbs_density)
        clean_density.to_csv(os.path.join(run_folder, "cleaned", "clean_density.csv"), index = False)
        cbs_criminality = DataLoader(basefolder).load_data('raw/cbs_datasets/cbs_criminality.csv')
        clean_criminality = DataCleaner().clean_criminality(cbs_criminality)
        clean_criminality.to_csv(os.path.join(run_folder, "cleaned", "clean_criminality.csv"), index = False)

        #merging cbs with funda
        funda_with_cbs = DataCleaner().merge_cbs_datasets(funda_with_buurt,clean_income, clean_density,clean_criminality,clean_school)

        #Featurizing Funda Dataset
        # CREATE MODELLING FEATURES
        #funda_final = DataLoader(basefolder).load_data2('cleaned/funda_with_buurt.csv')
        funda_final = DataFeaturisation().DateToMonths(funda_with_cbs)
        funda_final = DataFeaturisation().AvgKoopPrijsCatPerBuurt(funda_final)
        funda_final = DataFeaturisation().PricePerSqM2(funda_final)
        funda_final = DataFeaturisation().dummy_coding(funda_final)
        funda_final.to_csv(os.path.join(run_folder, "processed", "funda_final.csv"), index = False)


    ## 1. CREATE TRAIN AND TEST SETS AND CV SPLITS
    validation_mapping = DataPartitioner().partition_data(funda_final)
    validation_mapping.to_csv(os.path.join(run_folder, "processed", "validation_mapping.csv"))


    print('Please select your preferred model')
    print('---------------------------------------------------------')
    print('1: random forest - price per sq2 prediction (most popular)')
    print('---------------------------------------------------------')
    print('2: random forest (XGBoosted) - price per sq2 prediction')
    print('---------------------------------------------------------')
    print('3: random forest - selling days prediction')
    print('---------------------------------------------------------')
    print('4: random forest (XGBoosted) - selling days prediction')
    print('---------------------------------------------------------')
    print('5: neural network - price per sq2 prediction')
    print('---------------------------------------------------------')

    option = input("Please choose your preferred model now: ")

    if option == "1":
        ## 2. SELECT BEST FORECASTING MODEL WITH 5 FOLD CV ON TRAIN FEATURES
        hypertuner_price_sqm2 = Hypertuner_price_sqm2(estimator=RandomForestRegressor(random_state=1234),
                                                      tuning_params=conf["training_params"]["hypertuning"]["RF_params"],
                                                      validation_mapping=validation_mapping
                                                      ).tune_model(validation_mapping)
        filename = os.path.join(run_folder, "models", "rf_hypertuned_model_price_sqm2.sav")
        pickle.dump(hypertuner_price_sqm2, open(filename, 'wb'))



    elif option == "2":
        sq2_hypertuner_xgb = Hypertuner_price_sqm2(estimator=XGBRegressor(random_state=1234),
                                    tuning_params=conf["training_params"]["hypertuning"]["XGB_params"],
                                    validation_mapping=validation_mapping
                                    ).tune_model(validation_mapping)
        filename = os.path.join(run_folder, "models", "xgb_hypertuned_model_price_sqm2.sav")
        pickle.dump(sq2_hypertuner_xgb, open(filename, 'wb'))


    elif option == "3":

        hypertuner_selling_days = Hypertuner_selling_days(estimator = RandomForestRegressor(random_state=1234),
        tuning_params = conf["training_params"]["hypertuning"]["RF_params"],
        validation_mapping = validation_mapping
                                   ).tune_model(validation_mapping)
        filename = os.path.join(run_folder, "models", "rf_hypertuned_model_price_sqm2.sav")
        pickle.dump(hypertuner_selling_days, open(filename, 'wb'))

    elif option == "4":

        sd_hypertuner_xgb = Hypertuner_selling_days(estimator = XGBRegressor(random_state=1234),
        tuning_params = conf["training_params"]["hypertuning"]["XGB_params"],
        validation_mapping = validation_mapping
                                   ).tune_model(validation_mapping)
        filename = os.path.join(run_folder, "models", "xgb_hypertuned_model_price_sqm2.sav")
        pickle.dump(sd_hypertuner_xgb, open(filename, 'wb'))


    elif option == "5":
        ## Neural Network: retrieve the history of every epoch's training
        neural_network = neural_network().initial_sequence(funda_final)

        filename = os.path.join(run_folder, "models", "neural_network_price_sqm2.sav")
        pickle.dump(neural_network[1], open(filename, 'wb'))


    else:
        print('really? cant you even read?')


if __name__ == "__main__":
    main()


