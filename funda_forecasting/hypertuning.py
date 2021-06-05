from copy import deepcopy
import numpy as np
import pandas as pd
import os
import itertools
import statistics
import pickle
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import copy


class DataPartitioner(object):
    '''Class to partition data into train and test,
    and to partition test into 5-fold cross validation splits.
    '''
    def __init__(self, perc_test = 0.2, train_cv_splits = 5, seed = 9):
        self.perc_test = perc_test
        self.train_cv_splits = train_cv_splits
        self.seed = seed

    def partition_data(self, df):
        obs = df.copy()
        obs_test = obs.sample(frac=self.perc_test, random_state = self.seed).copy()
        obs_test = obs_test.assign(test = True).assign(cv_split=np.nan)

        obs_train = (
        obs.merge(obs_test, how = 'left', indicator = True)
        .query("_merge == 'left_only'")
        .drop('_merge', axis=1)
        .assign(test=False))
        cv_splits = np.random.randint(1, self.train_cv_splits + 1, obs_train.shape[0])
        obs_train = obs_train.assign(cv_split = cv_splits)
        final_obs = pd.concat([obs_test, obs_train]).reset_index(drop=True)
        return final_obs#, obs_train, obs_test

parameter_combos = []
parameter_combos_dicts = []

class Hypertuner_price_sqm2(object):
    def __init__(self, estimator, tuning_params, validation_mapping):
        self.estimator = estimator
        self.tuning_params = tuning_params
        self.validation_mapping = validation_mapping



    def calculate_mean_cv_error(self, train_set, estimator_cv):
        # now perform cross validation fitting for each split
        splits = train_set['cv_split'].unique().tolist()
        splits.sort()

        cv_errors = []
        cv_abs_errors = []

        for i in splits:
            train_set =  train_set.drop([ 'index', 'test'], axis = 1)
            train_split = train_set.query(f"cv_split != {i}")
            x_train = train_split.loc[:, train_split.columns != 'price_sqm2']
            x_train = x_train.drop('cv_split', axis = 1)
            y_train = train_split['price_sqm2']
            model = estimator_cv.fit(x_train, y_train)
            # evaluate the model on split 1
            test_split = train_set.query(f"cv_split == {i}")
            x_test = test_split.loc[:, test_split.columns != 'price_sqm2']
            x_test = x_test.drop('cv_split', axis = 1)
            y_test = test_split['price_sqm2']
            y_pred = model.predict(x_test)
            # calculate error measure on this fold for the estimator with the
            # given parameters
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            
            cv_errors.append(rmse)
            cv_abs_errors.append(mae)

            mean_rmse = statistics.mean(cv_errors)
            mean_mae = statistics.mean(cv_abs_errors)

            # saving the model to Pickle

            return mean_rmse, mean_mae, y_test, y_pred, model
        
        
    def tune_model(self, train_set):
        '''Perform the hypertuning of the estimator on the train set
        for all the combinations of the hyperparameters
        '''
        parameter_combos = []
        parameter_combos_dicts = []
        for a in itertools.product(*self.tuning_params.values()):
            parameter_combos.append(a)

        for i in parameter_combos:
            d = {}
            for j in range(len(i)):
                d[list(self.tuning_params.keys())[j]] = i[j]
            parameter_combos_dicts.append(d)

        validation_mapping_train = self.validation_mapping.query("test == False")
        train_set = train_set.merge(validation_mapping_train)

        params = ''
        mean_mae = 1000000 #just to make the condition works
        for d in parameter_combos_dicts:
            estimator_cv = deepcopy(self.estimator)
            estimator_cv = estimator_cv.set_params(**d)
            mean_cv_error = self.calculate_mean_cv_error(train_set, estimator_cv)
            print(d, '    RMSE: ', mean_cv_error[0], '---MAE: ', mean_cv_error[1])

            if mean_mae >= mean_cv_error[1]:
                mean_mae = mean_cv_error[1]
                params = d

        print('The best model is ', params, '---MAE', mean_mae)
        # creating train set
        train_set_model = train_set.drop(['cv_split', 'price_sqm2', 'test', 'index'], axis=1)
        # train the best model on all train set
        final_estimator = deepcopy(self.estimator)
        final_estimator = self.estimator.set_params(**params)
        model = final_estimator.fit(train_set_model, train_set['price_sqm2'])

        return mean_cv_error


class Hypertuner_selling_days(object):
    def __init__(self, estimator, tuning_params, validation_mapping):
        self.estimator = estimator
        self.tuning_params = tuning_params
        self.validation_mapping = validation_mapping

    def calculate_mean_cv_error(self, train_set, estimator_cv):
        # now perform cross validation fitting for each split
        splits = train_set['cv_split'].unique().tolist()
        splits.sort()

        cv_errors = []
        cv_abs_errors = []

        for i in splits:
            train_set = train_set.drop(['index', 'test'], axis=1)
            train_split = train_set.query(f"cv_split != {i}")
            x_train = train_split.loc[:, train_split.columns != 'selling_days']
            x_train = x_train.drop('cv_split', axis=1)
            y_train = train_split['selling_days']
            model = estimator_cv.fit(x_train, y_train)
            # evaluate the model on split 1
            test_split = train_set.query(f"cv_split == {i}")
            x_test = test_split.loc[:, test_split.columns != 'selling_days']
            x_test = x_test.drop('cv_split', axis=1)
            y_test = test_split['selling_days']
            y_pred = model.predict(x_test)
            # calculate error measure on this fold for the estimator with the
            # given parameters
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)

            cv_errors.append(rmse)
            cv_abs_errors.append(mae)

            mean_rmse = statistics.mean(cv_errors)
            mean_mae = statistics.mean(cv_abs_errors)
            return mean_rmse, mean_mae, y_test, y_pred, model

    def tune_model(self, train_set):
        '''Perform the hypertuning of the estimator on the train set
        for all the combinations of the hyperparameters
        '''
        parameter_combos = []
        parameter_combos_dicts = []
        for a in itertools.product(*self.tuning_params.values()):
            parameter_combos.append(a)

        for i in parameter_combos:
            d = {}
            for j in range(len(i)):
                d[list(self.tuning_params.keys())[j]] = i[j]
            parameter_combos_dicts.append(d)

        validation_mapping_train = self.validation_mapping.query("test == False")
        train_set = train_set.merge(validation_mapping_train)
        params = ''
        mean_mae = 1000000 #just to make the condition works
        for d in parameter_combos_dicts:
            estimator_cv = deepcopy(self.estimator)
            estimator_cv = estimator_cv.set_params(**d)
            mean_cv_error = self.calculate_mean_cv_error(train_set, estimator_cv)
            print(d, '    RMSE: ', mean_cv_error[0], '---MAE: ', mean_cv_error[1])

            if mean_mae >= mean_cv_error[1]:
                mean_mae = mean_cv_error[1]
                params = d

        print('The best model is ', params, '---MAE', mean_mae)
        # creating train set
        train_set_model = train_set.drop(['cv_split', 'selling_days', 'test', 'index'], axis=1)
        # train the best model on all train set
        final_estimator = deepcopy(self.estimator)
        final_estimator = self.estimator.set_params(**params)
        model = final_estimator.fit(train_set_model, train_set['selling_days'])
        return model


