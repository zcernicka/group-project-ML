# live coding session 

forecasting model:
 - predict for a given household, given the history of the consumption of this household, energy consumption 1 day ahead (next day consumption)

  - forecasting horizon (how far ahead the prediction is)

HOUSEHOLD 
t_-10, t_-9, ..., t_0 (today) -> t_1

retraining patterns
 - batch retraining 
 when new data comes, the model is retrained on the whole batch -> historical data + new data 

 new csv with the house prices -> retrain the pipeline on historical + new data 

 - online retraining:
 new data comes in, and the model is updated only on that new data (without retraining on the whole historical dataset)
 
 DATA BEST PRACTICE:
 - raw data types: text or binary or structured (database)
 - flat csv in a cloud bucket

 EXAMPLE COMPLEX PIPELINE
 IoT devices recording temperature
 -> cleaning step
 -> clean csv (output: clean_csv)
 -> upload to database
 -> analytics pipeline
 -> anomaly detection pipeline/prediction pipeline
 -> upload prediction to databse
 -> alert application
 -> alert the supervisor
 
 -- Upload the prediction to the db which will be taken over by the analytics or visualization tools

 ONLINE PREDICTION DEPLOYMENT
 in this case, the ML model lives in a web-api
 It's not always the best scenario to have the prediction online. do we need it real-time?

 ML model | web API <- python application requests

PIPELINE SEQUENCE:
1. load the raw data from disk -> energy_forecasting/load_data.py module
2. cleaning data energy_forecasting/load_data.py module
3. create features for the machine learning model energy_forecasting/featurezie_data.py module 
mean energy consumption last month | consumption for household 1 at time 0 | target (consumption for household 1 at time 1)
4. split your feature dataset into dataset: train set (80%), test set (20%)
5. fitting the ML model
- sklearn (scikit-learn)
- hypertuning the ML model parameters (grid search with CV)
6. refit the best model on the whole data 
7. Evaluate the best model on the test set 
- predict the target on the test set with your trained model (e.g house prices) 
- get the prediction for each test observation, and you the true target value for that observation 
- calculate model accuracy (RMSE)




 what does it mean by features?
 features (independent variables)
 additional features that may be constructed are: 
 - log of the square meter (help to transfor skewed distribution to normal distribution)
 - time to sale 






