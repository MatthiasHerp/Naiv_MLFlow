import numpy as np
import pandas as pd #data wrangeling
import matplotlib
import matplotlib.pyplot as plt
import os #for setting the working drectory
import mlflow

#Importing evaluation metrics
#Note: package is called scikit-learn (look for full name if i search it in anaconda)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

#defining the evaluation metrics
#from: https://mlflow.org/docs/latest/tutorials-and-examples/tutorial.html
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def naive_forecast(train, test, alpha):
    data = np.zeros(len(test))
    #create the naive forecasting series with the index identical to the days to forecast which are the test data set index
    naive_forecast = pd.Series(data,index=test.index)
    #loop goes threw the complete data and gets forecasts
    #we take the length thus the last day
    #minus 365 to get a year back
    #then minus one to get the same week day of the past year
    #then an additional minus 1 as 2016 is a leap year
    #i runs from 0 to 30 so that the horizon we go back deminishes as we forecast further and further
    for i in range (0,len(test)):
        naive_forecast[i] = alpha * train.revenue[len(train)-(365-1-i)]
    return naive_forecast



if __name__ == "__main__":
    
    revenue_CA_1_FOODS_day = pd.read_csv(os.path.join(os.getcwd(), "revenue_CA_1_FOODS_day.csv"), 
                                         index_col='date')

    # Split the data into training and test sets
    train = revenue_CA_1_FOODS_day.iloc[:(len(revenue_CA_1_FOODS_day)-31)]
    test = revenue_CA_1_FOODS_day.iloc[(len(revenue_CA_1_FOODS_day)-31):]

    with mlflow.start_run():

                prediction = naive_forecast(train, test, alpha)

                (rmse, mae, r2) = eval_metrics(test, prediction)

                print("Naive Model (alpha=%f):" % (alpha))
                print("  RMSE: %s" % rmse)
                print("  MAE: %s" % mae)
                print("  R2: %s" % r2)

                mlflow.log_param("alpha", alpha)
                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("r2", r2)
                mlflow.log_metric("mae", mae)

                #Plot the results

                plt.figure(figsize=(15, 5))
                plt.plot(test)
                plt.plot(prediction, color="red")
                plt.xlabel("date")
                plt.ylabel("revenue_CA_1_FOODS")
                plt.legend(("realization", "prediction"),  
                       loc="upper left")
                plt.savefig('prediction_plot.png')


                mlflow.log_artifact("./prediction_plot.png")
