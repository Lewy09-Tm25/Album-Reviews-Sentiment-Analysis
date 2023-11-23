import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from textblob import TextBlob

# below package is just to let me know that the model script has been completely run :) works only in windows
# import winsound

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LinearRegression, Lasso, Ridge, SGDRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

if __name__ == "__main__":

    # reading the data
    data = pd.read_csv("datasets/sentiments_extracted_data.csv")
    
    X = data.drop(["score","label"], axis = 1)
    y = data['score']

    # splitting the features and target
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

    # parameters for cross-validation
    lasso_params = {
        'alpha' : [0.01,0.1,1.0]
    }
    ridge_params = {
        'alpha' : [0.01,0.1,1.0]
    }
    sgd_params = {
        'penalty' : ['l1','l2','elasticnet',None],
        'alpha' : [0.0001, 0.001, 0.01, 0.1, 0.00001],
        'learning_rate' : ['invscaling','optimal'],
        'max_iter' : [1000, 2000, 3000]
    }
    svr_params = {
        'kernel' : ['rbf','poly','sigmoid'],
        'degree' : [2,3,4],
        'gamma' : ['scale','auto'],
    }
    dt_params = {
        'criterion' : ['squared_error','friedman_mse','absolute_error','poisson'],
        'splitter' : ['best','random']
    }
    rf_params = {
        'n_estimators' : [10,25,50,100],
        'criterion' : ['squared_error','friedman_mse','absolute_error','poisson'],
    }

    # initializing the models and their parameter/s
    models_params = [
        (LinearRegression,{}), 
        (Lasso, lasso_params), 
        (Ridge, ridge_params), 
        (SGDRegressor, sgd_params), 
        (SVR, svr_params), 
        (DecisionTreeRegressor, dt_params), 
        (RandomForestRegressor, rf_params)]

    # initializing mse for each model
    mses = []
    maes = []
    r2s = []

    for model, params in models_params:
        gs_reg = GridSearchCV(
            estimator = model(),
            param_grid = params,
            cv = 5,
        )
        gs_reg.fit(x_train, y_train)

        best_param = gs_reg.best_params_
        print(f"Best parameters for {model}: {best_param}")
        inloop_model = model(**gs_reg.best_params_)
        inloop_model.fit(x_train, y_train)

        # saving the model
        joblib.dump(inloop_model, f"models/{model.__name__}.pkl")

        pred = inloop_model.predict(x_test)
        pred = [round(score,1) for score in pred]

        mses.append(mean_squared_error(y_test, pred))
        maes.append(mean_absolute_error(y_test, pred))
        r2s.append(r2_score(y_test, pred))
        print("------------")
    
    # creating the metrics table
    models = [model.__name__ for model,_ in models_params]
    mse_table = pd.DataFrame(data = zip(models,mses,maes,r2s), columns = ['Model','MSE','MAE','R2'])
    mse_table.to_csv("metrics/mean_squared_errors.csv", index = False)
    print(mse_table)

    # duration = 5000  # milliseconds
    # freq = 1000  # Hz
    # winsound.Beep(freq, duration)