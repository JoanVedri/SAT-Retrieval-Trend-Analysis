import pandas as pd
from glob import glob
import numpy as np
import xgboost as xgb
from pathlib import Path
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import scipy.stats as stats
import matplotlib.pyplot as plt
import argparse

### Script to create the different models for each database and temperature, also creates files with the best parameters found and the importance of each variable for the models

# Train and save models, save parameter files and feature selection figures
def train_xgb(Xtrain, ytrain, Xtest, features, label, temp, output):

    # Paths
    output_dir = Path(output)
    models_dir = output_dir / "modeling" / "models"
    parameter_dir = output_dir / "modeling" / "parameters"
    FS_dir = output_dir / "modeling" / "feature_selection"

    # Parameters to optimize models
    param_dist = {
        'max_depth': stats.randint(1, 30),
        'learning_rate': stats.uniform(0.01, 0.5),
        'n_estimators': stats.randint(500, 3000),
        'subsample': stats.uniform(0.5, 0.5),
    }

    model = xgb.XGBRegressor()

    # 10 K-fold with 500 iterations, searching best parameters
    search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=500, cv=10,
                                scoring='neg_mean_squared_error', verbose=1, random_state=1234, n_jobs=-1)
    
    search.fit(Xtrain, ytrain)

    # Get model with best parameters
    best_model = search.best_estimator_

    # Save model
    best_model.save_model(models_dir / f"XGB_SAT_{label}_{temp}.json")

    # Save parameters
    with open(parameter_dir / f"parametersXGB_{label}_{temp}.txt", "w") as f:
        f.write(f"Best hyperparameters for {label} {temp}: {search.best_params_}")

    # Obtain predictions for test inputs
    predictions = best_model.predict(Xtest)

    # Feature importance figure creation
    importances = best_model.feature_importances_
    indices = np.argsort(importances)

    plt.figure(figsize=(10, 6))
    plt.title(f"Feature Importances - {label} - {temp}")
    plt.barh(range(len(indices)), importances[indices], color='g', align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel("Relative Importance")
    plt.tight_layout()
    plt.savefig(FS_dir / f"importance_xgboost_{label}_{temp}.png")
    plt.close()

    return predictions

# Function which computes statistics for test results
def compute_stats(dif):

    # Classic statistics (minimum, maximum, mean, standard deviation and root mean square error)
    minim = np.nanmin(dif)
    maxim = np.nanmax(dif)
    mean = np.nanmean(dif)
    std = np.nanstd(dif, ddof=1)
    rmse = np.sqrt(std**2 + mean**2)

    # Robust statistics (median, robust standard deviation and robust root mean square error)
    median = np.nanmedian(dif)
    rsd = np.nanmedian(np.abs(dif - median)) * 1.4826
    rrmse = np.sqrt(median**2 + rsd**2)

    # Number of data of test data
    n = dif.size - np.count_nonzero(np.isnan(dif))

    return [minim, maxim, mean, std, rmse, median, rsd, rrmse, n]

# Variable selection for each database
def select_variables(database, Xtrain, Xtest):

    if "daytime" in database or "inst_day" in database:
        cols = ['LST_day', 'NDVI', 'Albedo', 'LATITUD', 'LONGITUD', 'ALTITUD',
                'Aspect', 'Slope', 'Dist_coast', 'Zenital_angle', 'Azimutal_angle', 'Inclinacion_solar']
        
    elif "nighttime" in database or "inst_night" in database:
        cols = ['LST_night', 'NDVI', 'Albedo', 'LATITUD', 'LONGITUD', 'ALTITUD',
                'Aspect', 'Slope', 'Dist_coast']
        
    elif "mixtime" in database:
        cols = ['LST_day', 'LST_night', 'NDVI', 'Albedo', 'LATITUD', 'LONGITUD', 'ALTITUD',
                'Aspect', 'Slope', 'Dist_coast', 'Zenital_angle', 'Azimutal_angle', 'Inclinacion_solar']
        
    else:
        raise ValueError("Unknown database type")

    return Xtrain[cols], Xtest[cols], cols

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, help="Directory to save models and outputs")
    args = parser.parse_args()

    dir = Path(args.dir)

    input_dir = dir / "preprocessing" / "databases"
    output_dir = dir / "modeling" 

    # Get instantaneous databases
    databases_inst = list(glob(str(input_dir / "*inst*")))

    # Get daily databases
    databases = list(glob(str(input_dir / "*time*")))

    stats_df = pd.DataFrame({'Statistics': ['minim', 'maxim', 'mn', 'sd', 'rmse', 'mediana', 'rsd', 'rrmse', 'n']})

    # Loop for instantaneous models
    for database in databases_inst:

        df =pd.read_csv(database)

        # Database name
        label = database.replace(str(input_dir / "database_"),"")
        label = label[:-4]

        X = df
        y = df['Temp']
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state=1234)
        Xtrain, Xtest, features = select_variables(label, Xtrain, Xtest)
        predictions = train_xgb(Xtrain, ytrain, Xtest, features, label=label.split('_')[1],temp = "inst", output  = dir)
        dif = predictions - ytest
        stats_df[f'Dif_{label}'] = compute_stats(dif)

    # Loop for daily models
    for database in databases:

        df =pd.read_csv(database)

        # Database name
        label = database.replace(str(input_dir / "database_"),"")
        label = label[:-4]

        X = df
        for temp in ['T_max', 'T_min', 'T_mean']:
            y = df[temp]
            Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state=1234)
            Xtrain, Xtest, features = select_variables(label, Xtrain, Xtest)
            predictions = train_xgb(Xtrain, ytrain, Xtest, features, label=label,temp=temp.split('_')[1], output=dir)
            dif = predictions - ytest
            stats_df[f'Dif_{label}_{temp}'] = compute_stats(dif)
    
    # Save statistics
    stats_df.to_csv(output_dir / f"statistics_xgb.csv", index=False)


if __name__ == "__main__":

    main()