import os
import mlflow
import argparse
from mlflow.tracking import MlflowClient
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score,roc_curve

import mlflow
import mlflow.sklearn
import shutil
from sklearn.metrics import accuracy_score, precision_score, recall_score


# define functions
def main(args):
    # enable auto logging
    current_run = mlflow.start_run()
    #mlflow.sklearn.autolog()

    # read in data
    df = pd.read_csv(args.titanic_csv)
    model, X_test = model_train('Survived', df, args.randomstate)
    
    model_file = os.path.join('outputs', 'titanic_model.pkl')
    joblib.dump(value=model, filename=model_file)
    
    shutil.copy('./outputs/titanic_model.pkl', os.path.join(args.model_output, "titanic_model.pkl"))
    
    
    X_test.to_csv(args.test_data)

def model_train(LABEL, df, randomstate):
    print('df.columns = ')
    print(df.columns)
    y_raw           = df[LABEL]
    columns_to_keep = ['Embarked', 'Loc', 'Sex','Pclass', 'Age', 'Fare', 'GroupSize']
    X_raw           = df[columns_to_keep]
    
    X_raw['Embarked'] = X_raw['Embarked'].astype(object)
    X_raw['Loc'] = X_raw['Loc'].astype(object)
    X_raw['Loc'] = X_raw['Sex'].astype(object)
    X_raw['Pclass'] = X_raw['Pclass'].astype(float)
    X_raw['Age'] = X_raw['Age'].astype(float)
    X_raw['Fare'] = X_raw['Fare'].astype(float)
    X_raw['GroupSize'] = X_raw['GroupSize'].astype(float)
    


    print(X_raw.columns)
     # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.2, random_state=randomstate)
    
    #use Logistic Regression estimator from scikit learn
    lg = LogisticRegression(penalty='l2', C=1.0, solver='liblinear')
    preprocessor = buildpreprocessorpipeline(X_train)
    
    #estimator instance
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                               ('regressor', lg)], verbose=True)

    model = clf.fit(X_train, y_train)
    
    print('type of X_test = ' + str(type(X_test)))
          
    y_pred = model.predict(X_test)
    
    print('*****X_test************')
    print(X_test)
    
    metrics = mlflow.sklearn.eval_and_log_metrics(model, X_test, y_test, prefix="test_")
    
    #get the active run.
    run = mlflow.active_run()
    print("Active run_id: {}".format(run.info.run_id))
    MlflowClient().log_metric(run.info.run_id, "metric", 0.22)

    
    return model, X_test

    mlflow.end_run()


def buildpreprocessorpipeline(X_raw):

    categorical_features = X_raw.select_dtypes(include=['object', 'bool']).columns
    numeric_features = X_raw.select_dtypes(include=['float','int64']).columns

    categorical_transformer = Pipeline(steps=[('onehotencoder', 
                                               OneHotEncoder(categories='auto', sparse=False, handle_unknown='ignore'))])


    numeric_transformer1 = Pipeline(steps=[('scaler1', SimpleImputer(missing_values=np.nan, strategy = 'mean'))])
    

    preprocessor = ColumnTransformer(
        transformers=[
            ('numeric1', numeric_transformer1, numeric_features),
            ('categorical', categorical_transformer, categorical_features)], remainder='drop')
    
    return preprocessor



def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("---training_data", type=str)
    parser.add_argument("---randomstate", type=int, default=42)
    parser.add_argument("--test_data", type=str,)
    parser.add_argument("--model_output", type=str, help="Path of output model")

    # parse args
    args = parser.parse_args()
    print(args)
    # return args
    return args


# run script
if __name__ == "__main__":
    # parse args
    args = parse_args()

    # run main function
    main(args)
