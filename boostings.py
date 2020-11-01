# import libraries
import os
import pandas as pd
import lightgbm as lgb
# import xgboost as xgb
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from prepare_data import xy_split, train_test_split
from sklearn.externals import joblib
import settings # shared variables and funcions for models


# LGB classifier
class LGBClassifier:

    def __init__(self):                                 # initialization with creating data for test, train and clf model
        data = pd.read_csv(settings.prepared_data_path)
        X, y = xy_split(data)
        X = data.drop(['file_name', 'user_name', 'mark'], axis=1)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y)
        self.clf = lgb.LGBMClassifier()
        if os.path.exists(settings.lgb_weights):
            self.clf = joblib.load(settings.lgb_weights)
    
    def train(self):                                    # train clf model
        self.clf.fit(self.X_train, self.y_train)

    # returns clf model prediction on X input data 
    def predict(self, X):         
        print("TARGET DATA: ")
        print(X)          
        return self.clf.predict(X), self.clf.predict_proba(X)


#  interface for XGBoost classifier
class XGBoostClassifier:
    def __init__(self):                                 # initialization with creating train, test data and load pre-trained model
        data = pd.read_csv(settings.prepared_data_path)
        X, y = xy_split(data)
        X = data.drop(['file_name', 'user_name', 'mark'], axis=1)
        X = X.iloc[:, 231:]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y)
        self.load()

    def save(self):                                     # save current state of the model
        self.model.save_model(settings.xgb_weights)

    def load(self):                                     # load pre-trained model
        if os.path.exists(settings.xgb_weights):
            self.model.load_model(settings.xgb_weights)

    def train(                                      # train model
            self, 
            X_train,                                # 2D pandas array: 1 string = inputs for case
            y_train,                                # 1D pandas array: answers to every case, it should be [0, 1, ... (numberOfOutputCategories-1)]
            numberOfOutputCategories):              # number of possible output categories
    
        param = {                                       # Parameters of model
             'max_depth':X_train.shape[1] * 2,      # Maximum depth of a tree
             'eta':1,                               # decrease of step
             'objective':'multi:softmax',           # type of output
             'num_class':numberOfOutputCategories
            }       # number of possible output categories
        num_round = 1000                                # The number of rounds for training
        data_dmatrix = xgb.DMatrix(data=X_train,        # Create data container for train
                               label=y_train)
        self.model = xgb.train(param,                   # Train model
                           data_dmatrix, 
    num_round)

    def predict(self, X):                           # predict
        DMatrixX = xgb.DMatrix(X)
        return self.model.predict(DMatrixX)    