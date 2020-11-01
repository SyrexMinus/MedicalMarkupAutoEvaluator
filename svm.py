# Import libraries
import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib

import settings # shared variables and funcions for models

from prepare_data import xy_split, train_test_split
import os


class SVM:                  # Support Vector Machine model class
    def __init__(self):
        data = pd.read_csv(settings.prepared_data_path)     # read dataset which is defined in settings
        X, y = xy_split(data)                               # split inputs to model and answers
        X = X.filter(['iou', 'iou_intersections', 'center', # select only useful for this model features
                      'ml_count', 'expert_count',
                      'expert_intersect_area', 
                      'expert_union_area', 'expert_area', 
                      'ml_intersect_area', 'ml_union_area', 
                      'ml_area'])

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y)   # memorize data for model train and test
        print(self.X_train)
        self.clf = make_pipeline(StandardScaler(), SVC(gamma='auto', probability=True)) # defining model

        if os.path.exists(settings.svm_weights):
            self.clf = joblib.load(settings.svm_weights)


    def train(self):        # train model on train data
        self.clf.fit(self.X_train, self.y_train.ravel())


    # returns prediction of evaluating AI based on features of pair expert's an AI's markups
    def predict(self, X):   # model predict for X input data
        return self.clf.predict(X), self.clf.predict_proba(X)