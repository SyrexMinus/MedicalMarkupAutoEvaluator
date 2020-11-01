# Import libraries
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Input, concatenate
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from catboost import CatBoostClassifier
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import settings # shared variables and funcions for models

import pandas as pd
import cv2
import json
import os


class VGG16FeatureExtractor:    # feature extractor from images using convolutional VGG16 model

    def __init__(self):         # Initialize cutted VGG16 model method 
        self.vgg16 = VGG16(weights="imagenet", include_top=False)


    # Prepares image for passing through VGG
    def prepare_image(self, image_path, target_size=(224, 224)):
        img = image.load_img(image_path, target_size=target_size)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        return x


    # returns extracted from image features
    def extract(self, image_path):
        features = self.vgg16.predict(self.prepare_image(image_path))    
        features_np = np.array(features)
        return features_np.flatten()


    # returns features for all markups of given research
    def extract_research_features(self, research_name):
        expert_img_path = f"{settings.expert_dir_path}/{research_name}_expert.png"
        sample_1_img_path = f"{settings.sample1_dir_path}/{research_name}_s1.png"
        sample_2_img_path = f"{settings.sample2_dir_path}/{research_name}_s2.png"
        sample_3_img_path = f"{settings.sample3_dir_path}/{research_name}_s3.png"

        expert_features = self.extract(expert_img_path)     # features for expert markup
        sample_1_features = self.extract(sample_1_img_path) # features for 1st NN markup
        sample_2_features = self.extract(sample_2_img_path) # features for 2nd NN markup
        sample_3_features = self.extract(sample_3_img_path) # features for 3rd NN markup
        features_length = len(expert_features)

        X_np = np.vstack((                                  # table with features for every NN model markup 
            np.hstack((expert_features, sample_1_features)),
            np.hstack((expert_features, sample_2_features)),
            np.hstack((expert_features, sample_3_features)),
        ))

        y_np = np.array(settings.expert_decisions.loc[      # ?
            settings.expert_decisions["file_name"] == research_name,
            ["sample_1", "sample_2", "sample_3"]
            ])

        if len(y_np) == 0:
            return X_np, None
        return X_np, y_np[0]



class VGG16Classifier:                              # class for classifier based on VGG16 model

    def __init__(self,                              # initializer for feature_extractor and boosting
                 cat_iterations=2000,                       # number of epochs for train
                 cat_task_type="GPU",                       # type of computing for train
                 cat_loss="MultiClass",                     # type of loss for train
                 cat_dept=10,                               # depth of boosting tree
                 cat_lr=0.1):                               # learning rate
        self.feature_extractor = VGG16FeatureExtractor()
        self.boosting = CatBoostClassifier(iterations=cat_iterations, task_type=cat_task_type, 
            loss_function=cat_loss, depth=cat_dept, learning_rate=cat_lr)

        if os.path.exists(settings.catboost_weights):
            self.boosting.load_model(settings.catboost_weights)


    # returns data for train, ?, names of images for train and ?
    def build_dataset(self,                         # build dataset for train method
                      feature_size=50176):                  # number of extracted by VGG16 features
        X_train = np.empty((0, feature_size))
        y_train = np.array([])
        X_target = np.empty((0, feature_size))              # ?
        train_filenames, target_filenames = [], []
        for fn in settings.unique_research_names:
            if len(settings.expert_decisions.loc[settings.expert_decisions["file_name"] == fn]) == 0:
                target_filenames += [fn] * 3
                X, _ = self.feature_extractor.extract_research_features(fn)
                X_target = np.vstack((X_target, X))
                continue

            train_filenames += [fn] * 3
            X, y = self.feature_extractor.extract_research_features(fn)
            X_train = np.vstack((X_train, X))
            y_train = np.hstack((y_train, y))
        return X_train, y_train, X_target, train_filenames, target_filenames


    def train_classifier(self):                     # train classifier on train data method
        X_train, y_train, X_target, train_filenames, target_filenames = self.build_dataset()
        self.boosting.fit(X_train, y_train, verbose=True)


    # returns prediction of evaluating AI based on pair of expert's an AI's markups
    def predict(self, image1_path, image2_path):    # model predict method 
        image1 = self.feature_extractor.extract(image1_path)
        image2 = self.feature_extractor.extract(image2_path)

        X_np = np.hstack((image1, image2))
        prediction = self.boosting.predict(X_np)
        return prediction