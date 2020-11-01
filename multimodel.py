# import libraries
from boostings import LGBClassifier, XGBoostClassifier
from svm import SVM
from vgg16_classifier import VGG16Classifier
import settings
import pandas as pd
import numpy as np
from datetime import datetime

# model aggregator class
class MultiModel:
    def __init__(self):     # initialization with models init
        self.svm = SVM()
        self.vgg_cl = VGG16Classifier()
        self.lgb_cl = LGBClassifier()
        # self.xgb_cl = XGBoostClassifier()

    
    def predict_on_name(self, name):
        data = pd.read_csv(settings.prepared_data_path_wo_answers)

        target_data = data[data["file_name"] == name]

        target_data = target_data.sort_values(by="user_name")



        if len(target_data) < 3:
            print("Bad example: ", len(target_data))
            pr_data = pd.read_csv(settings.dx_path)

            exp_r = len(data[(data["file_name"] == name) & (data["user_name"] == "Expert")])

            answers = []
            for user in ["sample_1", "sample_2", "sample_3"]:
                user_r = len(data[(data["file_name"] == name) & (data["user_name"] == user)])
                if exp_r != user_r:
                    answers.append(1)
                else:
                    answers.append(5)
            return answers


        target_data_bu = target_data.copy()
        target_data.drop(["Unnamed: 0", "file_name", "user_name"], axis=1, inplace=True)

        print(target_data)
        svm_pr, svm_pr_proba = self.svm.predict(target_data.filter(['iou', 'iou_intersections', 'center', # select only useful for this model features
                      'ml_count', 'expert_count',
                      'expert_intersect_area', 
                      'expert_union_area', 'expert_area', 
                      'ml_intersect_area', 'ml_union_area', 
                      'ml_area']))
 
        print("SVM: ", svm_pr)
        print(target_data_bu.iloc[0]["user_name"])
        print(target_data_bu.iloc[1]["user_name"])
        print(target_data_bu.iloc[2]["user_name"])


        vgg_pr_1 = self.vgg_cl.predict(f"{settings.expert_dir_path}/{name}_expert.png", f"{settings.sample1_dir_path}/{name}_s1.png")
        vgg_pr_2 = self.vgg_cl.predict(f"{settings.expert_dir_path}/{name}_expert.png", f"{settings.sample2_dir_path}/{name}_s2.png")
        vgg_pr_3 = self.vgg_cl.predict(f"{settings.expert_dir_path}/{name}_expert.png", f"{settings.sample3_dir_path}/{name}_s3.png")
        vgg_pr = [vgg_pr_1[0], vgg_pr_2[0], vgg_pr_3[0]]

        lgb_pr = self.lgb_cl.predict(target_data)[0]
        print("LGB: ", lgb_pr)

        M = [0, 0, 0]
        for i in range(3):
            M[i] = svm_pr[i] * settings.k1 + vgg_pr[i] * settings.k2 + lgb_pr[i] * settings.k3
            M[i] = M[i] / (settings.k1 + settings.k2 + settings.k3)

        return M



if __name__ == "__main__":
    mm = MultiModel()

    not_dummy = pd.read_csv(settings.not_dummy_path)

    for i, file_name in enumerate(not_dummy["Case"]):
        preds = mm.predict_on_name(file_name.split(".")[0])
        not_dummy.at[i, "Sample 1"] = preds[0]
        not_dummy.at[i, "Sample 2"] = preds[1]
        not_dummy.at[i, "Sample 3"] = preds[2]
    
    not_dummy.to_csv(f"{settings.result_folder}/{datetime.now()}result.csv")