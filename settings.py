# import libraries
import os


# shared variables and funcions for models
working_directory = os.path.abspath(os.path.dirname(__file__))              # main directory

dx_path = os.path.join(working_directory, "data/DX_TEST_RESULT_FULL.csv")   # path to data about positions of markups

k1 = 1
k2 = 1
k3 = 1

max_distance = int(1e7)    # max distance between two markups centers

result_folder = os.path.join(working_directory, "results")

expert_dir_path = os.path.join(working_directory, "data/Expert")            # path to directory with expert's markups
sample1_dir_path = os.path.join(working_directory, "data/sample_1")         # path to directory with 1st AI's markups
sample2_dir_path = os.path.join(working_directory, "data/sample_2")         # path to directory with 2nd AI's markups
sample3_dir_path = os.path.join(working_directory, "data/sample_3")         # path to directory with 3rd AI's markups

lgb_weights = os.path.join(working_directory, "weights/lgb.pkl")                           # path to directory with weights to LGB model
catboost_weights = os.path.join(working_directory, "weights/cat_weights.cat")                      # path to directory with weights to catboost(VGG16) model
xgb_weights = os.path.join(working_directory, "weights/00.xgbmodel")                           # path to directory with weights to XGB model
svm_weights = os.path.join(working_directory, "weights/00svmmodel.joblib")                           # path to directory with weights to SVM model

prepared_data_path_wo_answers = os.path.join(working_directory, "data/std_data_without_marks.csv")

prepared_data_path = os.path.join(working_directory, "data/std_data_with_marks-2.csv") # path to directory with prepaired data for models
expert_decisions_path = os.path.join(working_directory, "data/std_OpenPart.csv") # path to directory with expert's desisions data

not_dummy_path = os.path.join(working_directory, "data/SecretPart_not_dummy.csv")      # set of names for radiograms images

def calculate_l1(y_test, y_pred):                                           # function for calculating L1 metric between 2 sets of answers
    l1 = 0
    for i in range(len(y_test)): 
        l1 += abs(y_test[i] - y_pred[i])
    return l1