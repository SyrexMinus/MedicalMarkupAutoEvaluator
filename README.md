# MedicalMarkupAutoEvaluator
A solution for evaluating the quality of the markup of AI models on medical radiograms. It compares two markups: expert's and AI's and based on this Evaluator give mark from 1 to 5 — quality of recognition of pathologies.


Solution uses ansemble of LGBM, CatBoost of features extracted by VGG-16 net, SVM, XGBoost models, which analyse features extracted from coordinates/sizes and pictures of markups. 


## Requirements:
- Python 3.7
- pip 19.2.3
- Tensorflow library
- pandas library
- lightgbm library
- sklearn library
- numpy library
- datetime library
- opencv-python (cv2) library
- keras library
- catboost library
- json library
- Update all libraries using "!pip instal --upgrade *name_of_library*" 
- Some software to open *.ipynb* files 


## Usage:
### Native: 
1. Download weights for models from [our google drive](https://drive.google.com/drive/folders/1o0vahEkKFOw3A060YHTTWaDf7nxRcWjy?usp=sharing) and put them to */weights* directory
1.1 If you want to make predictions on new dataset, rebuild dataset(See *Rebuild dataset* section how to do this)
2. Run multimodel.py
### Docker:
1. Download and build [continer](https://drive.google.com/file/d/1ptw-AFTEXtRSuig0Y-5RYaAPC_24k8hB/view?usp=sharing)  
`docker build . -t MedicalMarkupAutoEvaluator`
2. Run the container


## Rebuild dataset: 
1. Upload expert's markup images to the directory */Dataset/Expert*, AI's markup — copy it to */Dataset/sample_1* and */Dataset/sample_2* and *Dataset/sample_3*. Names of images should have format: *someNumber_expert.png* for expert's markup and *somenumber_s1.png*, *somenumber_s2.png*, *somenumber_s3.png* for AI's markup. Note that *somenumber_s...* is the same AI's markup.
2. Open *notebooks/dataset_generation.ipynb*
3. Change in section *Import data*, line `data = pd.read_csv("DX_TEST_RESULT_FULL.csv")` **path with path to your csv** file with locations and sizes of markups
4. Delete code marked as *ONLY FOR GOOGLE COLAB*
5. Run all code in notebook
6. Your new csv dataset will be saved in */notebooks* directory as *wo_aug.csv*.


## Metrics tuning
Contribution ratio of every submodel in prediction of Evaluation model can be changed in *settings.py* via parameters `k1`, `k2`, `k3` and `k4`.
- A decrease in the value of the coefficient for a model means that the result of the prediction of this model will give a smaller, possibly even negative, contribution to the response of the system as a whole. 
- The increase of coefficient works the other way around.


| Coefficient | Corresponding submodel |
|---|---|
| `k1` | SVM |
| `k2` | CatBoost + VGG16 |
| `k3` | LGBM |
| `k4` | XGBoost |


## Data augmentation
The algorithm applies the same random affine changes to each i-th image in each subfolder in the dataset folder and save new images as PNG files in augmented dataset folder.

### Example:
| Before        | After |
|----------------|----------------------|
| <img src="https://user-images.githubusercontent.com/57181871/97780377-511f1280-1b95-11eb-8b8a-4ab65b0bd20e.png" alt="Translation menu screenshot" width="300"/> | <img src="https://user-images.githubusercontent.com/57181871/97780374-4e242200-1b95-11eb-8e4c-085071b41f00.png" alt="Main menu screenshot" width="300"/> |


### Run:
0. Install requirements.
1. Put the folders with the images that you want to augment into the Dataset folder(by default: "Dataset" folder). Each folder must have the same number of images.
2. Open Data_Augmentation.ipynb with program, which can execute ipynb files, for example, Jupiter Notebook.
3. Optionally, tune settings of transformations on images in "Settings" section.
4. Run all lines in notebook.
5. Augmented dataset will be placed in the same folder with script in new folder (by default: "AugmentedDataset" folder). Names of pictures have form: "originalName_numberInAlphabeticalOrder_iterationNumber".


### Requirements:
- Python 3+
- Requires PIL library


### Settings:

In "Settings" section can be tuned:
| Feature        | Variable |
|----------------|----------------------|
| Folder dataset | train_dir            |
| Output folder  | result_path          |
| Dataset increase multiplier  |  multiple_output_images  |
| Output images size    | img_width, img_height |

Affine transformations (maximum values of transformations):
| Feature        | Variable |
|----------------|----------------------|
| Rotation in degrees | rotation_range |
| Rotation ratio in degrees | rotation_range_multiple |
| Horizontal shift | width_shift_range |
| Vertical shift | height_shift_range |
| Zoom | zoom_range |
| Horizontal mirroring | horizontal_mirror |
| Vertical mirroring | vertical_mirror |
