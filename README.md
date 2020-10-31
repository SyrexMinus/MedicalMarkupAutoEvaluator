# MedicalMarkupAutoEvaluator
A solution for evaluating the quality of the markup of AI models on medical radiograms.

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
