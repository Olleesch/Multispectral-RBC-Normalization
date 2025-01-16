# Multispectral-RBC-Normalization

This repository contains code to perform normalization of multimodal and multispectral usntained blood smear microscopy images for malaria diagnostics using smooth surface approximations. The method is based on estimating smooth surfaces corresponding to background intensity and local average cell intensity magnitude. Look at the steps under "Setup" below and follow the notebook 'visualize_normalization_process.ipynb' to try out the method. 

To run the normalization pipeline on an entire dataset, setup the dataset paths according to the steps under "Setup" and run the code in 'main_pipeline.py'. 

## Data

The method has been tested on and tuned for two datasets of microscopy images in the course "Advanced Image Analysis" at Uppsala University. Unfortunately, public access to the data cannot be provided here. The pipeline should hopefully work on other images of the same type too, but depending on the spatial size of pixels in real life, parameters in the method may need to be adjusted. 

## Repository

The code is structured as follows: 

- main_pipeline.py: The main normalization pipeline to normalize a set of samples. 
- visualize_normalization_process.ipynb: A notebook for visualization of the normalization process and visual comparison of a few samples. 
- evaluate_normalization_method.ipynb: A notebook for quantitative evaluation of normalized images. Before running the evaluation notebook, normalized samples of the dataset(s) in question must have been saved, for example by running the main pipeline. 
- plot_example_sample.ipynb: A notebook for plotting example samples. 

Further, all utility functions are found in the directory utils/, as follows:
- component_analysis.py: Contains code to compute principal components of samples.
- data.py: Contains code to handle the data, including dataset classes and functions for pre-processing, scaling, and format-checking. 
- surface_estimation.py: Contains code to estimate surfaces for the normalization process, including constructing binary masks, fitting a polynomial background surface, and estimating a local average cell intensity magnitude surface with Gaussian process regression. 
- format_mat_dataset.py: Contains code to process a dataset stored in a .mat file to the expected .tiff format. 

## Setup

To run the code, a few setup steps are required. 
1. Install packages listed in 'environment.yml'. The code is written in python 3.11. 
2. Set dataset paths. This can be done in two ways. Either set the paths manually in the code when creating dataset objects, or create a file '.env' specifying the paths. This is my preferred method to seemlessly work on multiple devices without having to manually write over paths. The .env file is ignored by git. The .env must contain lines on the following form: 

DATASET_PATH_1 = 'C:/.../dataset1/'  
DATASET_PATH_2 = 'C:/.../dataset2/'  
...

Each dataset is expected to conform to a specific format. The dataset path specified should give a root directory in which the data can be found. In this directory, a subdirectory "img_raw" is expected, where the raw images are found. Three tiff images per sample are expected, corresponding to the reflectance, scattering, and transmittance mode, indicated by subscripts "_R.tiff", "_S.tiff", "_T.tiff" respectively. 
