# Multispectral-RBC-Normalization

This repository contains code to perform normalization of red blood cells based on estimating the background by fitting a polynomial plane. Look at the steps under "Setup" below, then follow the notebook 'rbc_background_normalization.ipynb' to try out the method. 

## Data

The method has been tested on "Dataset 1&2" in the studium announcement, where links to download the data can be found. The code should hopefully work on other images of the same type too, but it has not been tested on anything else yet. 

## Code

The code is structured as follows: 
- rbc_background_normalization.ipynb: A notebook visualizing the steps of the background normalization pipeline. This is the main code to run. 
- data.py: Contains a dataset class and scaling function for easier data handling. 
- component_analysis: Contains a class for principal component analysis of image channels. 
- normalize_background: Contains code to fit a polynomial background to an image. 

## Setup

To run the code, a few setup steps are required. 
1. Make sure packages listed in 'requirements.txt' are installed. 
2. Set dataset paths. This can be done in two ways. Either set the paths manually in the second code cell in rbc_background_normalization.ipynb, or create a file '.env' specifying the paths. This is my preferred method to seemlessly work on multiple devices without having to manually write over paths. The .env file is ignored by git. The .env must contain lines on the following form: 

DATASET_PATH_1 = 'C:/.../toy/'
DATASET_PATH_2 = 'C:/.../toy2/'
