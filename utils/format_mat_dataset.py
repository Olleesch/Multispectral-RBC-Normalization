import os
import tifffile as tiff
from pathlib import Path
from scipy.io import loadmat

""" 
Short script to transform 'sample_Modele.mat' dataset 
to the expected format of the dataset class.
"""

def main():
    # Get dataset path from .env file
    dataset_path = Path(os.getenv('DATASET_PATH_3'))
    
    # Alternatively, manually write the correct paths in the following line: 
    # dataset_paths = Path('C:/.../dataset/')

    # Load data
    data = loadmat(dataset_path / 'sample_Modele.mat')
    image_R = data['Reflection'].transpose(2, 0, 1)
    image_S = data['Scattering'].transpose(2, 0, 1)
    image_T = data['Transmission'].transpose(2, 0, 1)

    # Create 'img_raw' dir
    raw_dir = dataset_path / 'img_raw'
    raw_dir.mkdir(exist_ok=True)

    # Write tiff images
    tiff.imwrite(raw_dir / 'sample_R.tiff', image_R)
    tiff.imwrite(raw_dir / 'sample_S.tiff', image_S)
    tiff.imwrite(raw_dir / 'sample_T.tiff', image_T)


if __name__=="__main__":
    main()