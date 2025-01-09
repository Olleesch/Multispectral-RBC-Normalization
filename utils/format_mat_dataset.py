import os
import tifffile as tiff
from pathlib import Path
from scipy.io import loadmat

dataset_path = Path(os.getenv('DATASET_PATH_3'))
data = loadmat(dataset_path / 'sample_Modele.mat')

image_R = data['Reflection'].transpose(2, 0, 1)
image_S = data['Scattering'].transpose(2, 0, 1)
image_T = data['Transmission'].transpose(2, 0, 1)

raw_dir = dataset_path / 'img_raw'
raw_dir.mkdir(exist_ok=True)

tiff.imwrite(raw_dir / 'sample_R.tiff', image_R)
tiff.imwrite(raw_dir / 'sample_S.tiff', image_S)
tiff.imwrite(raw_dir / 'sample_T.tiff', image_T)
