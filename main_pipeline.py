import os
import torch
import tifffile as tiff

from pathlib import Path
from dotenv import load_dotenv
from cellpose import models
from tqdm import tqdm

from utils.data import Dataset
from utils.surface_estimation import get_masks, fit_polynomial_background, fit_gpr_surface


def load_dataset_paths():
    """ Get dataset paths from .env file """
    load_dotenv()
    dataset_paths = []

    # Loop through environment variables and collect dataset paths
    for key, value in os.environ.items():
        if key.startswith("DATASET_PATH_"):  # Look for keys starting with "DATASET_PATH_"
            dataset_paths.append(Path(value.strip("'")))

    return dataset_paths


def main():
    # Parameter to write out details during processing
    verbose = True

    # Get dataset paths from .env file
    dataset_paths = load_dataset_paths()

    # Alternatively, manually write the correct paths in the following line: 
    # dataset_paths = [Path('C:/.../toy1/'), Path('C:/.../toy2/')]

    # Create dataset
    dataset = Dataset(dataset_paths)

    # Define segmentation model (here a pre-trained CellPose model)
    segmentation_model = models.Cellpose(model_type='cyto3', gpu=torch.cuda.is_available())
    
    def modify_path(path, new_dir, subscript):
        """ Modify path to a new parent dir and a new filename subscript """
        new_dir = path.parent.parent / new_dir
        return new_dir / f"{path.stem}_{subscript}{path.suffix}"
    
    def save_sample(paths, sample):
        """ Save reflectance, scattering, and transmittance modes of processed sample """
        dir = paths[0].parent
        dir.mkdir(parents=False, exist_ok=True)
        tiff.imwrite(paths[0], sample[0])
        tiff.imwrite(paths[1], sample[1])
        tiff.imwrite(paths[2], sample[2])

    # Process dataset, save background corrected and fully normalized samples
    # TODO: Rescale to some range when saving samples? 
    for data in tqdm(dataset):
        # Get sample and original path from dataset
        sample = data["sample"]
        paths = data["paths"]

        # Modify save path of bc and norm samples
        bc_paths = [modify_path(path, "img_background_corrected", "bc") for path in paths]
        norm_paths = [modify_path(path, "img_normalized", "norm") for path in paths]

        # If the bc and norm samples are not already saved to files, process sample
        if not (all(path.exists() for path in bc_paths) and all(path.exists() for path in norm_paths)):

            # Get binary masks
            background_mask, cell_mask = get_masks(sample, segmentation_model, plot=False, verbose=verbose)

            # Get background corrected sample
            bc_sample, _ = fit_polynomial_background(sample, background_mask, verbose=verbose)

            # Get normalized sample
            norm_sample, _ = fit_gpr_surface(bc_sample, cell_mask, verbose=verbose)

            # Save samples
            save_sample(bc_paths, bc_sample)
            save_sample(norm_paths, norm_sample)


if __name__=="__main__":
    main()