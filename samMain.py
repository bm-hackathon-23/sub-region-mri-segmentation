import os
import gc
import cv2
import ants
import torch
import tqdm
import pickle
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from tqdm import tqdm
import matplotlib.pyplot as plt
from segment_anything import SamPredictor, sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from segUtilities import ants_segmentation, Sam_Batch


def main():
    
    """
    nifti_image_path: is the path to the marmoset average mri dataset 
    nifti_label_path: is the current atlas label for the mri dataset
    offset: there is an offset of 10000 between the lebel values of the left and right brian hemispheres
    ID: is thw id or lable value of the ROI
    device: If GPU is available
    sam_model_type: is the SAM model type
    sam_checkpoint: is the address of the corresponding SAM pretrianed model
    mode_axis: is the selected axis for slicing ("axial", "coronal", "sagittal")
    write_results: if true, after each run the results will be saved in the current directoy as a nifti image data
    """
    
    # Due to our limited GPU capability, each batch will have two images
    
    
    sambatch = Sam_Batch(nifti_img_path = "Data/sym_avg_mri_2023_50mu.nii.gz",
                     nifti_label_path = "Data/sym_pop_atlas_label_map_2023.nii.gz" ,
                     offset = 10000,
                     ID = 233,
                     device = "cuda",
                     sam_model_type = "vit_h",
                     sam_checkpoint="sam_vit_h_4b8939.pth",
                     mode_axis = "coronal",
                     write_results = True,
                     debug = False)
    
if __name__ == "__main__":
         main()