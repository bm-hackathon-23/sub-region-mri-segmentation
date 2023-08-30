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
    write_results: if true, after each run the results will be saved in the current directoy as a nifti image data
    
    """
    ants_seg = ants_segmentation(nifti_image_path = "Data/sym_avg_mri_2023_50mu.nii.gz",
                                 nifti_label_path = "Data/sym_pop_atlas_label_map_2023.nii.gz",
                                 offset = 10000,
                                 ID = 233,
                                 write_results = True
                            )
    
    
    print("Loading the dataset and making probability images from the current label ...")

    
    # atropos segmentation method from ants
    print("Atropos segmentation method is running ...")
    atropos_labels = ants_seg.atropos (priorweight=0.0)
    print("Atropos is DONE.")
    
    # prior_based segmentation method from ants
    print("Prior_based segmentation method is running ...")
    priorbased_labels = ants_seg.prior_based_segmentation(priorweight=0.0, mrf=0.1, iterations=20)
    print("Prior_based is DONE.")
    
    # fuzzy_spatial_cmeans segmentation method from ants
    print("Fuzzy_spatial_cmeans segmentation method is running ...")
    fuzzyspatial_labels = ants_seg.fuzzy_spatial_cmeans_segmentation(number_of_clusters=2)
    print("Fuzzy_spatial_cmeans is DONE")

    
    
if __name__ == "__main__":
         main()