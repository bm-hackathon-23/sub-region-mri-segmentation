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



class ants_segmentation:
    def __init__(self, nifti_image_path, nifti_label_path, offset, ID,write_results):
        self.nifti_image_path = nifti_image_path
        self.nifti_label_path = nifti_label_path
        self.offset = offset # the offset between the symmetric labels in the left and right hemispheres
        self.ID = ID # geting the roi number
        self.nifti_data = self.read_img_nifti()
        self.nifti_label = self.read_label_nifti()
        self.origin = self.nifti_data.origin
        self.direction = self.nifti_data.direction
        self.spacing = self.nifti_data.spacing
        # Getting ants labels, only for ROI; it will be used for initial probability images
        self.label_id = (self.nifti_label==self.ID) + (self.nifti_label==self.offset+self.ID) 
        self.mask_id = self.make_masks()
        self.probability_images = self.make_probability_images()
        self.write_results = write_results
        
    # read a nifti dataset using ants
    def read_img_nifti(self):
        return ants.image_read(self.nifti_image_path)
    # read corresponding labels to the nifti dataset using ants
    def read_label_nifti(self):
        return ants.image_read(self.nifti_label_path)
    
    # create masks with boxes bigger than the ROI (based on the coronal axis)
    def make_masks(self):
        LABELS_id_temp = self.label_id[:,:,:].copy()
        MASKS_id = np.zeros(LABELS_id_temp.shape)
        pad_size = 7
        for SLICE in range(LABELS_id_temp.shape[1]):
            msk = (LABELS_id_temp[:,SLICE,:]).astype(np.uint8)
            mask_show = np.zeros((msk.shape[0],msk.shape[1],3))
            mask_show[:,:,0]=msk
            contours, _ = cv2.findContours(msk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                x1, y1, x2, y2 = x, y, x + w, y + h
                cv2.rectangle(mask_show, (x1-pad_size, y1-pad_size), (x2+pad_size, y2+pad_size), (255, 0, 0), thickness=cv2.FILLED)
            msk= mask_show[:,:,0]
            MASKS_id[:,SLICE,:] = msk
        MASKS_id = ants.from_numpy(MASKS_id,origin=self.origin,spacing=self.spacing,direction=self.direction)
        return MASKS_id
    
    # create a list of two probability images, original label id and its inverted version
    def make_probability_images (self):
        LABELS_inverted = self.label_id[:,:,:].copy()
        LABELS_inverted =(cv2.bitwise_not(LABELS_inverted)/255).astype(np.uint8)
        LABELS_inverted = ants.from_numpy(LABELS_inverted,origin=self.origin,spacing=self.spacing,direction=self.direction)
        ProbabilityImages = [self.label_id, LABELS_inverted]
        return ProbabilityImages
    
    # Atropos Segmentation from ants
    def atropos(self,priorweight):
        seg_results = ants.atropos(a = self.nifti_data, m = '[0.2,1x1x1]', c = '[2,0]', i = self.probability_images, x = self.mask_id, priorweight=priorweight)
        seg_id = seg_results['segmentation']==1
        if self.write_results == True:
            nifti_image = seg_id.to_nibabel()
            nifti_filename = 'atropos_pw_'+str(priorweight)+'_ID_'+str(self.ID)+'.nii.gz'
            nifti_image.to_filename(nifti_filename)
        return seg_id
    
    # Prior Based Segmentation from ants
    def prior_based_segmentation(self, priorweight,mrf, iterations):
        seg_results = ants.prior_based_segmentation(self.nifti_data, self.probability_images, self.mask_id, priorweight, mrf, iterations)
        seg_id = seg_results['segmentation']==1
        if self.write_results ==True:
            nifti_image = seg_id.to_nibabel()
            nifti_filename = nifti_filename = 'priorbased_pw_'+str(priorweight)+'_ID_'+str(self.ID)+'.nii.gz'
            nifti_image.to_filename(nifti_filename)  
        return seg_id
            
    # Fuzzy Spatial Cmeans Segmentation with 2 clusters
    def fuzzy_spatial_cmeans_segmentation(self, number_of_clusters):
        seg_results = ants.fuzzy_spatial_cmeans_segmentation(self.nifti_data, self.mask_id, number_of_clusters=number_of_clusters)
        seg_id = seg_results["segmentation_image"]==2
        if self.write_results ==True:
            nifti_image = seg_id.to_nibabel()
            nifti_filename = nifti_filename = 'fuzzyspatial_nc_'+str(number_of_clusters)+'_ID_'+str(self.ID)+'.nii.gz'
            nifti_image.to_filename(nifti_filename)
        return seg_id
    
    
# SEGMENTATION USING SAM model with respect to an ROI using box prompts
class Sam_Batch:
    def __init__(self, nifti_img_path, nifti_label_path, offset, ID, device, sam_model_type, sam_checkpoint, mode_axis, write_results, debug):
        self.debug = debug
        self.write_results = write_results
        self.nifti_img_path = nifti_img_path
        self.nifti_label_path = nifti_label_path
        self.nifti_data = self.read_img_nifti()
        self.nifti_label = self.read_label_nifti()
        self.spacing, self.origin, self.direction, self.size = self.get_info_from_nifti()
        self.sagittal = self.size[0]
        self.coronal = self.size[1]
        self.axial = self.size[2]
        self.offset = offset # the offset between the symmetric labels in the left and right hemispheres 
        self.ID = ID # geting the roi number
        self.device = device
        self.sam_model_type = sam_model_type
        self.sam_checkpoint = sam_checkpoint
        self.mode_axis = mode_axis # set the slicing axis: sagitall, coronal, or axial
        self.SAM = self.make_sam_model()
        self.resize_transform = ResizeLongestSide(self.SAM.image_encoder.img_size)
        self.batched_input, self.existing_label_index = self.get_tourch_batched_input()
        self.out_put_array = self.prediction()
        self.results = self.from_array_to_nifti()

    
    # read a nifti dataset using SimpleITK
    def read_img_nifti(self):
        return sitk.ReadImage(self.nifti_img_path)
    
    # read corresponding labels to the nifti dataset using SimpleITK
    def read_label_nifti(self):
        return sitk.ReadImage(self.nifti_label_path)
    
    # get and print info from nifti data
    def get_info_from_nifti (self):
        spacing = self.nifti_data.GetSpacing()
        origin = self.nifti_data.GetOrigin()
        direction = self.nifti_data.GetDirection()
        size = self.nifti_data.GetSize()
        if self.debug:
            print("******************************************************")
            print("Spacing (voxel size): ", spacing)
            print("Origin: ", origin)
            print("Get the directions describing the volume's coordinate system: ", direction)
            print("Size of the atlas volume: ", size)
            print("******************************************************")
        return spacing,origin,direction,size
    
    # get box prompts around the roi regions (in the left and right hemispheres with respect to the given id number
    # returns a list of boxes, each box is a rectangle specified by two pixel coordinate (top left and bottom right pixels of the box)
    def get_box_prompts(self, msk):
        list_of_boxes = []
        mask_show = (((msk==self.ID) + (msk==self.ID+self.offset))*255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_show, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            x1, y1, x2, y2 = x, y, x + w, y + h
            list_of_boxes.append([x1-7, y1-7, x2+7, y2+7])
            if self.debug:
                cv2.rectangle(mask_show, (x1, y1), (x2, y2), (255, 0, 0), 2)
        if self.debug:
            cv2.imshow('Prompts', mask_show)
            cv2.waitKey(1000)
            cv2.destroyAllWindows()
        return list_of_boxes
    
    # making the sam model
    def make_sam_model (self):
        sam = sam_model_registry[self.sam_model_type](checkpoint=self.sam_checkpoint)
        sam.to(device=self.device)
        return sam
    
    # get the batched input data including the corresponding box prompts as torch for SAM model
    def get_tourch_batched_input(self): 
        def prepare_image(image):
            image = self.resize_transform.apply_image(image)
            image = torch.as_tensor(image, device=self.device) 
            return image.permute(2, 0, 1).contiguous()
        
        def adjust_gamma(image, gamma=1.0):
            # Build a lookup table mapping the pixel values [0, 255] to their adjusted gamma values
            invGamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

            # Apply gamma correction using the lookup table
            return cv2.LUT(image, table)
        
        def sharpen_image(image):
            # Define a sharpening filter
            sharpening_kernel = np.array([[-1, -1, -1],[-1, 9, -1],[-1, -1, -1]])
            # Apply the sharpening filter to the image using the filter2D function
            sharpened_image = cv2.filter2D(image, -1, sharpening_kernel)
            return  sharpened_image
        
        batched_input = []
        existing_label_index = []
        if self.mode_axis == "sagittal":
            print ("Getting batched inputs including images and corresponding box prompts for sagittial axis ... ")
            for SLICE in tqdm(range (self.sagittal)):
                msk = sitk.GetArrayFromImage(self.nifti_label)[:,:,SLICE]
                if True in ((msk==self.ID) + (msk==self.ID+self.offset)):
                    existing_label_index.append(SLICE)
                    img = sitk.GetArrayFromImage(self.nifti_data)[:,:,SLICE]
                    img = (255*(img.copy()-img.min())/(img.max()-img.min())).astype(np.uint8)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    box = torch.tensor(self.get_box_prompts(msk),device=self.device)
                    batched_input.append(
                    {'image':prepare_image(img),
                     'boxes':self.resize_transform.apply_boxes_torch(box, img.shape[:2]),
                     'original_size': img.shape[:2] 
                    })
            print ("Getting batched inputs is done. ")
        
        elif self.mode_axis == "coronal":
            print ("Getting batched inputs including images and corresponding box prompts for coronol axis ... ")
            for SLICE in tqdm(range (self.coronal)):
                msk = sitk.GetArrayFromImage(self.nifti_label)[:,SLICE,:]
                if True in ((msk==self.ID) + (msk==self.ID+self.offset)):
                    existing_label_index.append(SLICE)
                    img = sitk.GetArrayFromImage(self.nifti_data)[:,SLICE,:]
                    img = (255*(img.copy()-img.min())/(img.max()-img.min())).astype(np.uint8)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    box = torch.tensor(self.get_box_prompts(msk),device=self.device)
                    batched_input.append(
                    {'image':prepare_image(img),
                     'boxes':self.resize_transform.apply_boxes_torch(box, img.shape[:2]),
                     'original_size': img.shape[:2] 
                    })
            print ("Getting batched inputs is done. ")
                    
        elif self.mode_axis == "axial":
            print ("Getting batched inputs including images and corresponding box prompts for axial axis ... ")
            for SLICE in tqdm(range (self.axial)):
                msk = sitk.GetArrayFromImage(self.nifti_label)[SLICE,:,:]
                if True in ((msk==self.ID) + (msk==self.ID+self.offset)):
                    existing_label_index.append(SLICE)
                    img = sitk.GetArrayFromImage(self.nifti_data)[SLICE,:,:]
                    img = (255*(img.copy()-img.min())/(img.max()-img.min())).astype(np.uint8)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    box = torch.tensor(self.get_box_prompts(msk),device=self.device)
                    batched_input.append(
                    {'image':prepare_image(img),
                     'boxes':self.resize_transform.apply_boxes_torch(box, img.shape[:2]),
                     'original_size': img.shape[:2] 
                    })
            print ("Getting batched inputs is done. ")
        torch.cuda.empty_cache()
        gc.collect() 
        return batched_input, existing_label_index
    
    
    # predict using the initialized sam model using box prompts and batch processing.
    # number of images in each batch is 2 
    # return the predicted labels as a three dimensional array
    def prediction(self):
        print("Batch prediction using SAM; each having two image frames ...")
        out_put_array = np.zeros(sitk.GetArrayFromImage(self.nifti_label).shape, dtype = np.uint8)
        Num_Labels = len(self.batched_input)
        if Num_Labels%2 == 0:
            for n in tqdm(range(Num_Labels//2)):
                batched_output = self.SAM(self.batched_input[n*2:(n+1)*2], multimask_output=False)
                indices = self.existing_label_index[n*2:(n+1)*2] 
                for m in range(2):
                    temp = np.zeros(batched_output[m]['masks'][0][0].shape,dtype = bool)
                    number_of_boxes = batched_output[m]['masks'].shape[0]
                    for i in range(number_of_boxes):
                        temp += batched_output[m]['masks'].cpu()[i][0].numpy()
                    if self.mode_axis == "coronal":
                        out_put_array[:,indices[m],:] = temp.copy()
                    elif self.mode_axis == "sagittal":
                        out_put_array[:,:,indices[m]] = temp.copy()
                    elif self.mode_axis == "axial":
                        out_put_array[indices[m],:,:] = temp.copy()
                torch.cuda.empty_cache()
                gc.collect() 
            print("Batch prediction is done.")
        else:
            for n in tqdm(range(Num_Labels//2)):
                batched_output = self.SAM(self.batched_input[n*2:(n+1)*2], multimask_output=False)
                indices = self.existing_label_index[n*2:(n+1)*2] 
                for m in range(2):
                    temp = np.zeros(batched_output[m]['masks'][0][0].shape,dtype = bool)
                    number_of_boxes = batched_output[m]['masks'].shape[0]
                    for i in range(number_of_boxes):
                        temp += batched_output[m]['masks'].cpu()[i][0].numpy()
                    if self.mode_axis == "coronal":
                        out_put_array[:,indices[m],:] = temp.copy()
                    elif self.mode_axis == "sagittal":
                        out_put_array[:,:,indices[m]] = temp.copy()
                    elif self.mode_axis == "axial":
                        out_put_array[indices[m],:,:] = temp.copy()
                torch.cuda.empty_cache()
                gc.collect() 

            batched_output = self.SAM(self.batched_input[Num_Labels-2:Num_Labels], multimask_output=False)
            indices = self.existing_label_index[Num_Labels-2:Num_Labels] 
            for m in range(2):
                temp = np.zeros(batched_output[m]['masks'][0][0].shape,dtype = bool)
                number_of_boxes = batched_output[m]['masks'].shape[0]
                for i in range(number_of_boxes):
                    temp += batched_output[m]['masks'].cpu()[i][0].numpy()

                if self.mode_axis == "coronal":
                    out_put_array[:,indices[m],:] = temp.copy()
                elif self.mode_axis == "sagittal":
                    out_put_array[:,:,indices[m]] = temp.copy()
                elif self.mode_axis == "axial":
                    out_put_array[indices[m],:,:] = temp.copy()

            torch.cuda.empty_cache()
            gc.collect()
            print("Batch prediction is done.")
            
        return out_put_array
         
    # convert the SAM prediction to nifti image
    def from_array_to_nifti(self):
        result_image = sitk.GetImageFromArray(self.out_put_array)
        result_image.CopyInformation(self.nifti_data)
        if self.write_results:
            # write the image
            sitk.WriteImage(result_image, "SAM_labels_for_id_"+str(self.ID)+"_"+self.mode_axis+".nii.gz")
            print("Results saved.")
        return result_image
    