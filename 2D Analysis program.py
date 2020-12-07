import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import warnings
import h5py
import numpy as np
import napari
import time
import matplotlib
from skimage.transform import resize
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = [16, 12]

from skimage.morphology import remove_small_objects, binary_closing, disk, erosion, dilation   # function for post-processing (size filter)
from skimage import transform, measure
import h5py
from tqdm import tqdm
import pandas as pd
import warnings
from skimage.measure import label, regionprops
from skimage.segmentation import clear_border
import os, glob
from tqdm import tqdm
from colorama import Fore
from scipy.ndimage import gaussian_filter
from cellpose import models, io
#Copy gaussian smooth function from aics segemntation
def image_smoothing_gaussian_3d(struct_img, sigma, truncate_range=3.0):

    structure_img_smooth = gaussian_filter(struct_img, sigma=sigma, mode='nearest', truncate=truncate_range)

    return structure_img_smooth

#Ignore warnings issued by skimage through conversion to uint8
warnings.simplefilter("ignore",UserWarning)
warnings.simplefilter("ignore",RuntimeWarning)
warnings.simplefilter("ignore",FutureWarning)
# Use tkinter to interactively select files to import
root = tk.Tk()
root.withdraw()

directory = filedialog.askdirectory(initialdir=os.path.dirname(os.getcwd()))

print(directory)
# A class to perform segmentation, store segmentation attributes and conduct image statistical analysis
class Nucleus_2D_Segmentation:

    def __init__ (self, img_direct):
        self.dir = img_direct
        self.Raw_img_list = []
        self.MIP_img_list = []
        self.bg_img_list = []
        self.DOG_img_list = []
        self.flow_list = []
        self.mask_list_original = []
        self.mask_list_processed = []
        self.center_list = []
        self.intensity_list = []
        self.intensity_bg_list = []
        self.intensity_ratio_list = []
        
    #function to retrieve images and combine into a single list
    def img_list_extract (self):
        os.chdir(self.dir)
        Image_Files = glob.glob("*.hdf5")

        pb = tqdm(range(len(Image_Files)), bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.LIGHTCYAN_EX, Fore.RESET))

        for n in pb:
          pb.set_description ('Import Image Sequence')
          file = Image_Files[n]
          f = h5py.File(file, 'r')

          image = f['561 Channel'][:]
          
          MIP_Image = np.max(image, axis = 1)[0]

          self.Raw_img_list.append(image)
          self.MIP_img_list.append(MIP_Image)
        
        self.Raw_img_list = np.array(self.Raw_img_list)
        self.MIP_img_list = np.array(self.MIP_img_list)

    
    #function to do background substraction through difference of gaussian (DOG)
    def MIP_DOG (self,sigma1 = 20, sigma2 = 1):
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        
        pb = tqdm(range(self.MIP_img_list.shape[0]), bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.RED, Fore.RESET))
        
        for n in pb:
           pb.set_description ('Gaussian Smooth and Background Substraction') 
           img = self.Raw_img_list[n]

           img_bg = image_smoothing_gaussian_3d (img, sigma = self.sigma1)
           smooth_img = image_smoothing_gaussian_3d (img, sigma = self.sigma2 )

           DOG_img = smooth_img - img_bg
           
           img_bg_MIP = np.max(img_bg, axis = 1)[0]
           DOG_img_MIP = np.max(DOG_img, axis = 1)[0]

           self.bg_img_list.append(img_bg_MIP)
           self.DOG_img_list.append(DOG_img_MIP)

        self.bg_img_list = np.array(self.bg_img_list)

    # function to perform segmentation using Cellpose deep learning algorithm
    def cell_pose_segmentation (self):

       # DEFINE CELLPOSE MODEL
       # model_type='cyto' or model_type='nuclei'
       model = models.Cellpose(gpu=False, model_type='nuclei')

       # define CHANNELS to run segementation on
       # grayscale=0, R=1, G=2, B=3
       # channels = [cytoplasm, nucleus]
       # if NUCLEUS channel does not exist, set the second channel to 0
       # channels = [0,0]
       # IF ALL YOUR IMAGES ARE THE SAME TYPE, you can give a list with 2 elements
       # channels = [0,0] # IF YOU HAVE GRAYSCALE
       # channels = [2,3] # IF YOU HAVE G=cytoplasm and B=nucleus
       # channels = [2,1] # IF YOU HAVE G=cytoplasm and R=nucleus

       # or if you have different types of channels in each image
       channels = [[0,0]]*len(self.DOG_img_list)

       # if diameter is set to None, the size of the cells is estimated on a per image basis
       # you can set the average cell `diameter` in pixels yourself (recommended) 
       # diameter can be a list or a single number for all images
       masks, flows, styles, diams = model.eval(self.DOG_img_list, diameter=None, channels=channels)
       
       self.mask_list_original = masks

       for n in range(len(flows)):
          
          flow_original = flows[n][0]
          
          # Resize as the mask size
          img_shape = self.MIP_img_list[0].shape
          flow_resize = resize(flow_original, (img_shape[0], img_shape[1]))

          self.flow_list.append(flow_resize)
       
       #convert DOG_img_list and flow_list to numpy array
       self.DOG_img_list = np.array(self.DOG_img_list)
       self.flow_list = np.array(self.flow_list)
    
    # function to remove small objects and touching objects
    def remove_touch_small (self, minimum_size = 600):
       self.min_size = minimum_size
       pb = tqdm(range(len(self.mask_list_original)), bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.BLUE, Fore.RESET))


       for n in pb:
         pb.set_description ('Remove small object and touching object')   
         msk = self.mask_list_original[n]

         msk = clear_border(msk)
         msk = remove_small_objects(msk, min_size=self.min_size, connectivity=1, in_place=False)
         msk = label(msk)
  
         self.mask_list_processed.append(msk)

    # function to quantify membrane binding intensity ratio for 1 image
    def Intensity_Quantification_Single (self,n):
         

         msk = self.mask_list_processed [n]
         intensity_img = self.DOG_img_list [n]
            
         region = regionprops (msk)
         
         num = 1
         for r in region:
           nucleus_label = r.label
           img = msk == nucleus_label
    
           y = r.centroid[0]
           x = r.centroid[1]
    
           self.center_list.append((n,y,x))
           
           # Look at 4 pixel ring around the contour
           segmented_image_shell = np.logical_xor(dilation(img, selem=disk(2)),erosion(img, selem=disk(2)))
    
           intensity_median = np.median(intensity_img[segmented_image_shell==True])

           intensity_background = np.median(intensity_img[int(y),int(x)])
    
           self.intensity_list.append(intensity_median)
    
           self.intensity_bg_list.append(intensity_background)
    
           intensity_median_ratio = intensity_median / intensity_background
    
           self.intensity_ratio_list.append(intensity_median_ratio)
        
           self.segmented_image_shell_final[n,:,:] += segmented_image_shell * num
    
           num += 1

    # function to quantify membrane binding intensity ratio in multiple images
    def intensity_Quantification_Multiple (self):
        self.segmented_image_shell_final = np.zeros(np.array(self.mask_list_processed).shape)

        pb = tqdm(range(self.DOG_img_list.shape[0]), bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.MAGENTA, Fore.RESET))
          
        for n in pb:
          self.Intensity_Quantification_Single(n)
         
        self.mask_list_processed = np.array(self.mask_list_processed)

        self.intensity_list = np.array(self.intensity_list)
        self.intensity_bg_list = np.array(self.intensity_bg_list)
        self.intensity_ratio_list = np.array(self.intensity_ratio_list)
        self.center_list = np.array(self.center_list)

    

             

    

Nucleus_Segment = Nucleus_2D_Segmentation(directory)

#Extract all images and combine into 1 list
Nucleus_Segment.img_list_extract()

#Perform gaussian smooth and background substraction
Nucleus_Segment.MIP_DOG(sigma1=10,sigma2=1)

#Perform Cellpose segmentation
Nucleus_Segment.cell_pose_segmentation()

#Remove touching object and small object
#Nucleus_Segment.remove_touch_small(minimum_size= 600)

# Statistic Quantification
#Nucleus_Segment.intensity_Quantification_Multiple()

# Add labels with text to images
properties = {
    'Med': Nucleus_Segment.intensity_list,
    'bg': Nucleus_Segment.intensity_bg_list,
    'ratio': Nucleus_Segment.intensity_ratio_list
}

text_parameters = {
    'text': 'Int: {Med: .4f}\nBg: {bg:.4f}\nRatio: {ratio: .4f}',
    'size': 14,
    'color': 'white',
    'anchor': 'upper_right',
    'translation': [-5, 0]
}


# Open image and analysis result in Napari Veiwer. Do manual anotations in Napari viewer
with napari.gui_qt():
    viewer = napari.Viewer()
    viewer.add_image(Nucleus_Segment.MIP_img_list,name='Raw Image',colormap='red')
    viewer.add_image(Nucleus_Segment.bg_img_list,name='Background Image',colormap='red')
    viewer.add_image(Nucleus_Segment.DOG_img_list,name='Background Substracted Image',colormap='red')
    viewer.add_image(Nucleus_Segment.flow_list,name='Cell_Pose_Flow',rgb=True)
    viewer.add_labels(Nucleus_Segment.mask_list_processed,name='Nucleus Labels')
    viewer.add_labels(Nucleus_Segment.segmented_image_shell_final,name='Nucleus Contour Labels')

    pts = viewer.add_points(Nucleus_Segment.center_list,face_color = 'white',size=6,edge_color = 'blue',edge_width=2,name = 'Nucleus centroids with Stats',properties=properties,text=text_parameters)
    

# Save data into csv
df = pd.DataFrame(pts.properties)

print(df)

number = directory.split("/")[-1]
stats_save_name='{Folder_name}/{num}_analysis_result.csv'.format(Folder_name = directory, num = number)

df.to_csv(stats_save_name)