# DeepDA

The DeepDA model, which is a GAN model for oceanic DA, comprises a generator that implements partial convolution as well as a discriminator.

# Process of DeepDA

  1. Training with CESM2 Large Ensemble data set (csh/train.run.csh)
  2. Generate reanalysis (i.e., test) (csh/evaluate.run.csh)

# Data set (data format: netCDF4)

  - You can download data set here (36.2GB): https://168.131.122.201/OCL/Data/DeepDA/data.zip.
  
  - The data set consists of the following:
  
    (1) Training set (CESM2 LE):
    
        - Input
          1) Traing_Dataset_Masked_CESM2_LE_Ocean_Potential_Temperature_Anomaly.nc
          2) Observation_Mask.nc
          3) Land_Mask.nc        
        - Label
          1) Traing_Dataset_CESM2_LE_Ocean_Potential_Temperature_Anomaly.nc
       
    (2) Validation set (CESM2 LE):
    
        - Input
          1) Validation_Dataset_Masked_CESM2_LE_Ocean_Potential_Temperature_Anomaly.nc
          2) Observation_Mask.nc
          3) Land_mask.nc
        - Label
          1) Validation_Dataset_CESM2_LE_Ocean_Potential_Temperature_Anomaly.nc

    (3) Test set (Met Office Hadley Centre Intergrated Ocean Database, HadIOD 1.2.0):
    
        - Input
          1) HadIOD_version1.2.0.0_Ocean_Potential_Anomaly.nc
          2) Observation_Mask.nc
          3) Land_Mask.nc
        

# Requirement (our environment)
  - OS: Ubuntu 18.04.6 LTS
  - GPU: Nvidia Tesla V100
  - CUDA version: 11.4
  - python = 3.6
  - Tensowflow = 2.2.0 (https://www.tensorflow.org/install/)
  - netCDF4 = 1.5.3
  - numpy = 1.19.1

# License
This project is covered under the MIT License.
