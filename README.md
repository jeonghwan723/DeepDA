# DeepDA

The DeepDA model, which is a GAN model for oceanic DA, comprises a generator that implements partial convolution as well as a discriminator.

# Process of DeepDA

  1. Training with CESM2 Large Ensemble data set (csh/train.run.csh)
  2. Generate reanalysis (i.e., test) (csh/evaluate.run.csh)

# Data set (data format: netCDF4)

  - You can download data set here: in preparation..
  
  - The data set consists of the following:
  
    (1) Training set (CESM2 LE):
        
        Input: [land_file.nc,tr_data.nc,init_y.nc,clim.nc,obs_info.nc,random_error.nc]
        Label: [tr_data.nc]
        
    (2) Validation set (CESM2 LE):
        
        Input: [land_file.nc,tr_data.nc,val_data.nc,init_y.nc,clim.nc,obs_info.nc,random_error.nc]
        Label: [val_data.nc]
        
    (3) Test set (Met Office Hadley Centre Intergrated Ocean Database, HadIOD 1.2.0):
        
        Input: [land_file.nc,tr_data.nc,real_obs.nc,clim.nc,obs_info.nc]


# Requirement (python packages)

  -
