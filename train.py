import numpy as np
#import tensorflow as tf
from tensorflow.keras import Model, datasets, layers, models, backend
#from tensorflow import keras
from contextlib import redirect_stdout
import os
import math,time
from netCDF4 import Dataset
from module import Module


class Train :
  def __init__(self,xdim,ydim,zdim) :
    super().__init__()

    self.xdim = xdim
    self.ydim = ydim
    self.zdim = zdim

    self.module = Module()

    self.dis_shape = self.module.model(self.xdim,self.ydim,self.zdim)[0]

    self.dis_model = self.module.model(self.xdim,self.ydim,self.zdim)[1]
    self.gen_model = self.module.model(self.xdim,self.ydim,self.zdim)[2]
    self.combined_model = self.module.model(self.xdim,self.ydim,self.zdim)[3]
    

  def train_model(self,opath,exp_name,epochs) :

  
    real_label = np.ones((1,) + self.dis_shape)
    fake_label = np.zeros((1,) + self.dis_shape)
    trlen = 41*73
    vlen = 41*73
    total_batch = 41*73
  

    total_pa = 10
    pa = 0
    best_val_loss = 999

    tr_gen_loss1 = np.zeros((epochs))
    tr_gen_loss2 = np.zeros((epochs))
    tr_dis_loss0 = np.zeros((epochs))
    tr_dis_loss1 = np.zeros((epochs))
    tr_dis_loss2 = np.zeros((epochs))
  
    val_gen_loss1 = np.zeros((epochs))
    val_gen_loss2 = np.zeros((epochs))
    val_dis_loss0 = np.zeros((epochs))
    val_dis_loss1 = np.zeros((epochs))
    val_dis_loss2 = np.zeros((epochs))
  
  
    re_tr_gen_loss1 = np.zeros((epochs,trlen))
    re_tr_gen_loss2 = np.zeros((epochs,trlen))
    re_tr_dis_loss0 = np.zeros((epochs,trlen))
    re_tr_dis_loss1 = np.zeros((epochs,trlen))
    re_tr_dis_loss2 = np.zeros((epochs,trlen))
  
  
    re_val_gen_loss1 = np.zeros((epochs,vlen))
    re_val_gen_loss2 = np.zeros((epochs,vlen))
    re_val_dis_loss0 = np.zeros((epochs,vlen))
    re_val_dis_loss1 = np.zeros((epochs,vlen))
    re_val_dis_loss2 = np.zeros((epochs,vlen))

    with open(opath+'/output/'+exp_name+'/summary/gen_model_summary.md', 'w') as f:
      with redirect_stdout(f):
        self.gen_model.summary()

    with open(opath+'/output/'+exp_name+'/summary/dis_model_summary.md', 'w') as f:
      with redirect_stdout(f):
        self.dis_model.summary()

    with open(opath+'/output/'+exp_name+'/summary/combined_model_summary.md', 'w') as f:
      with redirect_stdout(f):
        self.combined_model.summary()


    tr_epoch = 0
    for epoch in range(epochs) :
      init_number = (np.random.choice(41-2,1) +1)*73
  
      start = time.time()
      i = 0
      ii = i%73
      i0 = 72
  
      land = Dataset('/home/ysjoo/data_assimilation/CESM2/data.v3/v3.land_file.nc','r')
      land = land['t'][0,:self.zdim]
      land = land.filled(0)
      init_map = Dataset('/home/ysjoo/data_assimilation/CESM2/data.v3/tr_data.nc','r')
      init_map = init_map['t'][init_number,:self.zdim]
      init_map = np.ma.masked_values(init_map,-9.99e+08)
      init_map = init_map.filled(0)
      x_map = Dataset('/home/ysjoo/data_assimilation/CESM2/data.v3/tr_data.nc','r')
      x_map = x_map['t'][i,:self.zdim]
      x_map = np.ma.masked_values(x_map,-9.99e+08)
      x_map = x_map.filled(0)
      init_y = Dataset('/home/ysjoo/data_assimilation/CESM2/data.v3/v3.init_y.nc','r')
      init_y = init_y['t'][0,:self.zdim]
      init_y = np.ma.masked_values(init_y,-9.99e+08)
      init_y = init_y.filled(0)
  
      clim = Dataset('/home/ysjoo/data_assimilation/CESM2/data.v3/v3.clim.nc','r')
      clim = clim['t'][ii,:self.zdim]
      clim = np.ma.masked_values(clim,-9.99e+08)
      clim = clim.filled(0)
  
      init_clim = Dataset('/home/ysjoo/data_assimilation/CESM2/data.v3/v3.clim.nc','r')
      init_clim = init_clim['t'][i0,:self.zdim]
      init_clim = np.ma.masked_values(init_clim,-9.99e+08)
      init_clim = init_clim.filled(0)
  
  
      mask_info = Dataset('/home/ysjoo/data_assimilation/CESM2/new_data.v3/v3.obs_info.nc','r')
      mask_info = mask_info['t'][i,:self.zdim]
      mask_info = mask_info.filled(0)
      random_error = Dataset('/home/ysjoo/data_assimilation/CESM2/new_data.v3/v3.random_error.nc','r')
      random_error = random_error['t'][i,:self.zdim]
      random_error = random_error.filled(0)
  
  
      init_map =  init_map - clim
      x_map = x_map - clim
      init_y = init_y - init_clim
  
      # time,zdim,ydim,xdim
      land = land.reshape(-1,self.zdim,self.ydim,self.xdim,1)
      init_map = init_map.reshape(-1,self.zdim,self.ydim,self.xdim,1)
      x_map = x_map.reshape(-1,self.zdim,self.ydim,self.xdim,1)
      init_y = init_y.reshape(-1,self.zdim,self.ydim,self.xdim,1)
      mask_info = mask_info.reshape(-1,self.zdim,self.ydim,self.xdim,1)
      random_error = random_error.reshape(-1,self.zdim,self.ydim,self.xdim,1)
      # ==> time,xdim,ydim,zdim,1
      land = np.swapaxes(land,1,3)
      init_map = np.swapaxes(init_map,1,3)
      x_map = np.swapaxes(x_map,1,3)
      init_y = np.swapaxes(init_y,1,3)
      mask_info = np.swapaxes(mask_info,1,3)
      random_error = np.swapaxes(random_error,1,3)
  
      input_x1 = init_map
      input_x2 = x_map*mask_info + random_error
      input_x3 = land
      input_x4 = mask_info
      sample_x = np.append(input_x1,input_x2,axis=4)
      sample_x = np.append(sample_x,input_x3,axis=4)
      sample_x = np.append(sample_x,input_x4,axis=4)
  
      sample_y0 = init_y
      sample_y = x_map
  
      dis_real_sample = np.append(sample_y,sample_y0,axis=4)
      gen_map = self.gen_model.predict(sample_x)
      dis_fake_sample = np.append(gen_map,sample_x[:,:,:,:,:1],axis=4)
  
      # Train discriminator 
      dis_loss_real = self.dis_model.train_on_batch(dis_real_sample,real_label)
      dis_loss_fake = self.dis_model.train_on_batch(dis_fake_sample,fake_label)
      total_dis_loss = np.add(dis_loss_real,dis_loss_fake)
      # Train generator
      total_gen_loss = self.combined_model.train_on_batch([sample_x],[real_label,sample_y])
  
      gen_map = self.gen_model.predict(sample_x)
      for i in range(1,total_batch) :
  
        ii = i%73
        i0 = (i-1)%73
        land = Dataset('/home/ysjoo/data_assimilation/CESM2/data.v3/v3.land_file.nc','r')
        land = land['t'][0,:self.zdim]
        land = land.filled(0)
        x_map = Dataset('/home/ysjoo/data_assimilation/CESM2/data.v3/tr_data.nc','r')
        x_map = x_map['t'][i,:self.zdim]
        x_map = np.ma.masked_values(x_map,-9.99e+08)
        x_map = x_map.filled(0)
        init_y = Dataset('/home/ysjoo/data_assimilation/CESM2/data.v3/tr_data.nc','r')
        init_y = init_y['t'][i-1,:self.zdim]
        init_y = np.ma.masked_values(init_y,-9.99e+08)
        init_y = init_y.filled(0)
        clim = Dataset('/home/ysjoo/data_assimilation/CESM2/data.v3/v3.clim.nc','r')
        clim = clim['t'][ii,:self.zdim]
        clim = np.ma.masked_values(clim,-9.99e+08)
        clim = clim.filled(0)
  
        init_clim = Dataset('/home/ysjoo/data_assimilation/CESM2/data.v3/v3.clim.nc','r')
        init_clim = init_clim['t'][i0,:self.zdim]
        init_clim = np.ma.masked_values(init_clim,-9.99e+08)
        init_clim = init_clim.filled(0)
  
  
        mask_info = Dataset('/home/ysjoo/data_assimilation/CESM2/new_data.v3/v3.obs_info.nc','r')
        mask_info = mask_info['t'][i,:self.zdim]
        mask_info = mask_info.filled(0)
        random_error = Dataset('/home/ysjoo/data_assimilation/CESM2/new_data.v3/v3.random_error.nc','r')
        random_error = random_error['t'][i,:self.zdim]
        random_error = random_error.filled(0)
        x_map = x_map - clim
        init_y = init_y - init_clim
  
  
        # time,zdim,ydim,xdim
        land = land.reshape(-1,self.zdim,self.ydim,self.xdim,1)
        x_map = x_map.reshape(-1,self.zdim,self.ydim,self.xdim,1)
        init_y = init_y.reshape(-1,self.zdim,self.ydim,self.xdim,1)
        mask_info = mask_info.reshape(-1,self.zdim,self.ydim,self.xdim,1)
        random_error = random_error.reshape(-1,self.zdim,self.ydim,self.xdim,1)
        # ==> time,xdim,ydim,zdim,1
        land = np.swapaxes(land,1,3)
        x_map = np.swapaxes(x_map,1,3)
        init_y = np.swapaxes(init_y,1,3)
        mask_info = np.swapaxes(mask_info,1,3)
        random_error = np.swapaxes(random_error,1,3)
        input_x1 = gen_map
        input_x2 = x_map*mask_info + random_error
        input_x3 = land
        input_x4 = mask_info
        sample_x = np.append(input_x1,input_x2,axis=4)
        sample_x = np.append(sample_x,input_x3,axis=4)
        sample_x = np.append(sample_x,input_x4,axis=4)
        sample_y0 = init_y
        sample_y = x_map
  
        dis_real_sample = np.append(sample_y,sample_y0,axis=4)
        gen_map = self.gen_model.predict(sample_x)
        dis_fake_sample = np.append(gen_map,sample_x[:,:,:,:,:1],axis=4)
  
        # Train discriminator
        dis_loss_real = self.dis_model.train_on_batch(dis_real_sample,real_label)
        dis_loss_fake = self.dis_model.train_on_batch(dis_fake_sample,fake_label)
        total_dis_loss = np.add(dis_loss_real,dis_loss_fake)
        # Train generator
        total_gen_loss = self.combined_model.train_on_batch([sample_x],[real_label,sample_y])
        gen_map = self.gen_model.predict(sample_x)
  
  
  
      # epoch loss
      tr_total_dis_loss = 0
      tr_total_gen_loss0 = 0
      tr_total_gen_loss1 = 0
      tr_total_gen_loss2 = 0
      tr_total_dis_loss1 = 0
      tr_total_dis_loss2 = 0
  
  
      i = 0
      ii = i%73
      i0 = 72
      land = Dataset('/home/ysjoo/data_assimilation/CESM2/data.v3/v3.land_file.nc','r')
      land = land['t'][0,:self.zdim]
      land = land.filled(0)
      init_map = Dataset('/home/ysjoo/data_assimilation/CESM2/data.v3/tr_data.nc','r')
      init_map = init_map['t'][init_number,:self.zdim]
      init_map = np.ma.masked_values(init_map,-9.99e+08)
      init_map = init_map.filled(0)
      x_map = Dataset('/home/ysjoo/data_assimilation/CESM2/data.v3/tr_data.nc','r')
      x_map = x_map['t'][i,:self.zdim]
      x_map = np.ma.masked_values(x_map,-9.99e+08)
      x_map = x_map.filled(0)
      init_y = Dataset('/home/ysjoo/data_assimilation/CESM2/data.v3/v3.init_y.nc','r')
      init_y = init_y['t'][0,:self.zdim]
      init_y = np.ma.masked_values(init_y,-9.99e+08)
      init_y = init_y.filled(0)
  
      clim = Dataset('/home/ysjoo/data_assimilation/CESM2/data.v3/v3.clim.nc','r')
      clim = clim['t'][ii,:self.zdim]
      clim = np.ma.masked_values(clim,-9.99e+08)
      clim = clim.filled(0)
  
      init_clim = Dataset('/home/ysjoo/data_assimilation/CESM2/data.v3/v3.clim.nc','r')
      init_clim = init_clim['t'][i0,:self.zdim]
      init_clim = np.ma.masked_values(init_clim,-9.99e+08)
      init_clim = init_clim.filled(0)
  
  
  
  
  
      mask_info = Dataset('/home/ysjoo/data_assimilation/CESM2/new_data.v3/v3.obs_info.nc','r')
      mask_info = mask_info['t'][i,:self.zdim]
      mask_info = mask_info.filled(0)
      random_error = Dataset('/home/ysjoo/data_assimilation/CESM2/new_data.v3/v3.random_error.nc','r')
      random_error = random_error['t'][i,:self.zdim]
      random_error = random_error.filled(0)
  
      init_map =  init_map - clim
      x_map = x_map - clim
      init_y = init_y - init_clim
  
      # time,zdim,ydim,xdim
      land = land.reshape(-1,self.zdim,self.ydim,self.xdim,1)
      init_map = init_map.reshape(-1,self.zdim,self.ydim,self.xdim,1)
      x_map = x_map.reshape(-1,self.zdim,self.ydim,self.xdim,1)
      init_y = init_y.reshape(-1,self.zdim,self.ydim,self.xdim,1)
      mask_info = mask_info.reshape(-1,self.zdim,self.ydim,self.xdim,1)
      random_error = random_error.reshape(-1,self.zdim,self.ydim,self.xdim,1)
      # ==> time,xdim,ydim,zdim,1
      land = np.swapaxes(land,1,3)
      init_map = np.swapaxes(init_map,1,3)
      x_map = np.swapaxes(x_map,1,3)
      init_y = np.swapaxes(init_y,1,3)
      mask_info = np.swapaxes(mask_info,1,3)
      random_error = np.swapaxes(random_error,1,3)
  
      input_x1 = init_map
      input_x2 = x_map*mask_info + random_error
      input_x3 = land
      input_x4 = mask_info
      sample_x = np.append(input_x1,input_x2,axis=4)
      sample_x = np.append(sample_x,input_x3,axis=4)
      sample_x = np.append(sample_x,input_x4,axis=4)
      sample_y0 = init_y
      sample_y = x_map
  
      tr_dis_real_sample = np.append(sample_y,sample_y0,axis=4)
  
      tr_gen_map = self.gen_model.predict(sample_x)
      tr_dis_fake_sample = np.append(tr_gen_map,sample_x[:,:,:,:,:1],axis=4)
  
      # loss
      tr_dis_loss_real = self.dis_model.evaluate(tr_dis_real_sample,real_label,verbose=0)
      tr_dis_loss_fake = self.dis_model.evaluate(tr_dis_fake_sample,fake_label,verbose=0)
      tr_dis_loss = np.add(tr_dis_loss_real,tr_dis_loss_fake)
      tr_gen_loss = self.combined_model.evaluate(sample_x,[real_label,sample_y],verbose=0)
  
      tr_total_dis_loss1 += tr_dis_loss_real
      tr_total_dis_loss2 += tr_dis_loss_fake
      tr_total_dis_loss += tr_dis_loss
      tr_total_gen_loss0 += tr_gen_loss[0]
      tr_total_gen_loss1 += tr_gen_loss[1]
      tr_total_gen_loss2 += tr_gen_loss[2]
  
      re_tr_dis_loss1[epoch,i] = tr_dis_loss_real
      re_tr_dis_loss2[epoch,i] = tr_dis_loss_fake
      re_tr_dis_loss0[epoch,i] = tr_dis_loss
      re_tr_gen_loss1[epoch,i] = tr_gen_loss[1]
      re_tr_gen_loss2[epoch,i] = tr_gen_loss[2]
  
  
  
      for i in range(1,trlen) :
  
        ii = i%73
        i0 = (i-1)%73
        land = Dataset('/home/ysjoo/data_assimilation/CESM2/data.v3/v3.land_file.nc','r')
        land = land['t'][0,:self.zdim]
        land = land.filled(0)
        x_map = Dataset('/home/ysjoo/data_assimilation/CESM2/data.v3/tr_data.nc','r')
        x_map = x_map['t'][i,:self.zdim]
        x_map = np.ma.masked_values(x_map,-9.99e+08)
        x_map = x_map.filled(0)
        init_y = Dataset('/home/ysjoo/data_assimilation/CESM2/data.v3/tr_data.nc','r')
        init_y = init_y['t'][i-1,:self.zdim]
        init_y = np.ma.masked_values(init_y,-9.99e+08)
        init_y = init_y.filled(0)
  
        clim = Dataset('/home/ysjoo/data_assimilation/CESM2/data.v3/v3.clim.nc','r')
        clim = clim['t'][ii,:self.zdim]
        clim = np.ma.masked_values(clim,-9.99e+08)
        clim = clim.filled(0)
  
        init_clim = Dataset('/home/ysjoo/data_assimilation/CESM2/data.v3/v3.clim.nc','r')
        init_clim = init_clim['t'][i0,:self.zdim]
        init_clim = np.ma.masked_values(init_clim,-9.99e+08)
        init_clim = init_clim.filled(0)
  
  
        mask_info = Dataset('/home/ysjoo/data_assimilation/CESM2/new_data.v3/v3.obs_info.nc','r')
        mask_info = mask_info['t'][i,:self.zdim]
        mask_info = mask_info.filled(0)
        random_error = Dataset('/home/ysjoo/data_assimilation/CESM2/new_data.v3/v3.random_error.nc','r')
        random_error = random_error['t'][i,:self.zdim]
        random_error = random_error.filled(0)
        x_map = x_map - clim
        init_y = init_y - init_clim
  
        # time,zdim,ydim,xdim
        land = land.reshape(-1,self.zdim,self.ydim,self.xdim,1)
        x_map = x_map.reshape(-1,self.zdim,self.ydim,self.xdim,1)
        init_y = init_y.reshape(-1,self.zdim,self.ydim,self.xdim,1)
        mask_info = mask_info.reshape(-1,self.zdim,self.ydim,self.xdim,1)
        random_error = random_error.reshape(-1,self.zdim,self.ydim,self.xdim,1)
        # ==> time,xdim,ydim,zdim,1
        land = np.swapaxes(land,1,3)
        x_map = np.swapaxes(x_map,1,3)
        init_y = np.swapaxes(init_y,1,3)
        mask_info = np.swapaxes(mask_info,1,3)
        random_error = np.swapaxes(random_error,1,3)
        input_x1 = tr_gen_map
  
  
  
        input_x2 = x_map*mask_info + random_error
        input_x3 = land
        input_x4 = mask_info
        sample_x = np.append(input_x1,input_x2,axis=4)
        sample_x = np.append(sample_x,input_x3,axis=4)
        sample_x = np.append(sample_x,input_x4,axis=4)
        sample_y0 = init_y
        sample_y = x_map
  
        tr_dis_real_sample = np.append(sample_y,sample_y0,axis=4)
  
        tr_gen_map = self.gen_model.predict(sample_x)
        tr_dis_fake_sample = np.append(tr_gen_map,sample_x[:,:,:,:,:1],axis=4)
  
  
        # loss
        tr_dis_loss_real = self.dis_model.evaluate(tr_dis_real_sample,real_label,verbose=0)
        tr_dis_loss_fake = self.dis_model.evaluate(tr_dis_fake_sample,fake_label,verbose=0)
        tr_dis_loss = np.add(tr_dis_loss_real,tr_dis_loss_fake)
        tr_gen_loss = self.combined_model.evaluate(sample_x,[real_label,sample_y],verbose=0)
  
        tr_total_dis_loss1 += tr_dis_loss_real
        tr_total_dis_loss2 += tr_dis_loss_fake
        tr_total_dis_loss += tr_dis_loss
        tr_total_gen_loss0 += tr_gen_loss[0]
        tr_total_gen_loss1 += tr_gen_loss[1]
        tr_total_gen_loss2 += tr_gen_loss[2]
  
        re_tr_dis_loss1[epoch,i] = tr_dis_loss_real
        re_tr_dis_loss2[epoch,i] = tr_dis_loss_fake
        re_tr_dis_loss0[epoch,i] = tr_dis_loss
        re_tr_gen_loss1[epoch,i] = tr_gen_loss[1]
        re_tr_gen_loss2[epoch,i] = tr_gen_loss[2]
  
  
      tr_loss = (tr_total_dis_loss + tr_total_gen_loss0)/trlen
  
      tr_dis_loss1[epoch] = (tr_total_dis_loss1/trlen)
      tr_dis_loss2[epoch] = (tr_total_dis_loss2/trlen)
      tr_dis_loss0[epoch] = (tr_total_dis_loss/trlen)
      tr_gen_loss1[epoch] = (tr_total_gen_loss1/trlen)
      tr_gen_loss2[epoch] = (tr_total_gen_loss2/trlen)
  
      val_total_dis_loss1 = 0
      val_total_dis_loss2 = 0
      val_total_dis_loss = 0
      val_total_gen_loss0 = 0
      val_total_gen_loss1 = 0
      val_total_gen_loss2 = 0
  
  
      i = 0
      ii = i%73
      i0 = 72
  
      land = Dataset('/home/ysjoo/data_assimilation/CESM2/data.v3/v3.land_file.nc','r')
      land = land['t'][0,:self.zdim]
      land = land.filled(0)
      init_map = Dataset('/home/ysjoo/data_assimilation/CESM2/data.v3/tr_data.nc','r')
      init_map = init_map['t'][init_number,:self.zdim]
      init_map = np.ma.masked_values(init_map,-9.99e+08)
      init_map = init_map.filled(0)
      x_map = Dataset('/home/ysjoo/data_assimilation/CESM2/data.v3/val_data.nc','r')
      x_map = x_map['t'][i,:self.zdim]
      x_map = np.ma.masked_values(x_map,-9.99e+08)
      x_map = x_map.filled(0)
      init_y = Dataset('/home/ysjoo/data_assimilation/CESM2/data.v3/tr_data.nc','r')
      init_y = init_y['t'][trlen-1,:self.zdim]
      init_y = np.ma.masked_values(init_y,-9.99e+08)
      init_y = init_y.filled(0)
  
      clim = Dataset('/home/ysjoo/data_assimilation/CESM2/data.v3/v3.clim.nc','r')
      clim = clim['t'][ii,:self.zdim]
      clim = np.ma.masked_values(clim,-9.99e+08)
      clim = clim.filled(0)
  
  
  
      init_clim = Dataset('/home/ysjoo/data_assimilation/CESM2/data.v3/v3.clim.nc','r')
      init_clim = init_clim['t'][i0,:self.zdim]
      init_clim = np.ma.masked_values(init_clim,-9.99e+08)
      init_clim = init_clim.filled(0)
  
  
      mask_info = Dataset('/home/ysjoo/data_assimilation/CESM2/new_data.v3/v3.obs_info.nc','r')
      mask_info = mask_info['t'][i,:self.zdim]
      mask_info = mask_info.filled(0)
      random_error = Dataset('/home/ysjoo/data_assimilation/CESM2/new_data.v3/v3.random_error.nc','r')
      random_error = random_error['t'][i,:self.zdim]
      random_error = random_error.filled(0)
  
      init_map =  init_map - clim
      x_map = x_map - clim
      init_y = init_y - init_clim
  
  
      # time,zdim,ydim,xdim
      land = land.reshape(-1,self.zdim,self.ydim,self.xdim,1)
      init_map = init_map.reshape(-1,self.zdim,self.ydim,self.xdim,1)
      x_map = x_map.reshape(-1,self.zdim,self.ydim,self.xdim,1)
      init_y = init_y.reshape(-1,self.zdim,self.ydim,self.xdim,1)
      mask_info = mask_info.reshape(-1,self.zdim,self.ydim,self.xdim,1)
      random_error = random_error.reshape(-1,self.zdim,self.ydim,self.xdim,1)
      # ==> time,xdim,ydim,zdim,1
      land = np.swapaxes(land,1,3)
      init_map = np.swapaxes(init_map,1,3)
      x_map = np.swapaxes(x_map,1,3)
      init_y = np.swapaxes(init_y,1,3)
      mask_info = np.swapaxes(mask_info,1,3)
      random_error = np.swapaxes(random_error,1,3)
  
  
      input_x1 = init_map
      input_x2 = x_map*mask_info + random_error
      input_x3 = land
      input_x4 = mask_info
      sample_x = np.append(input_x1,input_x2,axis=4)
      sample_x = np.append(sample_x,input_x3,axis=4)
      sample_x = np.append(sample_x,input_x4,axis=4)
      sample_y0 = init_y
      sample_y = x_map
  
      val_dis_real_sample = np.append(sample_y,sample_y0,axis=4)
  
      val_gen_map = self.gen_model.predict(sample_x)
      val_dis_fake_sample = np.append(val_gen_map,sample_x[:,:,:,:,:1],axis=4)
  
      val_gen_map = self.gen_model.predict(sample_x)
      val_dis_loss_real = self.dis_model.evaluate(val_dis_real_sample,real_label,verbose=0)
      val_dis_loss_fake = self.dis_model.evaluate(val_dis_fake_sample,fake_label,verbose=0)
      val_dis_loss = np.add(val_dis_loss_real,val_dis_loss_fake)
      val_gen_loss = self.combined_model.evaluate(sample_x,[real_label,sample_y],verbose=0)
  
      val_total_dis_loss1 += val_dis_loss_real
      val_total_dis_loss2 += val_dis_loss_fake
      val_total_dis_loss += val_dis_loss
      val_total_gen_loss0 += val_gen_loss[0]
      val_total_gen_loss1 += val_gen_loss[1]
      val_total_gen_loss2 += val_gen_loss[2]
      re_val_dis_loss0[epoch,i] = val_dis_loss
      re_val_gen_loss1[epoch,i] = val_gen_loss[1]
      re_val_gen_loss2[epoch,i] = val_gen_loss[2]
  
  
      for i in range(1,vlen) :
        ii = i%73
        i0 = (i-1)%73
        land = Dataset('/home/ysjoo/data_assimilation/CESM2/data.v3/v3.land_file.nc','r')
        land = land['t'][0,:self.zdim]
        land = land.filled(0)
        x_map = Dataset('/home/ysjoo/data_assimilation/CESM2/data.v3/val_data.nc','r')
        x_map = x_map['t'][i,:self.zdim]
        x_map = np.ma.masked_values(x_map,-9.99e+08)
        x_map = x_map.filled(0)
        init_y = Dataset('/home/ysjoo/data_assimilation/CESM2/data.v3/val_data.nc','r')
        init_y = init_y['t'][i-1,:self.zdim]
        init_y = np.ma.masked_values(init_y,-9.99e+08)
        init_y = init_y.filled(0)
  
        clim = Dataset('/home/ysjoo/data_assimilation/CESM2/data.v3/v3.clim.nc','r')
        clim = clim['t'][ii,:self.zdim]
        clim = np.ma.masked_values(clim,-9.99e+08)
        clim = clim.filled(0)
  
        init_clim = Dataset('/home/ysjoo/data_assimilation/CESM2/data.v3/v3.clim.nc','r')
        init_clim = init_clim['t'][i0,:self.zdim]
        init_clim = np.ma.masked_values(init_clim,-9.99e+08)
        init_clim = init_clim.filled(0)
  
  
        mask_info = Dataset('/home/ysjoo/data_assimilation/CESM2/new_data.v3/v3.obs_info.nc','r')
        mask_info = mask_info['t'][i,:self.zdim]
        mask_info = mask_info.filled(0)
        random_error = Dataset('/home/ysjoo/data_assimilation/CESM2/new_data.v3/v3.random_error.nc','r')
        random_error = random_error['t'][i,:self.zdim]
        random_error = random_error.filled(0)
        x_map = x_map - clim
        init_y = init_y - clim
  
  
        # time,zdim,ydim,xdim
        land = land.reshape(-1,self.zdim,self.ydim,self.xdim,1)
        x_map = x_map.reshape(-1,self.zdim,self.ydim,self.xdim,1)
        init_y = init_y.reshape(-1,self.zdim,self.ydim,self.xdim,1)
        mask_info = mask_info.reshape(-1,self.zdim,self.ydim,self.xdim,1)
        random_error = random_error.reshape(-1,self.zdim,self.ydim,self.xdim,1)
        # ==> time,xdim,ydim,zdim,1
        land = np.swapaxes(land,1,3)
        x_map = np.swapaxes(x_map,1,3)
        init_y = np.swapaxes(init_y,1,3)
        mask_info = np.swapaxes(mask_info,1,3)
        random_error = np.swapaxes(random_error,1,3)
        input_x1 = val_gen_map
        input_x2 = x_map*mask_info + random_error
        input_x3 = land
        input_x4 = mask_info
        sample_x = np.append(input_x1,input_x2,axis=4)
        sample_x = np.append(sample_x,input_x3,axis=4)
        sample_x = np.append(sample_x,input_x4,axis=4)
        sample_y0 = init_y
        sample_y = x_map
  
        val_dis_real_sample = np.append(sample_y,sample_y0,axis=4)
  #      val_dis_real_sample = np.append(val_dis_real_sample,input_x3,axis=4)
  
        val_gen_map = self.gen_model.predict(sample_x)
        val_dis_fake_sample = np.append(val_gen_map,sample_x[:,:,:,:,:1],axis=4)
  #      val_dis_fake_sample = np.append(val_dis_fake_sample,input_x3,axis=4)
  
        val_dis_loss_real = self.dis_model.evaluate(val_dis_real_sample,real_label,verbose=0)
        val_dis_loss_fake = self.dis_model.evaluate(val_dis_fake_sample,fake_label,verbose=0)
        val_dis_loss = np.add(val_dis_loss_real,val_dis_loss_fake)
        val_gen_loss = self.combined_model.evaluate(sample_x,[real_label,sample_y],verbose=0)
  
        val_total_dis_loss1 += val_dis_loss_real
        val_total_dis_loss2 += val_dis_loss_fake
        val_total_dis_loss += val_dis_loss
        val_total_gen_loss0 += val_gen_loss[0]
        val_total_gen_loss1 += val_gen_loss[1]
        val_total_gen_loss2 += val_gen_loss[2]
        re_val_dis_loss0[epoch,i] = val_dis_loss
        re_val_gen_loss1[epoch,i] = val_gen_loss[1]
        re_val_gen_loss2[epoch,i] = val_gen_loss[2]
  
  
      val_loss = (val_total_dis_loss + val_total_gen_loss0)/vlen
      val_dis_loss1[epoch] = (val_total_dis_loss1/vlen)
      val_dis_loss2[epoch] = (val_total_dis_loss2/vlen)
      val_dis_loss0[epoch] = (val_total_dis_loss/vlen)
      val_gen_loss1[epoch] = (val_total_gen_loss1/vlen)
      val_gen_loss2[epoch] = (val_total_gen_loss2/vlen)
  
  
      self.gen_model.save(opath+'/output/'+exp_name+'/model/last.model.hdf5')
  
      self.dis_model.save(opath+'/output/'+exp_name+'/model/last.dis.model.hdf5')
  
      if val_loss < best_val_loss :
        self.gen_model.save(opath+'/output/'+exp_name+'/model/best.model.hdf5')
        self.dis_model.save(opath+'/output/'+exp_name+'/model/best.dis.model.hdf5')
        best_val_loss = val_loss
        pa = 0
  
      else :
        pa = pa+1
      tr_epoch +=1
  
      print('time:',time.time() - start)
      print(epoch,'Epoch end ')
      print('tr_dis_Loss: ', tr_total_dis_loss1/trlen, tr_total_dis_loss2/trlen, 'tr_gen_loss: ', tr_total_gen_loss1/trlen,tr_total_gen_loss2/trlen)
      print('val_dis_Loss: ', val_total_dis_loss1/vlen, val_total_dis_loss2/vlen, 'val_gen_loss: ', val_total_gen_loss1/vlen,val_total_gen_loss2/vlen)
  
      if pa > total_pa :
        break
  
    self.gen_model.save(opath+'/output/'+exp_name+'/model/last.model.hdf5')

    self.dis_model.save(opath+'/output/'+exp_name+'/model/last.dis.model.hdf5')

    val_gen_loss1 = val_gen_loss1[:tr_epoch]
    val_gen_loss2 = val_gen_loss2[:tr_epoch]
    val_dis_loss1 = val_dis_loss1[:tr_epoch]
    val_dis_loss2 = val_dis_loss2[:tr_epoch]
    val_dis_loss0 = val_dis_loss0[:tr_epoch]
  
    val_total = val_gen_loss1+val_gen_loss2+val_dis_loss0
  
    tr_gen_loss1 = tr_gen_loss1[:tr_epoch]
    tr_gen_loss2 = tr_gen_loss2[:tr_epoch]
    tr_dis_loss0 = tr_dis_loss0[:tr_epoch]
    tr_dis_loss1 = tr_dis_loss1[:tr_epoch]
    tr_dis_loss2 = tr_dis_loss2[:tr_epoch]
  
    tr_total = tr_gen_loss1+tr_gen_loss2+tr_dis_loss0
  
  
    re_val_gen_loss1 = re_val_gen_loss1[:tr_epoch]
    re_val_gen_loss2 = re_val_gen_loss2[:tr_epoch]
    re_val_dis_loss0 = re_val_dis_loss0[:tr_epoch]
    re_val_dis_loss1 = re_val_dis_loss1[:tr_epoch]
    re_val_dis_loss2 = re_val_dis_loss2[:tr_epoch]
  
    re_val_total = re_val_gen_loss1+re_val_gen_loss2+re_val_dis_loss0
  
    re_tr_gen_loss1 = re_tr_gen_loss1[:tr_epoch]
    re_tr_gen_loss2 = re_tr_gen_loss2[:tr_epoch]
    re_tr_dis_loss0 = re_tr_dis_loss0[:tr_epoch]
    re_tr_dis_loss1 = re_tr_dis_loss1[:tr_epoch]
    re_tr_dis_loss2 = re_tr_dis_loss2[:tr_epoch]
  
  
  
    re_tr_total = re_tr_gen_loss1+re_tr_gen_loss2+re_tr_dis_loss0
  
    val_gen_loss1.astype('float32').tofile(opath+'/output/'+exp_name+'/loss/val_gen_loss1.gdat')
    ctl = open(opath+'/output/'+exp_name+'/loss/val_gen_loss1.ctl','w')
    ctl.write('dset ^val_gen_loss1.gdat\n')
    ctl.write('undef -9.99e+08\n')
    ctl.write('xdef   '+str(1)+'  linear 0.5  1.0\n')
    ctl.write('ydef   '+str(1)+'  linear -74.1667 0.3333309\n')
    ctl.write('zdef   '+str(1)+'  levels 5  15  25  35  45\n')
    ctl.write('tdef   '+str(tr_epoch)+'  linear jan1980 5dy\n')
    ctl.write('vars   1\n')
    ctl.write('p    '+str(1)+'   1  variable\n')
    ctl.write('ENDVARS\n')
    ctl.close()
  
    val_gen_loss2.astype('float32').tofile(opath+'/output/'+exp_name+'/loss/val_gen_loss2.gdat')
    ctl = open(opath+'/output/'+exp_name+'/loss/val_gen_loss2.ctl','w')
    ctl.write('dset ^val_gen_loss2.gdat\n')
    ctl.write('undef -9.99e+08\n')
    ctl.write('xdef   '+str(1)+'  linear 0.5  1.0\n')
    ctl.write('ydef   '+str(1)+'  linear -74.1667 0.3333309\n')
    ctl.write('zdef   '+str(1)+'  levels 5  15  25  35  45\n')
    ctl.write('tdef   '+str(tr_epoch)+'  linear jan1980 5dy\n')
    ctl.write('vars   1\n')
    ctl.write('p    '+str(1)+'   1  variable\n')
    ctl.write('ENDVARS\n')
    ctl.close()
  
  
    val_dis_loss0.astype('float32').tofile(opath+'/output/'+exp_name+'/loss/val_dis_loss.gdat')
    ctl = open(opath+'/output/'+exp_name+'/loss/val_dis_loss.ctl','w')
    ctl.write('dset ^val_dis_loss.gdat\n')
    ctl.write('undef -9.99e+08\n')
    ctl.write('xdef   '+str(1)+'  linear 0.5  1.0\n')
    ctl.write('ydef   '+str(1)+'  linear -74.1667 0.3333309\n')
    ctl.write('zdef   '+str(1)+'  levels 5  15  25  35  45\n')
    ctl.write('tdef   '+str(tr_epoch)+'  linear jan1980 5dy\n')
    ctl.write('vars   1\n')
    ctl.write('p    '+str(1)+'   1  variable\n')
    ctl.write('ENDVARS\n')
    ctl.close()
  
    val_dis_loss1.astype('float32').tofile(opath+'/output/'+exp_name+'/loss/val_dis_loss1.gdat')
    ctl = open(opath+'/output/'+exp_name+'/loss/val_dis_loss1.ctl','w')
    ctl.write('dset ^val_dis_loss1.gdat\n')
    ctl.write('undef -9.99e+08\n')
    ctl.write('xdef   '+str(1)+'  linear 0.5  1.0\n')
    ctl.write('ydef   '+str(1)+'  linear -74.1667 0.3333309\n')
    ctl.write('zdef   '+str(1)+'  levels 5  15  25  35  45\n')
    ctl.write('tdef   '+str(tr_epoch)+'  linear jan1980 5dy\n')
    ctl.write('vars   1\n')
    ctl.write('p    '+str(1)+'   1  variable\n')
    ctl.write('ENDVARS\n')
    ctl.close()
  
    val_dis_loss2.astype('float32').tofile(opath+'/output/'+exp_name+'/loss/val_dis_loss2.gdat')
    ctl = open(opath+'/output/'+exp_name+'/loss/val_dis_loss2.ctl','w')
    ctl.write('dset ^val_dis_loss2.gdat\n')
    ctl.write('undef -9.99e+08\n')
    ctl.write('xdef   '+str(1)+'  linear 0.5  1.0\n')
    ctl.write('ydef   '+str(1)+'  linear -74.1667 0.3333309\n')
    ctl.write('zdef   '+str(1)+'  levels 5  15  25  35  45\n')
    ctl.write('tdef   '+str(tr_epoch)+'  linear jan1980 5dy\n')
    ctl.write('vars   1\n')
    ctl.write('p    '+str(1)+'   1  variable\n')
    ctl.write('ENDVARS\n')
    ctl.close()
  
  
    val_total.astype('float32').tofile(opath+'/output/'+exp_name+'/loss/val_total.gdat')
    ctl = open(opath+'/output/'+exp_name+'/loss/val_total.ctl','w')
    ctl.write('dset ^val_total.gdat\n')
    ctl.write('undef -9.99e+08\n')
    ctl.write('xdef   '+str(1)+'  linear 0.5  1.0\n')
    ctl.write('ydef   '+str(1)+'  linear -74.1667 0.3333309\n')
    ctl.write('zdef   '+str(1)+'  levels 5  15  25  35  45\n')
    ctl.write('tdef   '+str(tr_epoch)+'  linear jan1980 5dy\n')
    ctl.write('vars   1\n')
    ctl.write('p    '+str(1)+'   1  variable\n')
    ctl.write('ENDVARS\n')
    ctl.close()
  
    re_val_gen_loss1.astype('float32').tofile(opath+'/output/'+exp_name+'/loss/total.val_gen_loss1.gdat')
    ctl = open(opath+'/output/'+exp_name+'/loss/total.val_gen_loss1.ctl','w')
    ctl.write('dset ^total.val_gen_loss1.gdat\n')
    ctl.write('undef -9.99e+08\n')
    ctl.write('xdef   '+str(1)+'  linear 0.5  1.0\n')
    ctl.write('ydef   '+str(1)+'  linear -74.1667 0.3333309\n')
    ctl.write('zdef   '+str(vlen)+'  linear 1 1\n')
    ctl.write('tdef   '+str(tr_epoch)+'  linear jan1980 5dy\n')
    ctl.write('vars   1\n')
    ctl.write('p    '+str(vlen)+'   1  variable\n')
    ctl.write('ENDVARS\n')
    ctl.close()
  
    re_val_gen_loss2.astype('float32').tofile(opath+'/output/'+exp_name+'/loss/total.val_gen_loss2.gdat')
    ctl = open(opath+'/output/'+exp_name+'/loss/total.val_gen_loss2.ctl','w')
    ctl.write('dset ^total.val_gen_loss2.gdat\n')
    ctl.write('undef -9.99e+08\n')
    ctl.write('xdef   '+str(1)+'  linear 0.5  1.0\n')
    ctl.write('ydef   '+str(1)+'  linear -74.1667 0.3333309\n')
    ctl.write('zdef   '+str(vlen)+'  linear 1 1\n')
    ctl.write('tdef   '+str(tr_epoch)+'  linear jan1980 5dy\n')
    ctl.write('vars   1\n')
    ctl.write('p    '+str(vlen)+'   1  variable\n')
    ctl.write('ENDVARS\n')
    ctl.close()
  
  
    re_val_dis_loss0.astype('float32').tofile(opath+'/output/'+exp_name+'/loss/total.val_dis_loss.gdat')
    ctl = open(opath+'/output/'+exp_name+'/loss/total.val_dis_loss.ctl','w')
    ctl.write('dset ^total.val_dis_loss.gdat\n')
    ctl.write('undef -9.99e+08\n')
    ctl.write('xdef   '+str(1)+'  linear 0.5  1.0\n')
    ctl.write('ydef   '+str(1)+'  linear -74.1667 0.3333309\n')
    ctl.write('zdef   '+str(vlen)+'  linear 1 1\n')
    ctl.write('tdef   '+str(tr_epoch)+'  linear jan1980 5dy\n')
    ctl.write('vars   1\n')
    ctl.write('p    '+str(vlen)+'   1  variable\n')
    ctl.write('ENDVARS\n')
    ctl.close()
  
    re_val_dis_loss1.astype('float32').tofile(opath+'/output/'+exp_name+'/loss/total.val_dis_loss1.gdat')
    ctl = open(opath+'/output/'+exp_name+'/loss/total.val_dis_loss1.ctl','w')
    ctl.write('dset ^total.val_dis_loss1.gdat\n')
    ctl.write('undef -9.99e+08\n')
    ctl.write('xdef   '+str(1)+'  linear 0.5  1.0\n')
    ctl.write('ydef   '+str(1)+'  linear -74.1667 0.3333309\n')
    ctl.write('zdef   '+str(vlen)+'  linear 1 1\n')
    ctl.write('tdef   '+str(tr_epoch)+'  linear jan1980 5dy\n')
    ctl.write('vars   1\n')
    ctl.write('p    '+str(vlen)+'   1  variable\n')
    ctl.write('ENDVARS\n')
    ctl.close()
  
    re_val_dis_loss2.astype('float32').tofile(opath+'/output/'+exp_name+'/loss/total.val_dis_loss2.gdat')
    ctl = open(opath+'/output/'+exp_name+'/loss/total.val_dis_loss2.ctl','w')
    ctl.write('dset ^total.val_dis_loss2.gdat\n')
    ctl.write('undef -9.99e+08\n')
    ctl.write('xdef   '+str(1)+'  linear 0.5  1.0\n')
    ctl.write('ydef   '+str(1)+'  linear -74.1667 0.3333309\n')
    ctl.write('zdef   '+str(vlen)+'  linear 1 1\n')
    ctl.write('tdef   '+str(tr_epoch)+'  linear jan1980 5dy\n')
    ctl.write('vars   1\n')
    ctl.write('p    '+str(vlen)+'   1  variable\n')
    ctl.write('ENDVARS\n')
    ctl.close()
  
  
    re_val_total.astype('float32').tofile(opath+'/output/'+exp_name+'/loss/total.val.total.gdat')
    ctl = open(opath+'/output/'+exp_name+'/loss/total.val.total.ctl','w')
    ctl.write('dset ^total.val.total.gdat\n')
    ctl.write('undef -9.99e+08\n')
    ctl.write('xdef   '+str(1)+'  linear 0.5  1.0\n')
    ctl.write('ydef   '+str(1)+'  linear -74.1667 0.3333309\n')
    ctl.write('zdef   '+str(vlen)+'  linear 1 1\n')
    ctl.write('tdef   '+str(tr_epoch)+'  linear jan1980 5dy\n')
    ctl.write('vars   1\n')
    ctl.write('p    '+str(vlen)+'   1  variable\n')
    ctl.write('ENDVARS\n')
    ctl.close()
  
  
  
    tr_gen_loss1.astype('float32').tofile(opath+'/output/'+exp_name+'/loss/tr_gen_loss1.gdat')
    ctl = open(opath+'/output/'+exp_name+'/loss/tr_gen_loss1.ctl','w')
    ctl.write('dset ^tr_gen_loss1.gdat\n')
    ctl.write('undef -9.99e+08\n')
    ctl.write('xdef   '+str(1)+'  linear 0.5  1.0\n')
    ctl.write('ydef   '+str(1)+'  linear -74.1667 0.3333309\n')
    ctl.write('zdef   '+str(1)+'  levels 5  15  25  35  45\n')
    ctl.write('tdef   '+str(tr_epoch)+'  linear jan1980 5dy\n')
    ctl.write('vars   1\n')
    ctl.write('p    '+str(1)+'   1  variable\n')
    ctl.write('ENDVARS\n')
    ctl.close()
  
    tr_gen_loss2.astype('float32').tofile(opath+'/output/'+exp_name+'/loss/tr_gen_loss2.gdat')
    ctl = open(opath+'/output/'+exp_name+'/loss/tr_gen_loss2.ctl','w')
    ctl.write('dset ^tr_gen_loss2.gdat\n')
    ctl.write('undef -9.99e+08\n')
    ctl.write('xdef   '+str(1)+'  linear 0.5  1.0\n')
    ctl.write('ydef   '+str(1)+'  linear -74.1667 0.3333309\n')
    ctl.write('zdef   '+str(1)+'  levels 5  15  25  35  45\n')
    ctl.write('tdef   '+str(tr_epoch)+'  linear jan1980 5dy\n')
    ctl.write('vars   1\n')
    ctl.write('p    '+str(1)+'   1  variable\n')
    ctl.write('ENDVARS\n')
    ctl.close()
  
  
    tr_dis_loss0.astype('float32').tofile(opath+'/output/'+exp_name+'/loss/tr_dis_loss.gdat')
    ctl = open(opath+'/output/'+exp_name+'/loss/tr_dis_loss.ctl','w')
    ctl.write('dset ^tr_dis_loss.gdat\n')
    ctl.write('undef -9.99e+08\n')
    ctl.write('xdef   '+str(1)+'  linear 0.5  1.0\n')
    ctl.write('ydef   '+str(1)+'  linear -74.1667 0.3333309\n')
    ctl.write('zdef   '+str(1)+'  levels 5  15  25  35  45\n')
    ctl.write('tdef   '+str(tr_epoch)+'  linear jan1980 5dy\n')
    ctl.write('vars   1\n')
    ctl.write('p    '+str(1)+'   1  variable\n')
    ctl.write('ENDVARS\n')
    ctl.close()
  
    tr_dis_loss1.astype('float32').tofile(opath+'/output/'+exp_name+'/loss/tr_dis_loss1.gdat')
    ctl = open(opath+'/output/'+exp_name+'/loss/tr_dis_loss1.ctl','w')
    ctl.write('dset ^tr_dis_loss1.gdat\n')
    ctl.write('undef -9.99e+08\n')
    ctl.write('xdef   '+str(1)+'  linear 0.5  1.0\n')
    ctl.write('ydef   '+str(1)+'  linear -74.1667 0.3333309\n')
    ctl.write('zdef   '+str(1)+'  levels 5  15  25  35  45\n')
    ctl.write('tdef   '+str(tr_epoch)+'  linear jan1980 5dy\n')
    ctl.write('vars   1\n')
    ctl.write('p    '+str(1)+'   1  variable\n')
    ctl.write('ENDVARS\n')
    ctl.close()
  
    tr_dis_loss2.astype('float32').tofile(opath+'/output/'+exp_name+'/loss/tr_dis_loss2.gdat')
    ctl = open(opath+'/output/'+exp_name+'/loss/tr_dis_loss2.ctl','w')
    ctl.write('dset ^tr_dis_loss2.gdat\n')
    ctl.write('undef -9.99e+08\n')
    ctl.write('xdef   '+str(1)+'  linear 0.5  1.0\n')
    ctl.write('ydef   '+str(1)+'  linear -74.1667 0.3333309\n')
    ctl.write('zdef   '+str(1)+'  levels 5  15  25  35  45\n')
    ctl.write('tdef   '+str(tr_epoch)+'  linear jan1980 5dy\n')
    ctl.write('vars   1\n')
    ctl.write('p    '+str(1)+'   1  variable\n')
    ctl.write('ENDVARS\n')
    ctl.close()
  
  
    tr_total.astype('float32').tofile(opath+'/output/'+exp_name+'/loss/tr_total.gdat')
    ctl = open(opath+'/output/'+exp_name+'/loss/tr_total.ctl','w')
    ctl.write('dset ^tr_total.gdat\n')
    ctl.write('undef -9.99e+08\n')
    ctl.write('xdef   '+str(1)+'  linear 0.5  1.0\n')
    ctl.write('ydef   '+str(1)+'  linear -74.1667 0.3333309\n')
    ctl.write('zdef   '+str(1)+'  levels 5  15  25  35  45\n')
    ctl.write('tdef   '+str(tr_epoch)+'  linear jan1980 5dy\n')
    ctl.write('vars   1\n')
    ctl.write('p    '+str(1)+'   1  variable\n')
    ctl.write('ENDVARS\n')
    ctl.close()
  
  
    re_tr_gen_loss1.astype('float32').tofile(opath+'/output/'+exp_name+'/loss/total.tr_gen_loss1.gdat')
    ctl = open(opath+'/output/'+exp_name+'/loss/total.tr_gen_loss1.ctl','w')
    ctl.write('dset ^total.tr_gen_loss1.gdat\n')
    ctl.write('undef -9.99e+08\n')
    ctl.write('xdef   '+str(1)+'  linear 0.5  1.0\n')
    ctl.write('ydef   '+str(1)+'  linear -74.1667 0.3333309\n')
    ctl.write('zdef   '+str(trlen)+'  linear 1 1\n')
    ctl.write('tdef   '+str(tr_epoch)+'  linear jan1980 5dy\n')
    ctl.write('vars   1\n')
    ctl.write('p    '+str(trlen)+'   1  variable\n')
    ctl.write('ENDVARS\n')
    ctl.close()
  
    re_tr_gen_loss2.astype('float32').tofile(opath+'/output/'+exp_name+'/loss/total.tr_gen_loss2.gdat')
    ctl = open(opath+'/output/'+exp_name+'/loss/total.tr_gen_loss2.ctl','w')
    ctl.write('dset ^total.tr_gen_loss2.gdat\n')
    ctl.write('undef -9.99e+08\n')
    ctl.write('xdef   '+str(1)+'  linear 0.5  1.0\n')
    ctl.write('ydef   '+str(1)+'  linear -74.1667 0.3333309\n')
    ctl.write('zdef   '+str(trlen)+'  linear 1 1\n')
    ctl.write('tdef   '+str(tr_epoch)+'  linear jan1980 5dy\n')
    ctl.write('vars   1\n')
    ctl.write('p    '+str(trlen)+'   1  variable\n')
    ctl.write('ENDVARS\n')
    ctl.close()
  
  
    re_tr_dis_loss0.astype('float32').tofile(opath+'/output/'+exp_name+'/loss/total.tr_dis_loss.gdat')
    ctl = open(opath+'/output/'+exp_name+'/loss/total.tr_dis_loss.ctl','w')
    ctl.write('dset ^total.tr_dis_loss.gdat\n')
    ctl.write('undef -9.99e+08\n')
    ctl.write('xdef   '+str(1)+'  linear 0.5  1.0\n')
    ctl.write('ydef   '+str(1)+'  linear -74.1667 0.3333309\n')
    ctl.write('zdef   '+str(trlen)+'  linear 1 1\n')
    ctl.write('tdef   '+str(tr_epoch)+'  linear jan1980 5dy\n')
    ctl.write('vars   1\n')
    ctl.write('p    '+str(trlen)+'   1  variable\n')
    ctl.write('ENDVARS\n')
    ctl.close()
  
    re_tr_dis_loss1.astype('float32').tofile(opath+'/output/'+exp_name+'/loss/total.tr_dis_loss1.gdat')
    ctl = open(opath+'/output/'+exp_name+'/loss/total.tr_dis_loss1.ctl','w')
    ctl.write('dset ^total.tr_dis_loss1.gdat\n')
    ctl.write('undef -9.99e+08\n')
    ctl.write('xdef   '+str(1)+'  linear 0.5  1.0\n')
    ctl.write('ydef   '+str(1)+'  linear -74.1667 0.3333309\n')
    ctl.write('zdef   '+str(trlen)+'  linear 1 1\n')
    ctl.write('tdef   '+str(tr_epoch)+'  linear jan1980 5dy\n')
    ctl.write('vars   1\n')
    ctl.write('p    '+str(trlen)+'   1  variable\n')
    ctl.write('ENDVARS\n')
    ctl.close()
  
    re_tr_dis_loss2.astype('float32').tofile(opath+'/output/'+exp_name+'/loss/total.tr_dis_loss2.gdat')
    ctl = open(opath+'/output/'+exp_name+'/loss/total.tr_dis_loss2.ctl','w')
    ctl.write('dset ^total.tr_dis_loss2.gdat\n')
    ctl.write('undef -9.99e+08\n')
    ctl.write('xdef   '+str(1)+'  linear 0.5  1.0\n')
    ctl.write('ydef   '+str(1)+'  linear -74.1667 0.3333309\n')
    ctl.write('zdef   '+str(trlen)+'  linear 1 1\n')
    ctl.write('tdef   '+str(tr_epoch)+'  linear jan1980 5dy\n')
    ctl.write('vars   1\n')
    ctl.write('p    '+str(trlen)+'   1  variable\n')
    ctl.write('ENDVARS\n')
    ctl.close()
  
  
    re_tr_total.astype('float32').tofile(opath+'/output/'+exp_name+'/loss/total.tr.total.gdat')
    ctl = open(opath+'/output/'+exp_name+'/loss/total.tr.total.ctl','w')
    ctl.write('dset ^total.tr.total.gdat\n')
    ctl.write('undef -9.99e+08\n')
    ctl.write('xdef   '+str(1)+'  linear 0.5  1.0\n')
    ctl.write('ydef   '+str(1)+'  linear -74.1667 0.3333309\n')
    ctl.write('zdef   '+str(trlen)+'  linear 1 1\n')
    ctl.write('tdef   '+str(tr_epoch)+'  linear jan1980 5dy\n')
    ctl.write('vars   1\n')
    ctl.write('p    '+str(trlen)+'   1  variable\n')
    ctl.write('ENDVARS\n')
    ctl.close()

    
