import numpy as np
from tensorflow.keras import Model, datasets, layers, models, backend
from contextlib import redirect_stdout
import os
import math,time
from netCDF4 import Dataset
from module import Module


class Train(Module) :
  def __init__(self,xdim,ydim,zdim) :
    super(Train,self).__init__()

    self.xdim = xdim
    self.ydim = ydim
    self.zdim = zdim

    

  def train_model(self,opath,exp_name,epochs) :

    dis_shape,dis_model,gen_model,combined_model = self.model(self.xdim,self.ydim,self.zdim)

  
    real_label = np.ones((1,) + dis_shape)
    fake_label = np.zeros((1,) + dis_shape)

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
        gen_model.summary()

    with open(opath+'/output/'+exp_name+'/summary/dis_model_summary.md', 'w') as f:
      with redirect_stdout(f):
        dis_model.summary()

    with open(opath+'/output/'+exp_name+'/summary/combined_model_summary.md', 'w') as f:
      with redirect_stdout(f):
        combined_model.summary()


    tr_epoch = 0
    for epoch in range(epochs) :
      init_number = (np.random.choice(41-2,1) +1)*73 

      start = time.time()
      i = 0
      ii = i%73
      i0 = 72
      
      # Traing_Dataset_CESM2_LE_Ocean_Potential_Temperature_Anomaly.nc: Only Using Discriminator at 1st time
      #
      
      background_map = Dataset('data/Traing_Dataset_CESM2_LE_Ocean_Potential_Temperature_Anomaly.nc','r')
      background_map = background_map['t'][init_number+1,:self.zdim]
      background_map = np.ma.masked_values(background_map,-9.99e+08)
      background_map = background_map.filled(0)
      
      observation_map = Dataset('data/Traing_Dataset_Masked_CESM2_LE_Ocean_Potential_Temperature_Anomaly.nc','r')
      observation_map = observation_map['t'][i,:self.zdim]
      observation_map = np.ma.masked_values(observation_map,-9.99e+08)
      observation_map = observation_map.filled(0)
      
      label_map = Dataset('data/Traing_Dataset_CESM2_LE_Ocean_Potential_Temperature_Anomaly.nc','r')
      label_map = label_map['t'][i+1,:self.zdim]
      label_map = np.ma.masked_values(label_map,-9.99e+08)
      label_map = label_map.filled(0)
      
      initial_dis_map = Dataset('data/Traing_Dataset_CESM2_LE_Ocean_Potential_Temperature_Anomaly.nc','r')
      initial_dis_map = initial_dis_map['t'][0,:self.zdim]
      initial_dis_map = np.ma.masked_values(initial_dis_map,-9.99e+08)
      initial_dis_map = initial_dis_map.filled(0)
  
      # Observation_Mask: Only 0 or 1
      observation_mask = Dataset('data/Observation_Mask','r')
      observation_mask = obs_mask['t'][i,:self.zdim]
      observation_mask = obs_mask.filled(0)
      
      land = Dataset('data/Land_Mask.nc','r')
      land = land['t'][0,:self.zdim]
      land = land.filled(0)
  
      # reshape [time,zdim,ydim,xdim => time,zdim,ydim,xdim,1]
      land = land.reshape(-1,self.zdim,self.ydim,self.xdim,1)
      background_map = background_map.reshape(-1,self.zdim,self.ydim,self.xdim,1)
      observation_map = observation_map.reshape(-1,self.zdim,self.ydim,self.xdim,1)
      label_map = label_map.reshape(-1,self.zdim,self.ydim,self.xdim,1)
      initial_dis_map = initial_dis_map.reshape(-1,self.zdim,self.ydim,self.xdim,1)
      observation_mask = observation_mask.reshape(-1,self.zdim,self.ydim,self.xdim,1)

      # axes change [time,zdim,ydim,xdim,1 => time,xdim,ydim,zdim,1]
      land = np.swapaxes(land,1,3)
      background_map = np.swapaxes(background_map,1,3)
      observation_map = np.swapaxes(observation_map,1,3)
      label_map = np.swapaxes(label_map,1,3)
      initial_dis_map = np.swapaxes(initial_dis_map,1,3)
      observation_mask = np.swapaxes(observation_mask,1,3)
 
  
      input_x1 = background_map
      input_x2 = observation_map
      input_x3 = land
      input_x4 = observation_mask
      sample_x = np.append(input_x1,input_x2,axis=4)
      sample_x = np.append(sample_x,input_x3,axis=4)
      sample_x = np.append(sample_x,input_x4,axis=4)
  
      sample_y0 = initial_dis_map
      sample_y = label_map
  
      dis_real_sample = np.append(sample_y,sample_y0,axis=4)
      gen_map = gen_model.predict(sample_x)
      dis_fake_sample = np.append(gen_map,sample_x[:,:,:,:,:1],axis=4)
  
      # Train discriminator 
      dis_loss_real = dis_model.train_on_batch(dis_real_sample,real_label)
      dis_loss_fake = dis_model.train_on_batch(dis_fake_sample,fake_label)
      total_dis_loss = np.add(dis_loss_real,dis_loss_fake)
      # Train generator
      total_gen_loss = combined_model.train_on_batch([sample_x],[real_label,sample_y])
  
      gen_map = gen_model.predict(sample_x)

      print('batch:',0,'int:',init_number,gen_map[:,0,0,0,0],'dis:',total_dis_loss,'gen:',total_gen_loss)
      for i in range(1,total_batch) :
        ii = i%73
        i0 = (i-1)%73
        
        observation_map = Dataset('data/Traing_Dataset_Masked_CESM2_LE_Ocean_Potential_Temperature_Anomaly.nc','r')
        observation_map = observation_map['t'][i,:self.zdim]
        observation_map = np.ma.masked_values(observation_map,-9.99e+08)
        observation_map = observation_map.filled(0)
        

        label_map = Dataset('data/Traing_Dataset_CESM2_LE_Ocean_Potential_Temperature_Anomaly.nc','r')
        label_map = label_map['t'][i+1,:self.zdim]
        label_map = np.ma.masked_values(label_map,-9.99e+08)
        label_map = label_map.filled(0)
        
        initial_dis_map = Dataset('data/Traing_Dataset_CESM2_LE_Ocean_Potential_Temperature_Anomaly.nc','r')
        initial_dis_map = initial_dis_map['t'][i,:self.zdim]
        initial_dis_map = np.ma.masked_values(initial_dis_map,-9.99e+08)
        initial_dis_map = initial_dis_map.filled(0)
  
  
        observation_mask = Dataset('data/Observation_Mask','r')
        observation_mask = observation_mask['t'][i,:self.zdim]
        observation_mask = observation_mask.filled(0)
        

        
        land = Dataset('data/Land_Mask.nc','r')
        land = land['t'][0,:self.zdim]
        land = land.filled(0)
 
  
        # reshape [time,zdim,ydim,xdim => time,zdim,ydim,xdim,1]
    
        land = land.reshape(-1,self.zdim,self.ydim,self.xdim,1)
        observation_map = observation_map.reshape(-1,self.zdim,self.ydim,self.xdim,1)
        label_map = label_map.reshape(-1,self.zdim,self.ydim,self.xdim,1)
        initial_dis_map = initial_dis_map.reshape(-1,self.zdim,self.ydim,self.xdim,1)
        observation_mask = observation_mask.reshape(-1,self.zdim,self.ydim,self.xdim,1)
        
        # axes change [time,zdim,ydim,xdim,1 => time,xdim,ydim,zdim,1]
        land = np.swapaxes(land,1,3)
        observation_map = np.swapaxes(observation_map,1,3)
        label_map = np.swapaxes(label_map,1,3)
        initial_dis_map = np.swapaxes(initial_dis_map,1,3)
        observation_mask = np.swapaxes(observation_mask,1,3)
 
        input_x1 = gen_map
        input_x2 = observation_map
        input_x3 = land
        input_x4 = observation_mask
        sample_x = np.append(input_x1,input_x2,axis=4)
        sample_x = np.append(sample_x,input_x3,axis=4)
        sample_x = np.append(sample_x,input_x4,axis=4)
  
        sample_y0 = initial_dis_map
        sample_y = label_map


  
        dis_real_sample = np.append(sample_y,sample_y0,axis=4)
        gen_map = gen_model.predict(sample_x)
        dis_fake_sample = np.append(gen_map,sample_x[:,:,:,:,:1],axis=4)
  
        # Train discriminator
        dis_loss_real = dis_model.train_on_batch(dis_real_sample,real_label)
        dis_loss_fake = dis_model.train_on_batch(dis_fake_sample,fake_label)
        total_dis_loss = np.add(dis_loss_real,dis_loss_fake)
        # Train generator
        total_gen_loss = combined_model.train_on_batch([sample_x],[real_label,sample_y])
        gen_map = gen_model.predict(sample_x)
        if i%100 ==0 :
          print('batch:',i,gen_map[:,0,0,0,0],'dis:',total_dis_loss,'gen:',total_gen_loss)
  
  
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
      
      background_map = Dataset('data/Traing_Dataset_CESM2_LE_Ocean_Potential_Temperature_Anomaly.nc','r')
      background_map = background_map['t'][init_number+1,:self.zdim]
      background_map = np.ma.masked_values(background_map,-9.99e+08)
      background_map = background_map.filled(0)
      
      observation_map = Dataset('data/Traing_Dataset_Masked_CESM2_LE_Ocean_Potential_Temperature_Anomaly.nc','r')
      observation_map = observation_map['t'][i,:self.zdim]
      observation_map = np.ma.masked_values(observation_map,-9.99e+08)
      observation_map = observation_map.filled(0)
      
      label_map = Dataset('data/Traing_Dataset_CESM2_LE_Ocean_Potential_Temperature_Anomaly.nc','r')
      label_map = label_map['t'][i+1,:self.zdim]
      label_map = np.ma.masked_values(label_map,-9.99e+08)
      label_map = label_map.filled(0)
      
      initial_dis_map = Dataset('data/Traing_Dataset_CESM2_LE_Ocean_Potential_Temperature_Anomaly.nc','r')
      initial_dis_map = initial_dis_map['t'][0,:self.zdim]
      initial_dis_map = np.ma.masked_values(initial_dis_map,-9.99e+08)
      initial_dis_map = initial_dis_map.filled(0)
  
      # Observation_Mask: Only 0 or 1
      observation_mask = Dataset('data/Observation_Mask','r')
      observation_mask = obs_mask['t'][i,:self.zdim]
      observation_mask = obs_mask.filled(0)
      
      land = Dataset('data/Land_Mask.nc','r')
      land = land['t'][0,:self.zdim]
      land = land.filled(0)
  
      # reshape [time,zdim,ydim,xdim => time,zdim,ydim,xdim,1]
      land = land.reshape(-1,self.zdim,self.ydim,self.xdim,1)
      background_map = background_map.reshape(-1,self.zdim,self.ydim,self.xdim,1)
      observation_map = observation_map.reshape(-1,self.zdim,self.ydim,self.xdim,1)
      label_map = label_map.reshape(-1,self.zdim,self.ydim,self.xdim,1)
      initial_dis_map = initial_dis_map.reshape(-1,self.zdim,self.ydim,self.xdim,1)
      observation_mask = observation_mask.reshape(-1,self.zdim,self.ydim,self.xdim,1)

      # axes change [time,zdim,ydim,xdim,1 => time,xdim,ydim,zdim,1]
      land = np.swapaxes(land,1,3)
      background_map = np.swapaxes(background_map,1,3)
      observation_map = np.swapaxes(observation_map,1,3)
      label_map = np.swapaxes(label_map,1,3)
      initial_dis_map = np.swapaxes(initial_dis_map,1,3)
      observation_mask = np.swapaxes(observation_mask,1,3)
 
  
      input_x1 = background_map
      input_x2 = observation_map
      input_x3 = land
      input_x4 = observation_mask
      sample_x = np.append(input_x1,input_x2,axis=4)
      sample_x = np.append(sample_x,input_x3,axis=4)
      sample_x = np.append(sample_x,input_x4,axis=4)
  
      sample_y0 = initial_dis_map
      sample_y = label_map

      tr_dis_real_sample = np.append(sample_y,sample_y0,axis=4)
  
      tr_gen_map = gen_model.predict(sample_x)
      tr_dis_fake_sample = np.append(tr_gen_map,sample_x[:,:,:,:,:1],axis=4)
  
      # loss
      tr_dis_loss_real = dis_model.evaluate(tr_dis_real_sample,real_label,verbose=0)
      tr_dis_loss_fake = dis_model.evaluate(tr_dis_fake_sample,fake_label,verbose=0)
      tr_dis_loss = np.add(tr_dis_loss_real,tr_dis_loss_fake)
      tr_gen_loss = combined_model.evaluate(sample_x,[real_label,sample_y],verbose=0)
  
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
        observation_map = Dataset('data/Traing_Dataset_Masked_CESM2_LE_Ocean_Potential_Temperature_Anomaly.nc','r')
        observation_map = observation_map['t'][i,:self.zdim]
        observation_map = np.ma.masked_values(observation_map,-9.99e+08)
        observation_map = observation_map.filled(0)
        

        label_map = Dataset('data/Traing_Dataset_CESM2_LE_Ocean_Potential_Temperature_Anomaly.nc','r')
        label_map = label_map['t'][i+1,:self.zdim]
        label_map = np.ma.masked_values(label_map,-9.99e+08)
        label_map = label_map.filled(0)
        
        initial_dis_map = Dataset('data/Traing_Dataset_CESM2_LE_Ocean_Potential_Temperature_Anomaly.nc','r')
        initial_dis_map = initial_dis_map['t'][i,:self.zdim]
        initial_dis_map = np.ma.masked_values(initial_dis_map,-9.99e+08)
        initial_dis_map = initial_dis_map.filled(0)
  
  
        observation_mask = Dataset('data/Observation_Mask','r')
        observation_mask = observation_mask['t'][i,:self.zdim]
        observation_mask = observation_mask.filled(0)
        

        
        land = Dataset('data/Land_Mask.nc','r')
        land = land['t'][0,:self.zdim]
        land = land.filled(0)
 
  
        # reshape [time,zdim,ydim,xdim => time,zdim,ydim,xdim,1]
    
        land = land.reshape(-1,self.zdim,self.ydim,self.xdim,1)
        observation_map = observation_map.reshape(-1,self.zdim,self.ydim,self.xdim,1)
        label_map = label_map.reshape(-1,self.zdim,self.ydim,self.xdim,1)
        initial_dis_map = initial_dis_map.reshape(-1,self.zdim,self.ydim,self.xdim,1)
        observation_mask = observation_mask.reshape(-1,self.zdim,self.ydim,self.xdim,1)
        
        # axes change [time,zdim,ydim,xdim,1 => time,xdim,ydim,zdim,1]
        land = np.swapaxes(land,1,3)
        observation_map = np.swapaxes(observation_map,1,3)
        label_map = np.swapaxes(label_map,1,3)
        initial_dis_map = np.swapaxes(initial_dis_map,1,3)
        observation_mask = np.swapaxes(observation_mask,1,3)
 
        input_x1 = tr_gen_map
        input_x2 = observation_map
        input_x3 = land
        input_x4 = observation_mask
        sample_x = np.append(input_x1,input_x2,axis=4)
        sample_x = np.append(sample_x,input_x3,axis=4)
        sample_x = np.append(sample_x,input_x4,axis=4)
  
        sample_y0 = initial_dis_map
        sample_y = label_map

        tr_dis_real_sample = np.append(sample_y,sample_y0,axis=4)
  
        tr_gen_map = gen_model.predict(sample_x)
        tr_dis_fake_sample = np.append(tr_gen_map,sample_x[:,:,:,:,:1],axis=4)
  
  
        # loss
        tr_dis_loss_real = dis_model.evaluate(tr_dis_real_sample,real_label,verbose=0)
        tr_dis_loss_fake = dis_model.evaluate(tr_dis_fake_sample,fake_label,verbose=0)
        tr_dis_loss = np.add(tr_dis_loss_real,tr_dis_loss_fake)
        tr_gen_loss = combined_model.evaluate(sample_x,[real_label,sample_y],verbose=0)
  
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
  
      background_map = Dataset('data/Traing_Dataset_CESM2_LE_Ocean_Potential_Temperature_Anomaly.nc','r')
      background_map = background_map['t'][init_number+1,:self.zdim]
      background_map = np.ma.masked_values(background_map,-9.99e+08)
      background_map = background_map.filled(0)
      
      observation_map = Dataset('data/Validation_Dataset_Masked_CESM2_LE_Ocean_Potential_Temperature_Anomaly.nc','r')
      observation_map = observation_map['t'][i,:self.zdim]
      observation_map = np.ma.masked_values(observation_map,-9.99e+08)
      observation_map = observation_map.filled(0)
      
      label_map = Dataset('data/Vadlidation_Dataset_CESM2_LE_Ocean_Potential_Temperature_Anomaly.nc','r')
      label_map = label_map['t'][i+1,:self.zdim]
      label_map = np.ma.masked_values(label_map,-9.99e+08)
      label_map = label_map.filled(0)
      
      initial_dis_map = Dataset('data/Vadlidation_Dataset_CESM2_LE_Ocean_Potential_Temperature_Anomaly.nc','r')
      initial_dis_map = initial_dis_map['t'][0,:self.zdim]
      initial_dis_map = np.ma.masked_values(initial_dis_map,-9.99e+08)
      initial_dis_map = initial_dis_map.filled(0)
  
      # Observation_Mask: Only 0 or 1
      observation_mask = Dataset('data/Observation_Mask','r')
      observation_mask = obs_mask['t'][i,:self.zdim]
      observation_mask = obs_mask.filled(0)
      
      land = Dataset('data/Land_Mask.nc','r')
      land = land['t'][0,:self.zdim]
      land = land.filled(0)
  
      # reshape [time,zdim,ydim,xdim => time,zdim,ydim,xdim,1]
      land = land.reshape(-1,self.zdim,self.ydim,self.xdim,1)
      background_map = background_map.reshape(-1,self.zdim,self.ydim,self.xdim,1)
      observation_map = observation_map.reshape(-1,self.zdim,self.ydim,self.xdim,1)
      label_map = label_map.reshape(-1,self.zdim,self.ydim,self.xdim,1)
      initial_dis_map = initial_dis_map.reshape(-1,self.zdim,self.ydim,self.xdim,1)
      observation_mask = observation_mask.reshape(-1,self.zdim,self.ydim,self.xdim,1)

      # axes change [time,zdim,ydim,xdim,1 => time,xdim,ydim,zdim,1]
      land = np.swapaxes(land,1,3)
      background_map = np.swapaxes(background_map,1,3)
      observation_map = np.swapaxes(observation_map,1,3)
      label_map = np.swapaxes(label_map,1,3)
      initial_dis_map = np.swapaxes(initial_dis_map,1,3)
      observation_mask = np.swapaxes(observation_mask,1,3)
 
  
      input_x1 = background_map
      input_x2 = observation_map
      input_x3 = land
      input_x4 = observation_mask
      sample_x = np.append(input_x1,input_x2,axis=4)
      sample_x = np.append(sample_x,input_x3,axis=4)
      sample_x = np.append(sample_x,input_x4,axis=4)
  
      sample_y0 = initial_dis_map
      sample_y = label_map

  
      val_dis_real_sample = np.append(sample_y,sample_y0,axis=4)
  
      val_gen_map = gen_model.predict(sample_x)
      val_dis_fake_sample = np.append(val_gen_map,sample_x[:,:,:,:,:1],axis=4)
  
      val_gen_map = gen_model.predict(sample_x)
      val_dis_loss_real = dis_model.evaluate(val_dis_real_sample,real_label,verbose=0)
      val_dis_loss_fake = dis_model.evaluate(val_dis_fake_sample,fake_label,verbose=0)
      val_dis_loss = np.add(val_dis_loss_real,val_dis_loss_fake)
      val_gen_loss = combined_model.evaluate(sample_x,[real_label,sample_y],verbose=0)
  
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
        
        observation_map = Dataset('data/Validation_Dataset_Masked_CESM2_LE_Ocean_Potential_Temperature_Anomaly.nc','r')
        observation_map = observation_map['t'][i,:self.zdim]
        observation_map = np.ma.masked_values(observation_map,-9.99e+08)
        observation_map = observation_map.filled(0)
        

        label_map = Dataset('data/Validation_Dataset_CESM2_LE_Ocean_Potential_Temperature_Anomaly.nc','r')
        label_map = label_map['t'][i+1,:self.zdim]
        label_map = np.ma.masked_values(label_map,-9.99e+08)
        label_map = label_map.filled(0)
        
        initial_dis_map = Dataset('data/Validation_Dataset_CESM2_LE_Ocean_Potential_Temperature_Anomaly.nc','r')
        initial_dis_map = initial_dis_map['t'][i,:self.zdim]
        initial_dis_map = np.ma.masked_values(initial_dis_map,-9.99e+08)
        initial_dis_map = initial_dis_map.filled(0)
  
  
        observation_mask = Dataset('data/Observation_Mask','r')
        observation_mask = observation_mask['t'][i,:self.zdim]
        observation_mask = observation_mask.filled(0)
        

        
        land = Dataset('data/Land_Mask.nc','r')
        land = land['t'][0,:self.zdim]
        land = land.filled(0)
 
  
        # reshape [time,zdim,ydim,xdim => time,zdim,ydim,xdim,1]
    
        land = land.reshape(-1,self.zdim,self.ydim,self.xdim,1)
        observation_map = observation_map.reshape(-1,self.zdim,self.ydim,self.xdim,1)
        label_map = label_map.reshape(-1,self.zdim,self.ydim,self.xdim,1)
        initial_dis_map = initial_dis_map.reshape(-1,self.zdim,self.ydim,self.xdim,1)
        observation_mask = observation_mask.reshape(-1,self.zdim,self.ydim,self.xdim,1)
        
        # axes change [time,zdim,ydim,xdim,1 => time,xdim,ydim,zdim,1]
        land = np.swapaxes(land,1,3)
        observation_map = np.swapaxes(observation_map,1,3)
        label_map = np.swapaxes(label_map,1,3)
        initial_dis_map = np.swapaxes(initial_dis_map,1,3)
        observation_mask = np.swapaxes(observation_mask,1,3)
 
        input_x1 = val_gen_map
        input_x2 = observation_map
        input_x3 = land
        input_x4 = observation_mask
        sample_x = np.append(input_x1,input_x2,axis=4)
        sample_x = np.append(sample_x,input_x3,axis=4)
        sample_x = np.append(sample_x,input_x4,axis=4)
  
        sample_y0 = initial_dis_map
        sample_y = label_map
        
      
  
        val_dis_real_sample = np.append(sample_y,sample_y0,axis=4)
  #      val_dis_real_sample = np.append(val_dis_real_sample,input_x3,axis=4)
  
        val_gen_map = gen_model.predict(sample_x)
        val_dis_fake_sample = np.append(val_gen_map,sample_x[:,:,:,:,:1],axis=4)
  #      val_dis_fake_sample = np.append(val_dis_fake_sample,input_x3,axis=4)
  
        val_dis_loss_real = dis_model.evaluate(val_dis_real_sample,real_label,verbose=0)
        val_dis_loss_fake = dis_model.evaluate(val_dis_fake_sample,fake_label,verbose=0)
        val_dis_loss = np.add(val_dis_loss_real,val_dis_loss_fake)
        val_gen_loss = combined_model.evaluate(sample_x,[real_label,sample_y],verbose=0)
  
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
  
  
      gen_model.save(opath+'/output/'+exp_name+'/model/last.model.hdf5')
  
      dis_model.save(opath+'/output/'+exp_name+'/model/last.dis.model.hdf5')
  
      if val_loss < best_val_loss :
        gen_model.save(opath+'/output/'+exp_name+'/model/best.model.hdf5')
        dis_model.save(opath+'/output/'+exp_name+'/model/best.dis.model.hdf5')
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
  
    gen_model.save(opath+'/output/'+exp_name+'/model/last.model.hdf5')

    dis_model.save(opath+'/output/'+exp_name+'/model/last.dis.model.hdf5')

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

    
