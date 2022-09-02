from tensorflow.keras import Model, datasets, layers, models, backend,activations,initializers

class Method  :
  def __init__(self) :
    super().__init__()

#    activations.relu = activations.relu
#    activations.tanh = activations.tanh
#    activations.linear = activations.linear
#    self.mask_int = initializers.Constant(1)


  def conv(self,X,n_feat,k_size,act,int_nor,stride,pad,dr_rate=0.0) :
 
    if act =='relu' :
      act_fun = activations.relu
    elif act == 'tanh' :
      act_fun = activations.tanh
    else :
      act_fun = activations.linear

    out_conv = layers.Conv3D(n_feat,k_size,activation=act_fun,data_format='channels_last',use_bias=True,
                           padding=pad,kernel_initializer=int_nor,strides=stride)(X)
    out_conv = layers.Dropout(dr_rate)(out_conv)
  
    return out_conv
  
  def pconv(self,X1,X2,n_feat,k_size,act,int_nor,stride,pad,dr_rate=0.0) :
    mask_int = initializers.Constant(1)

    if act =='relu' :
      act_fun = activations.relu
    elif act == 'tanh' :
      act_fun = activations.tanh
    else :
      act_fun = activations.linear

    conv_data = layers.Conv3D(n_feat,k_size,data_format='channels_last',use_bias=True,
                           padding=pad,kernel_initializer=int_nor,strides=stride)(X1)
  
    obs_inf_data = layers.Conv3D(n_feat,k_size, data_format='channels_last',use_bias=False,
                             padding=pad,kernel_initializer=mask_int,strides=stride)(X2)
  
    window_size = k_size[0]*k_size[1]*k_size[2]
    mask_ratio = window_size/(obs_inf_data+1e-8)
    obs_inf_output = backend.clip(obs_inf_data,0,1)
    mask_ratio = mask_ratio*obs_inf_output

    pconv_output = conv_data*mask_ratio
    act_output =  layers.Activation(act_fun)(pconv_output)

    return act_output, obs_inf_output
  
  def deconv(self,X,n_feat,k_size,act,int_nor,stride,pad,dr_rate=0.0) :

    if act =='relu' :
      act_fun = activations.relu
    elif act == 'tanh' :
      act_fun = activations.tanh
    else :
      act_fun = activations.linear

    out_deconv = layers.Conv3DTranspose(n_feat,k_size,activation=act_fun,data_format='channels_last', use_bias=True,
                           padding=pad,kernel_initializer=int_nor,strides=stride)(X)
    out_deconv = layers.Dropout(dr_rate)(out_deconv)

    return out_deconv

  def re_block(self,X,n_feat,k_size,act,int_nor,stride,pad,dr_rate=0.0) :

    if act =='relu' :
      act_fun = activations.relu
    elif act == 'tanh' :
      act_fun = activations.tanh
    else :
      act_fun = activations.linear

    resnet = layers.Conv3D(n_feat,k_size,activation=act_fun,data_format='channels_last', use_bias=True,
                           padding=pad,kernel_initializer=int_nor,strides=stride)(X)
    resnet = layers.Dropout(dr_rate)(resnet)

    return resnet
  
  def pconv_re_block(self,X1,X2,n_feat,k_size,act,int_nor,stride,pad,dr_rate=0.0) :

    if act =='relu' :
      act_fun = activations.relu
    elif act == 'tanh' :
      act_fun = activations.tanh
    else :
      act_fun = activations.linear

    pconv_resnet,_ = self.pconv(X1,X2,n_feat,k_size,act_fun,int_nor,stride,pad)
  
    pconv_resnet = layers.Dropout(dr_rate)(pconv_resnet)
  
    return pconv_resnet
  
  def de_dense(X,dr_rate=0.0) :

    dense = layers.Dense(xdim*ydim*zdim,activation=None,use_bias=True,
                         kernel_initializer=None)(X)
    dense = layers.Dropout(dr_rate)(de_conv)

    return dense



