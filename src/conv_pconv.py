from tensorflow.keras import Model, datasets, layers, models, backend,activations,initializers

class Method  :
  def __init__(self) :
    super().__init__()

    self.act_relu = activations.relu
    self.act_tanh = activations.tanh
    self.act_none = activations.linear
    self.mask_int = initializers.Constant(1)


  def conv(self,X,n_feat,k_size,act,int_nor,stride,pad,dr_rate=0.0) :
 
    if act =='relu' :
      act_fun = self.act_relu
    elif act == 'tanh' :
      act_fun = self.act_tanh
    else :
      act_fun = self.act_none

    conv = layers.Conv3D(n_feat,k_size,activation=act_fun,data_format='channels_last',use_bias=True,
                           padding=pad,kernel_initializer=int_nor,strides=stride)(X)
    conv = layers.Dropout(dr_rate)(conv)
  
    return conv
  
  def pconv(self,X1,X2,n_feat,k_size,act,int_nor,stride,pad,dr_rate=0.0) :

    if act =='relu' :
      act_fun = self.act_relu
    elif act == 'tanh' :
      act_fun = self.act_tanh
    else :
      act_fun = self.act_none

    output = layers.Conv3D(n_feat,k_size,data_format='channels_last',use_bias=True,
                           padding=pad,kernel_initializer=int_nor,strides=stride)(X1)
  
    mask_output = layers.Conv3D(n_feat,k_size, data_format='channels_last',use_bias=False,
                             padding=pad,kernel_initializer=self.mask_int,strides=stride)(X2)
  
    window_size = k_size[0]*k_size[1]*k_size[2]
    mask_ratio = window_size/(mask_output+1e-8)
    mask_output = backend.clip(mask_output,0,1)
    mask_ratio = mask_ratio*mask_output

    output = output*mask_ratio
    output =  layers.Activation(act_fun)(output)

    return output, mask_output
  
  def deconv(self,X,n_feat,k_size,act,int_nor,stride,pad,dr_rate=0.0) :

    if act =='relu' :
      act_fun = self.act_relu
    elif act == 'tanh' :
      act_fun = self.act_tanh
    else :
      act_fun = self.act_none

    deconv = layers.Conv3DTranspose(n_feat,k_size,activation=act_fun,data_format='channels_last', use_bias=True,
                           padding=pad,kernel_initializer=int_nor,strides=stride)(X)
    deconv = layers.Dropout(dr_rate)(deconv)

    return deconv

  def re_block(self,X,n_feat,k_size,act,int_nor,stride,pad,dr_rate=0.0) :

    if act =='relu' :
      act_fun = self.act_relu
    elif act == 'tanh' :
      act_fun = self.act_tanh
    else :
      act_fun = self.act_none

    resnet = layers.Conv3D(n_feat,k_size,activation=act_fun,data_format='channels_last', use_bias=True,
                           padding=pad,kernel_initializer=int_nor,strides=stride)(X)
    resnet = layers.Dropout(dr_rate)(resnet)

    return resnet
  
  def pconv_re_block(self,X1,X2,n_feat,k_size,act,int_nor,stride,pad,dr_rate=0.0) :

    if act =='relu' :
      act_fun = self.act_relu
    elif act == 'tanh' :
      act_fun = self.act_tanh
    else :
      act_fun = self.act_none

    resnet,_ = self.pconv(X1,X2,n_feat,k_size,act_fun,int_nor,stride,pad)
  
    resnet = layers.Dropout(dr_rate)(resnet)
  
    return resnet
  
  def de_dense(X,dr_rate=0.0) :

    dense = layers.Dense(xdim*ydim*zdim,activation=None,use_bias=True,
                         kernel_initializer=None)(X)
    dense = layers.Dropout(dr_rate)(de_conv)

    return dense



