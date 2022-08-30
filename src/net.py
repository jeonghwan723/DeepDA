from tensorflow.keras import Model, datasets, layers, models, backend, initializers
from conv_pconv import Method


class Net :
  def __init__(self) :
    super().__init__()

    self.relu_int_he = initializers.he_normal()
    self.tanh_int_gl = initializers.glorot_normal()
    self.relu_num = 16
    self.tanh_num = 8
    self.method = Method()


  def generator(self,X1,X2,X3,X4) :

    # input X
    conv1_1_1 = self.method.conv(X1,self.relu_num,[8,6,4],'relu',self.relu_int_he,[1,1,1],'same')
    conv1_1_2 = self.method.conv(X1,self.tanh_num,[8,6,4],'tanh',self.tanh_int_gl,[1,1,1],'same')
    cat1_1 = layers.concatenate([conv1_1_1,conv1_1_2],axis=4)
  
    # input M
    conv2_1_1, mconv2_1_1 = self.method.pconv(X2,X4,self.relu_num,[6,4,4],'relu',self.relu_int_he,[1,1,1],'same')
    conv2_1_2, mconv2_1_2 = self.method.pconv(X2,X4,self.tanh_num,[6,4,4],'tanh',self.tanh_int_gl,[1,1,1],'same')
    cat2_1 = layers.concatenate([conv2_1_1,conv2_1_2],axis=4)
    mcat2_1 = layers.concatenate([mconv2_1_1,conv2_1_2],axis=4)
  
    # input X  
    conv1_2_1 = self.method.conv(cat1_1,self.relu_num*2,[6,4,4],'relu',self.relu_int_he,[1,1,1],'same')
    max1_1_1 = layers.MaxPool3D((2,2,1),strides=[2,2,1],padding='same')(conv1_2_1)
    conv1_2_2 = self.method.conv(cat1_1,self.tanh_num*2,[6,4,4],'tanh',self.tanh_int_gl,[1,1,1],'same')
    max1_1_2 = layers.MaxPool3D((2,2,1),strides=[2,2,1],padding='same')(conv1_2_2)
    cat1_2 = layers.concatenate([max1_1_1,max1_1_2],axis=4)
  
    # input M
    conv2_2_1, mconv2_2_1 = self.method.pconv(cat2_1,mcat2_1,self.relu_num*2,[6,4,4],'relu',self.relu_int_he,[1,1,1],'same')
    max2_1_1 = layers.MaxPool3D((2,2,1),strides=[2,2,1],padding='same')(conv2_2_1)
    mmax2_1_1 = layers.MaxPool3D((2,2,1),strides=[2,2,1],padding='same')(mconv2_2_1)
  
    conv2_2_2, mconv2_2_2 = self.method.pconv(cat2_1,mcat2_1,self.tanh_num*2,[6,4,4],'tanh',self.tanh_int_gl,[1,1,1],'same')
    max2_1_2 = layers.MaxPool3D((2,2,1),strides=[2,2,1],padding='same')(conv2_2_2)
    mmax2_1_2 = layers.MaxPool3D((2,2,1),strides=[2,2,1],padding='same')(mconv2_2_2)
  
    cat2_2 = layers.concatenate([max2_1_1,max2_1_2],axis=4)
    mcat2_2 = layers.concatenate([mmax2_1_1,mmax2_1_2],axis=4)
  
    # input X 
    conv1_3_1 = self.method.conv(cat1_2,self.relu_num*4,[3,3,3],'relu',self.relu_int_he,[1,1,1],'same')
    max1_2_1 = layers.MaxPool3D((2,2,1),strides=[2,2,1],padding='same')(conv1_3_1)
    conv1_3_2 = self.method.conv(cat1_2,self.tanh_num*4,[3,3,3],'tanh',self.tanh_int_gl,[1,1,1],'same')
    max1_2_2 = layers.MaxPool3D((2,2,1),strides=[2,2,1],padding='same')(conv1_3_2)
    cat1_3 = layers.concatenate([max1_2_1,max1_2_2],axis=4)
  
    # input M 
    conv2_3_1,mconv2_3_1 = self.method.pconv(cat2_2,mcat2_2,self.relu_num*4,[3,3,3],'relu',self.relu_int_he,[1,1,1],'same')
    max2_2_1 = layers.MaxPool3D((2,2,1),strides=[2,2,1],padding='same')(conv2_3_1)
    mmax2_2_1 = layers.MaxPool3D((2,2,1),strides=[2,2,1],padding='same')(mconv2_3_1)
  
    conv2_3_2 ,mconv2_3_2 = self.method.pconv(cat2_2,mcat2_2,self.tanh_num*4,[3,3,3],'tanh',self.tanh_int_gl,[1,1,1],'same')
    max2_2_2 = layers.MaxPool3D((2,2,1),strides=[2,2,1],padding='same')(conv2_3_2)
    mmax2_2_2 = layers.MaxPool3D((2,2,1),strides=[2,2,1],padding='same')(mconv2_3_2)
  
    cat2_3 = layers.concatenate([max2_2_1,max2_2_2],axis=4)
    mcat2_3 = layers.concatenate([mmax2_2_1,mmax2_2_2],axis=4)
  
    # residual input X
    resnet_block1 = self.method.re_block(X1,(self.relu_num+self.tanh_num)*4,[4,4,4],'relu',self.relu_int_he,[4,4,1],'same')
  
    # residual input M
    resnet_block2 = self.method.pconv_re_block(X2,X4,(self.relu_num+self.tanh_num)*4,[4,4,4],'relu',self.relu_int_he,[4,4,1],'same')
  
    # input X
    en_last1 = cat1_3+resnet_block1
  
    # input M 
    en_last2 = cat2_3+resnet_block2
  
    en_last = layers.concatenate([en_last1,en_last2],axis=4)
  
    deconv1_1 = self.method.deconv(en_last,self.relu_num*2,[3,3,3],'relu',self.relu_int_he,[2,2,1],'same')
    deconv1_2 = self.method.deconv(en_last,self.tanh_num*2,[3,3,3],'tanh',self.tanh_int_gl,[2,2,1],'same')
    de_cat1 = layers.concatenate([deconv1_1,deconv1_2],axis=4)
  
    deconv2_1 = self.method.deconv(de_cat1,self.relu_num,[6,4,4],'relu',self.relu_int_he,[2,2,1],'same')
    deconv2_2 = self.method.deconv(de_cat1,self.tanh_num,[6,4,4],'tanh',self.tanh_int_gl,[2,2,1],'same')
    de_cat2 = layers.concatenate([deconv2_1,deconv2_2],axis=4)
  
    deconv3_1 = self.method.deconv(de_cat2,1,[8,6,4],'relu',self.relu_int_he,[1,1,1],'same')
    deconv3_2 = self.method.deconv(de_cat2,1,[8,6,4],'tanh',self.tanh_int_gl,[1,1,1],'same')
    de_cat3 = layers.concatenate([deconv3_1,deconv3_2],axis=4)
  
    output = self.method.deconv(de_cat3,1,[1,1,1],'linear',self.tanh_int_gl,[1,1,1],'same')
    output = output*X3
  
    return output
  
  
  def discriminator(self,X) :

    conv1_1 = self.method.conv(X,self.relu_num,[8,4,4],'relu',self.relu_int_he,[2,2,2],'same')
    conv1_2 = self.method.conv(X,self.tanh_num,[8,4,4],'tanh',self.tanh_int_gl,[2,2,2],'same')
    cat1 = layers.concatenate([conv1_1,conv1_2],axis=4)
  
  
    conv2_1 = self.method.conv(cat1,self.relu_num*2,[6,6,3],'relu',self.relu_int_he,[2,2,2],'same')
    conv2_2 = self.method.conv(cat1,self.tanh_num*2,[6,6,3],'tanh',self.tanh_int_gl,[2,2,2],'same')
    cat2 = layers.concatenate([conv2_1,conv2_2],axis=4)
  
    conv3_1 = self.method.conv(cat2,self.relu_num*4,[4,4,3],'relu',self.relu_int_he,[2,2,1],'same')
    conv3_2 = self.method.conv(cat2,self.tanh_num*4,[4,4,3],'tanh',self.tanh_int_gl,[2,2,1],'same')
    cat3 = layers.concatenate([conv3_1,conv3_2],axis=4)
  
    conv4_1 = self.method.conv(cat3,self.relu_num*8,[3,3,3],'relu',self.relu_int_he,[2,2,1],'same')
    conv4_2 = self.method.conv(cat3,self.tanh_num*8,[3,3,3],'tanh',self.tanh_int_gl,[2,2,1],'same')
    cat4 = layers.concatenate([conv4_1,conv4_2],axis=4)
  
    conv5_1 = self.method.conv(cat4,1,[3,3,3],'relu',self.relu_int_he,[2,2,2],'same')
    conv5_2 = self.method.conv(cat4,1,[3,3,3],'tanh',self.tanh_int_gl,[2,2,2],'same')
    cat5 = layers.concatenate([conv5_1,conv5_2],axis=4)
    output = cat5
    return output



