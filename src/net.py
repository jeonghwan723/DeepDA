from tensorflow.keras import Model, datasets, layers, models, backend, initializers
from conv_pconv import Method


class Net(Method) :
  def __init__(self) :
    super(Net,self).__init__()


  def generator(self,X1,X2,X3,X4) :

    relu_int_he = initializers.he_normal()
    tanh_int_gl = initializers.glorot_normal()
    relu_num = 16
    tanh_num = 8
    mask_int = initializers.Constant(1)

    # input X
    gen_conv1_1_1 = self.conv(X1,relu_num,[8,6,4],'relu',relu_int_he,[1,1,1],'same')
    gen_conv1_1_2 = self.conv(X1,tanh_num,[8,6,4],'tanh',tanh_int_gl,[1,1,1],'same')
    gen_cat1_1 = layers.concatenate([gen_conv1_1_1,gen_conv1_1_2],axis=4)
  
    # input M
    gen_conv2_1_1, gen_mconv2_1_1 = self.pconv(X2,X4,relu_num,[6,4,4],'relu',relu_int_he,[1,1,1],'same',mask_int)
    gen_conv2_1_2, gen_mconv2_1_2 = self.pconv(X2,X4,tanh_num,[6,4,4],'tanh',tanh_int_gl,[1,1,1],'same',mask_int)

    gen_cat2_1 = layers.concatenate([gen_conv2_1_1,gen_conv2_1_2],axis=4)
    gen_mcat2_1 = layers.concatenate([gen_mconv2_1_1,gen_conv2_1_2],axis=4)
  
    # input X  
    gen_conv1_2_1 = self.conv(gen_cat1_1,relu_num*2,[6,4,4],'relu',relu_int_he,[1,1,1],'same')
    gen_max1_1_1 = layers.MaxPool3D((2,2,1),strides=[2,2,1],padding='same')(gen_conv1_2_1)

    gen_conv1_2_2 = self.conv(gen_cat1_1,tanh_num*2,[6,4,4],'tanh',tanh_int_gl,[1,1,1],'same')
    gen_max1_1_2 = layers.MaxPool3D((2,2,1),strides=[2,2,1],padding='same')(gen_conv1_2_2)

    gen_cat1_2 = layers.concatenate([gen_max1_1_1,gen_max1_1_2],axis=4)
  
    # input M
    gen_conv2_2_1, gen_mconv2_2_1 = self.pconv(gen_cat2_1,gen_mcat2_1,relu_num*2,[6,4,4],'relu',relu_int_he,[1,1,1],'same',mask_int)
    gen_max2_1_1 = layers.MaxPool3D((2,2,1),strides=[2,2,1],padding='same')(gen_conv2_2_1)
    gen_mmax2_1_1 = layers.MaxPool3D((2,2,1),strides=[2,2,1],padding='same')(gen_mconv2_2_1)
  
    gen_conv2_2_2, gen_mconv2_2_2 = self.pconv(gen_cat2_1,gen_mcat2_1,tanh_num*2,[6,4,4],'tanh',tanh_int_gl,[1,1,1],'same',mask_int)
    gen_max2_1_2 = layers.MaxPool3D((2,2,1),strides=[2,2,1],padding='same')(gen_conv2_2_2)
    gen_mmax2_1_2 = layers.MaxPool3D((2,2,1),strides=[2,2,1],padding='same')(gen_mconv2_2_2)
  
    gen_cat2_2 = layers.concatenate([gen_max2_1_1,gen_max2_1_2],axis=4)
    gen_mcat2_2 = layers.concatenate([gen_mmax2_1_1,gen_mmax2_1_2],axis=4)
  
    # input X 
    gen_conv1_3_1 = self.conv(gen_cat1_2,relu_num*4,[3,3,3],'relu',relu_int_he,[1,1,1],'same')
    gen_max1_2_1 = layers.MaxPool3D((2,2,1),strides=[2,2,1],padding='same')(gen_conv1_3_1)

    gen_conv1_3_2 = self.conv(gen_cat1_2,tanh_num*4,[3,3,3],'tanh',tanh_int_gl,[1,1,1],'same')
    gen_max1_2_2 = layers.MaxPool3D((2,2,1),strides=[2,2,1],padding='same')(gen_conv1_3_2)

    gen_cat1_3 = layers.concatenate([gen_max1_2_1,gen_max1_2_2],axis=4)
  
    # input M 
    gen_conv2_3_1,gen_mconv2_3_1 = self.pconv(gen_cat2_2,gen_mcat2_2,relu_num*4,[3,3,3],'relu',relu_int_he,[1,1,1],'same',mask_int)
    gen_max2_2_1 = layers.MaxPool3D((2,2,1),strides=[2,2,1],padding='same')(gen_conv2_3_1)
    gen_mmax2_2_1 = layers.MaxPool3D((2,2,1),strides=[2,2,1],padding='same')(gen_mconv2_3_1)
  
    gen_conv2_3_2 ,gen_mconv2_3_2 = self.pconv(gen_cat2_2,gen_mcat2_2,tanh_num*4,[3,3,3],'tanh',tanh_int_gl,[1,1,1],'same',mask_int)
    gen_max2_2_2 = layers.MaxPool3D((2,2,1),strides=[2,2,1],padding='same')(gen_conv2_3_2)
    gen_mmax2_2_2 = layers.MaxPool3D((2,2,1),strides=[2,2,1],padding='same')(gen_mconv2_3_2)
  
    gen_cat2_3 = layers.concatenate([gen_max2_2_1,gen_max2_2_2],axis=4)
    gen_mcat2_3 = layers.concatenate([gen_mmax2_2_1,gen_mmax2_2_2],axis=4)
  
    # residual input X
    gen_resnet_block1 = self.re_block(X1,(relu_num+tanh_num)*4,[4,4,4],'relu',relu_int_he,[4,4,1],'same')
  
    # residual input M
    gen_resnet_block2 = self.pconv_re_block(X2,X4,(relu_num+tanh_num)*4,[4,4,4],'relu',relu_int_he,[4,4,1],'same',mask_int)
  
    # input X
    gen_en_last1 = gen_cat1_3+gen_resnet_block1
  
    # input M 
    gen_en_last2 = gen_cat2_3+gen_resnet_block2
  
    gen_en_last = layers.concatenate([gen_en_last1,gen_en_last2],axis=4)
  
    gen_deconv1_1 = self.deconv(gen_en_last,relu_num*2,[3,3,3],'relu',relu_int_he,[2,2,1],'same')
    gen_deconv1_2 = self.deconv(gen_en_last,tanh_num*2,[3,3,3],'tanh',tanh_int_gl,[2,2,1],'same')
    gen_de_cat1 = layers.concatenate([gen_deconv1_1,gen_deconv1_2],axis=4)
  
    gen_deconv2_1 = self.deconv(gen_de_cat1,relu_num,[6,4,4],'relu',relu_int_he,[2,2,1],'same')
    gen_deconv2_2 = self.deconv(gen_de_cat1,tanh_num,[6,4,4],'tanh',tanh_int_gl,[2,2,1],'same')
    gen_de_cat2 = layers.concatenate([gen_deconv2_1,gen_deconv2_2],axis=4)
  
    gen_deconv3_1 = self.deconv(gen_de_cat2,1,[8,6,4],'relu',relu_int_he,[1,1,1],'same')
    gen_deconv3_2 = self.deconv(gen_de_cat2,1,[8,6,4],'tanh',tanh_int_gl,[1,1,1],'same')
    gen_de_cat3 = layers.concatenate([gen_deconv3_1,gen_deconv3_2],axis=4)
  
    gen_output = self.deconv(gen_de_cat3,1,[1,1,1],'linear',tanh_int_gl,[1,1,1],'same')
    gen_output = gen_output*X3
  
    return gen_output
  
  
  def discriminator(self,X) :

    relu_int_he = initializers.he_normal()
    tanh_int_gl = initializers.glorot_normal()
    relu_num = 16
    tanh_num = 8

    dis_conv1_1 = self.conv(X,relu_num,[8,4,4],'relu',relu_int_he,[2,2,2],'same')
    dis_conv1_2 = self.conv(X,tanh_num,[8,4,4],'tanh',tanh_int_gl,[2,2,2],'same')
    dis_cat1 = layers.concatenate([dis_conv1_1,dis_conv1_2],axis=4)
  
  
    dis_conv2_1 = self.conv(dis_cat1,relu_num*2,[6,6,3],'relu',relu_int_he,[2,2,2],'same')
    dis_conv2_2 = self.conv(dis_cat1,tanh_num*2,[6,6,3],'tanh',tanh_int_gl,[2,2,2],'same')
    dis_cat2 = layers.concatenate([dis_conv2_1,dis_conv2_2],axis=4)
  
    dis_conv3_1 = self.conv(dis_cat2,relu_num*4,[4,4,3],'relu',relu_int_he,[2,2,1],'same')
    dis_conv3_2 = self.conv(dis_cat2,tanh_num*4,[4,4,3],'tanh',tanh_int_gl,[2,2,1],'same')
    dis_cat3 = layers.concatenate([dis_conv3_1,dis_conv3_2],axis=4)
  
    dis_conv4_1 = self.conv(dis_cat3,relu_num*8,[3,3,3],'relu',relu_int_he,[2,2,1],'same')
    dis_conv4_2 = self.conv(dis_cat3,tanh_num*8,[3,3,3],'tanh',tanh_int_gl,[2,2,1],'same')
    dis_cat4 = layers.concatenate([dis_conv4_1,dis_conv4_2],axis=4)
  
    dis_conv5_1 = self.conv(dis_cat4,1,[3,3,3],'relu',relu_int_he,[2,2,2],'same')
    dis_conv5_2 = self.conv(dis_cat4,1,[3,3,3],'tanh',tanh_int_gl,[2,2,2],'same')
    dis_cat5 = layers.concatenate([dis_conv5_1,dis_conv5_2],axis=4)
    dis_output = dis_cat5
    return dis_output



