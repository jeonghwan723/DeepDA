from tensorflow.keras import Model, datasets, layers, models, backend, optimizers
from net import Net
from loss import Loss

class Module(Net,Loss) :
  def __init__(self) :
    super(Module,self).__init__()


  def model(self,xdim,ydim,zdim) :


    def dis_model() :
      X = layers.Input(shape=(xdim,ydim,zdim,2))
      dX1 = X[:,:,:,:,:1]
      dX0 = X[:,:,:,:,1:2]
      DX = dX1 - dX0
      dis_output = self.discriminator(DX)
      model = models.Model(inputs=X,outputs=dis_output)
      return model

    def gen_model() :
      X = layers.Input(shape=(xdim,ydim,zdim,4))
      gX1 = X[:,:,:,:,:1]
      gX2 = X[:,:,:,:,1:2]
      gX3 = X[:,:,:,:,2:3]
      gX4 = X[:,:,:,:,3:]
      gen_output = self.generator(gX1,gX2,gX3,gX4)
      model = models.Model(inputs=X,outputs=gen_output)
      return model

    ### discriminator
    dis_optimizer = optimizers.Adam()
    dis_model = dis_model()
    dis_model.compile(loss=self.dis_loss,optimizer=dis_optimizer)

    ### generator 
    gen_optimizer = optimizers.Adam()
    gen_input = layers.Input(shape=(xdim,ydim,zdim,4))

    gen_model = gen_model()

    gen_output = gen_model(gen_input)
    dis_shape = gen_output.shape
    dis_model.trainable = False

    dis_gen_input = layers.concatenate([gen_input[:,:,:,:,:1],gen_output],axis=4)
    dis_gen_output = dis_model(dis_gen_input)


    dis_shape = dis_gen_output.shape[1:]
    combined_model = models.Model(inputs=[gen_input],
                                  outputs = [dis_gen_output,gen_output])
    combined_model.compile(loss=[self.gen_loss,self.l1_loss],optimizer=gen_optimizer)

    return dis_shape, dis_model, gen_model, combined_model

