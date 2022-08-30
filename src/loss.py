from tensorflow.keras import Model, datasets, layers, models, backend



class Loss :
  def __init__(self) :
    super().__init__()

  def dis_loss(self,label,disoutput) :
  
    loss = backend.binary_crossentropy(label,disoutput+1e-08,from_logits=True)
    return loss*0.5
  
  def gen_loss(self,label,output,generator_weight=0.1) :
    loss = backend.binary_crossentropy(label,output+1e-08,from_logits=True)
    return loss*generator_weight
  
  def l1_loss(self,y_true,y_pred,l1_weight=1) :
    loss = backend.mean(backend.square(y_true-y_pred))
    return loss*l1_weight

