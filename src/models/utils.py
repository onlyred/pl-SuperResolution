from .models import SRCNN_Model, FSRCNN_Model, EDSR_Model, SRGAN_Model

def get_model(model_name):
    if model_name == "srcnn":
        Model = SRCNN_Model
    elif model_name == "fsrcnn":
        Model = FSRCNN_Model
    elif model_name == "edsr":
        Model = EDSR_Model
    elif model_name == "srgan":
        Model = SRGAN_Model
    else:
        raise NotImplementedError
    return Model 
