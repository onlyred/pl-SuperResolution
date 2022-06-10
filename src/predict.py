import torch
from PIL import Image
from models import get_model
from torchvision.transforms import functional as TF
from torchvision.utils import make_grid
import io

def output_sr(input_image, modelname, saved_model):
    device='cpu'
    # load model classes
    Model = get_model(modelname)

    model = Model.load_from_checkpoint( saved_model ) # save_hyperparameters()
    model.to(device)
    model.eval()
    model.freeze()
    # read input_image(byte type) 
    image_data = io.BytesIO(input_image)
    img = Image.open(image_data).convert('RGB')
    # Super resolution
    img_lr   = TF.to_tensor(img).unsqueeze(0)  # batch
    img_sr   = model(img_lr)
    grid = make_grid(img_sr, nrow=1)
    ndarr = grid.mul(255).add_(0.5).clamp(0, 255).permute(1,2,0).to('cpu', torch.uint8).numpy()
    return Image.fromarray(ndarr)
