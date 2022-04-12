import torch
import torch.nn as nn

class SRCNN(nn.Module):
    def __init__(self, in_channel=3, dims=64):
        super(SRCNN, self).__init__()
        
        self.layer = nn.Sequential( 
                       nn.Conv2d(in_channel, dims, 9, padding= 9 // 2),
                       nn.ReLU(),
                       nn.Conv2d(dims, dims//2, 5, padding=5 // 2),
                       nn.ReLU(),
                       nn.Conv2d(dims//2, in_channel, 5, padding=5 // 2) 
                   )
        
    def forward(self, x):
        return self.layer(x)

class FSRCNN(nn.Module):
    '''
    Fast-SRCNN
    1. Use input image without upsampling (bi-cubic interpolation > deconvolution layer)
    2. shrink size of networks (add shrinking & expanding layer)
    '''
    def __init__(self, scale_factor=2, in_channel=3, dims=56, shrk=12, m=4):
        super(FSRCNN, self).__init__()
 
        self.extract_layer = nn.Sequential(
                                nn.Conv2d(in_channel, dims, 5, padding=5 // 2),
                                nn.PReLU()
                             )
        self.shrink_layer  = nn.Sequential(
                                nn.Conv2d(dims, shrk, 1),
                                nn.PReLU()
                             )
        self.map_layers = nn.ModuleList()
        for _ in range(m):
            self.map_layers.append( 
                nn.Sequential(
                    nn.Conv2d(shrk, shrk, 3, padding= 3//2),
                    nn.PReLU()
                )
            )
        
        self.expand_layer = nn.Sequential(
                               nn.Conv2d(shrk, dims, 1),
                               nn.PReLU()
        )
  
        self.deconv_layer = nn.ConvTranspose2d(dims, in_channel, 9, stride=scale_factor, 
                                         padding= 9//2, output_padding=scale_factor-1)

    def forward(self, x):
        x = self.extract_layer(x)
        x = self.shrink_layer(x)
        for map_layer in self.map_layers:
            x = map_layer(x)
        x = self.expand_layer(x)
        x = self.deconv_layer(x)
        return x
