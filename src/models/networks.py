import torch
import torch.nn as nn

class SRCNN(nn.Module):
    def __init__(self, in_channels=3, n_dims=64):
        super(SRCNN, self).__init__()
        
        self.layer = nn.Sequential( 
                       nn.Conv2d(in_channels, n_dims, kernel_size=9, padding= 9 // 2),
                       nn.ReLU(True),
                       nn.Conv2d(n_dims, n_dims//2, kernel_size=5, padding=5 // 2),
                       nn.ReLU(True),
                       nn.Conv2d(n_dims//2, in_channels, kernel_size=5, padding=5 // 2) 
                   )
        
    def forward(self, x):
        return self.layer(x)

class FSRCNN(nn.Module):
    '''
    Fast-SRCNN
    1. Use input image without upsampling (bi-cubic interpolation > deconvolution layer)
    2. shrink size of networks (add shrinking & expanding layer)
    '''
    def __init__(self, scale_factor=2, in_channels=3, n_dims=56, shrk=12, m=4):
        super(FSRCNN, self).__init__()
 
        self.extract_layer = nn.Sequential(
                                nn.Conv2d(in_channels, n_dims, kernel_size=5, padding=5 // 2),
                                nn.PReLU(True)
                             )
        self.shrink_layer  = nn.Sequential(
                                nn.Conv2d(n_dims, shrk, kernel_size=1),
                                nn.PReLU(True)
                             )
        self.map_layers = nn.ModuleList()
        for _ in range(m):
            self.map_layers.append( 
                nn.Sequential(
                    nn.Conv2d(shrk, shrk, kernel_size=3, padding= 3//2),
                    nn.PReLU(True)
                )
            )
        
        self.expand_layer = nn.Sequential(
                               nn.Conv2d(shrk, n_dims, kernel_size=1),
                               nn.PReLU(True)
        )
  
        self.deconv_layer = nn.ConvTranspose2d(n_dims, in_channels, kernel_size=9, 
                                stride=scale_factor, padding= 9//2, output_padding=scale_factor-1)

    def forward(self, x):
        x = self.extract_layer(x)
        x = self.shrink_layer(x)
        for map_layer in self.map_layers:
            x = map_layer(x)
        x = self.expand_layer(x)
        x = self.deconv_layer(x)
        return x

class ResBlock(nn.Module):
    def __init__(self, n_dims, res_scale=1.0):
        super(ResBlock, self).__init__()
        self.body = nn.Sequential(
                        nn.Conv2d(n_dims, n_dims, kernel_size=3, bias=True, padding=3//2),
                        nn.ReLU(True),
                        nn.Conv2d(n_dims, n_dims, kernel_size=3, bias=True, padding=3//2)
                    )
        self.res_scale = res_scale
 
    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res 
                           
class EDSR(nn.Module):
    def __init__(self, scale_factor=2, in_channels=3, n_dims=64, n_blocks=16, res_scale=1.0):
        super(EDSR, self).__init__()
        self.first_layer = nn.Conv2d(in_channels, n_dims, kernel_size=3, padding=3//2)
        self.mid_layers  = nn.ModuleList()
        for _ in range(n_blocks):
            self.mid_layers.append( ResBlock(n_dims, res_scale) )
        self.last_layer  = nn.Sequential(
                               nn.Conv2d(n_dims, n_dims * (scale_factor ** 2), kernel_size=3, stride=1, padding=1),
                               nn.PixelShuffle(scale_factor),
                               nn.ReLU(),
                               nn.Conv2d(n_dims, in_channels, kernel_size=3, stride=1, padding=1)
                           )

    def forward(self, x):
        x = self.first_layer(x)
        for mid_layer in self.mid_layers:
            res = mid_layer(x)
        res += x
        x = self.last_layer(res)
        return x
