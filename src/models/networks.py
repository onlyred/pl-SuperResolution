import torch
import torch.nn as nn
import math

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
                                nn.PReLU(),
                             )
        self.shrink_layer  = nn.Sequential(
                                nn.Conv2d(n_dims, shrk, kernel_size=1),
                                nn.PReLU(),
                             )
        self.map_layers = nn.ModuleList()
        for _ in range(m):
            self.map_layers.append( 
                nn.Sequential(
                    nn.Conv2d(shrk, shrk, kernel_size=3, padding= 3//2),
                    nn.PReLU()
                )
            )
        
        self.expand_layer = nn.Sequential(
                               nn.Conv2d(shrk, n_dims, kernel_size=1),
                               nn.PReLU()
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

class ResBlock1(nn.Module):
    '''
    for EDSR
    '''
    def __init__(self, n_dims, res_scale=1.0):
        super(ResBlock1, self).__init__()
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
            self.mid_layers.append( ResBlock1(n_dims, res_scale) )
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

class ResBlock2(nn.Module):
    '''
    for SRGAN
    '''
    def __init__(self, channels):
        super(ResBlock2, self).__init__()
        self.res_block = nn.Sequential( 
                             nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                             nn.BatchNorm2d(channels),
                             nn.PReLU(),
                             nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                             nn.BatchNorm2d(channels),
                         )

    def forward(self, x):
        res = self.res_block(x)
        res = torch.add(res, x)
        return res

class UpsampleBlock(nn.Module):
    def __init__(self, channels, up_scale):
        super(UpsampleBlock, self).__init__()
        self.upsample = nn.Sequential(
                            nn.Conv2d(channels, channels * up_scale ** 2, kernel_size=3, padding=1),
                            nn.PixelShuffle(up_scale),
                            nn.PReLU(),
                        )

    def forward(self, x):
        out = self.upsample(x)
        return out
        

class Generator(nn.Module):
    def __init__(self, scale_factor):
        upsample_block_num = int(math.log(scale_factor,2)) 

        super(Generator,self).__init__()
        # first layer
        self.conv_block1 =  nn.Sequential(
                                nn.Conv2d(3, 64, kernel_size=9, padding=4),
                                nn.PReLU()
                            )
        # res layer
        res_block = [ ResBlock2(64) for _ in range(4) ]
        self.res_block = nn.Sequential(*res_block)
        # second layer    
        self.conv_block2 = nn.Sequential(
                               nn.Conv2d(64, 64, kernel_size=3, padding=1),
                               nn.BatchNorm2d(64),
                           )
        # upsampling 
        upsample_block = [ UpsampleBlock(64, 2)  for _ in range(upsample_block_num) ]
        self.upsample_block = nn.Sequential(*upsample_block)
        # final layer 
        
        self.conv_block3 = nn.Sequential( 
                               nn.Conv2d(64, 3, kernel_size=9, padding=4),
                               nn.Tanh()
                           )
         
    def forward(self, x):
        out1 = self.conv_block1(x)
        res  = self.res_block(out1)
        out2 = self.conv_block2(res)
        out = torch.add(out1, out2)
        out = self.upsample_block(out)
        out = self.conv_block3(out)
        return (out + 1) / 2.  # [-1,1] to [0,1]
        
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.hidden= 64
        self.block = nn.Sequential(
                         nn.Conv2d(3, self.hidden, kernel_size=3, padding=1),
                         nn.LeakyReLU(0.2, True),

                         *self._ConvBlock(self.hidden, self.hidden, 2),

                         *self._ConvBlock(self.hidden, self.hidden*2, 1),
                         *self._ConvBlock(self.hidden*2, self.hidden*2, 2),
                         
                         *self._ConvBlock(self.hidden*2, self.hidden*4, 1),
                         *self._ConvBlock(self.hidden*4, self.hidden*4, 2),

                         *self._ConvBlock(self.hidden*4, self.hidden*8, 1),
                         *self._ConvBlock(self.hidden*8, self.hidden*8, 2),
                     )   
        
        self.flatten = nn.Sequential(
                              nn.AdaptiveAvgPool2d(1),
                              nn.Conv2d(self.hidden * 8, 1024, kernel_size=1),
                              nn.LeakyReLU(0.2, True),
                              nn.Conv2d(1024, 1, kernel_size=1)
                          )

    def _ConvBlock(self, in_channels, out_channels, stride=1):
        out = [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
                nn.LeakyReLU(0.2, True),
                nn.BatchNorm2d(out_channels)
              ]
        return out

    def forward(self, x):
        batch_size = x.size(0)
        out = self.block(x)
        out = self.flatten(out).view(batch_size)
        out = torch.sigmoid(out)
        return out
