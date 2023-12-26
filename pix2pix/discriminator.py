import torch.nn as nn
import torch

class DiscriminatorBlock(nn.Module):
    def __init__(self,
                 input_ch,
                 output_ch,
                 kernel_size=4,
                 stride=2,
                 padding=1,
                 norm=True,
                 slope=0.2):
        super(DiscriminatorBlock, self).__init__()
        self.input_ch, self.output_ch = input_ch, output_ch
        conv_list = [nn.Conv2d(input_ch,
                               output_ch,
                               kernel_size,
                               stride,
                               padding)]
        if norm:
            conv_list.append(nn.InstanceNorm2d(output_ch))
        #ReLUs in the decoder are not leaky
        conv_list.append(nn.LeakyReLU(slope))
        self.conv_net = nn.Sequential(*conv_list)
    def forward(self, x):
        x = self.conv_net(x)
        return x


class Discriminator(nn.Module):
    def __init__(self,
                 input_ch=3,
                 output_ch=1,
                 kernel=4,
                 padding=1):
        super(Discriminator, self).__init__()
        #discriminator - C64-C128-C256-C512
        self.block_net = nn.Sequential(
            DiscriminatorBlock(input_ch * 2, 64, norm=False),
            DiscriminatorBlock(64, 128),
            DiscriminatorBlock(128, 256),
            DiscriminatorBlock(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, output_ch, kernel, padding=padding, bias=False)

        )


    def forward(self, x_img, y_img):
        x = torch.cat((x_img, y_img), 1)
        out = self.block_net(x)
        return out
