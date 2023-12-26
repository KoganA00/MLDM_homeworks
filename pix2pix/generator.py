import torch.nn as nn
import torch


class DownBlock(nn.Module):
    def __init__(self,
                 input_ch,
                 output_ch,
                 kernel_size=4,
                 stride=2,
                 padding=1,
                 norm=True,
                 slope=0.2):
        super(DownBlock, self).__init__()
        self.input_ch, self.output_ch = input_ch, output_ch
        conv_list = [nn.Conv2d(input_ch,
                               output_ch,
                               kernel_size,
                               stride,
                               padding,
                               bias=False)]
        if norm:
            conv_list.append(nn.InstanceNorm2d(output_ch))
        #All ReLUs in the encoder are leaky, with slope 0.2
        conv_list.append(nn.LeakyReLU(slope))

        self.conv_net = nn.Sequential(*conv_list)
    def forward(self, x):
        x = self.conv_net(x)
        return x



class UpBlock(nn.Module):
    def __init__(self,
                 input_ch,
                 output_ch,
                 kernel_size=4,
                 stride=2,
                 padding=1,
                 dropout=0.0

                 ):
        super(UpBlock, self).__init__()
        self.input_ch, self.output_ch, self.dropout = input_ch, output_ch, dropout
        #This is Upsample
        conv_list = [nn.ConvTranspose2d(input_ch,
                               output_ch,
                               kernel_size,
                               stride,
                               padding,
                               bias=False)]

        conv_list.append(nn.InstanceNorm2d(output_ch))
        #ReLUs in the decoder are not leaky
        conv_list.append(nn.ReLU())
        if dropout > 0:
            conv_list.append(nn.Dropout(dropout))
        self.conv_net = nn.Sequential(*conv_list)
    def forward(self, x, skip_conn):
        x = self.conv_net(x)
        return torch.cat((x, skip_conn), 1)


class UNetModel(nn.Module):
    def __init__(self, input_ch=3, #We are working with RGB
                 output_ch=3,
                 kernel_size=4,
                 padding=1):

        super(UNetModel, self).__init__()

        self.input_block = DownBlock(input_ch, 64, norm=False)
        #encoder - C64-C128-C256-C512-C512-C512-C512-C512
        e_list = [DownBlock(64, 128),
                  DownBlock(128, 256),
                  DownBlock(256, 512),
                  DownBlock(512, 512),
                  DownBlock(512, 512),
                  DownBlock(512, 512),
                  DownBlock(512, 512, norm=False)
                                        ]
        self.encoders = nn.ModuleList(e_list)

        #decoder - CD512-CD512-CD512-C512-C256-C128-C64
        d_list = [UpBlock(512, 512, dropout=0.5),
                  UpBlock(1024, 512, dropout=0.5),
                  UpBlock(1024, 512, dropout=0.5),
                  UpBlock(1024, 512, dropout=0.5),
                  UpBlock(1024, 256),
                  UpBlock(512, 128),
                  UpBlock(256, 64)]
        self.decoders = nn.ModuleList(d_list)


        self.output_block = nn.Sequential(
            nn.Upsample(scale_factor=2),
                       nn.ZeroPad2d((1, 0, 1, 0)),
                       nn.Conv2d(128, output_ch, kernel_size, padding=padding),
                       nn.Tanh()
             )

    def forward(self, x):
        x = self.input_block(x)
        copied_input = [x]
        #Encode
        for ii, layer in enumerate(self.encoders):
            x = layer(x)
            copied_input.append(x)
        #Decode
        for ii, layer in enumerate(self.decoders):
             x = layer(x, copied_input[-ii - 2])

        x = self.output_block(x)
        return x
