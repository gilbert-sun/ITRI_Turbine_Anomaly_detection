import torch
import torch.nn as nn

class AutoEncoder_MLP(nn.Module):
    def __init__(self, dim):
        super(AutoEncoder_MLP, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x.view(x.shape[0], -1))
        output = self.decoder(x)
        return output.view(x.shape[0], 1, 28, 28)

# ====================================================== #

class cnn_layer(nn.Module):
    def __init__(self, nin, nout):
        super(cnn_layer, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nin, nout, 3, 1, 1),
            nn.BatchNorm2d(nout),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.main(x)
class cnn_layer_up(nn.Module):
    def __init__(self, nin, nout):
        super(cnn_layer_up, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nin, nout, 3, 1, 1),
            nn.BatchNorm2d(nout),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.main(x)

# ====================================================== #

class cnn_encoder_256(nn.Module):
    def __init__(self, in_channel, dim):
        super(cnn_encoder_256, self).__init__()
        self.block1 = cnn_layer(in_channel, 64)
        self.block2 = cnn_layer(64, 128)
        self.block3 = cnn_layer(128, 128)
        self.block4 = cnn_layer(128, 256)
        self.block5 = cnn_layer(256, 256)
        self.block6 = cnn_layer(256, 512)
        self.block7 = nn.Sequential(
            nn.Conv2d(512, dim, kernel_size=4),
            nn.BatchNorm2d(dim),
            nn.Tanh(),
        )

        self.mp = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(self.mp(x))
        x = self.block3(self.mp(x))
        x = self.block4(self.mp(x))
        x = self.block5(self.mp(x))
        x = self.block6(self.mp(x))
        x = self.block7(self.mp(x))
        return x
    
class cnn_decoder_256(nn.Module):
    def __init__(self, out_channel, dim):
        super(cnn_decoder_256, self).__init__()
        self.block1 = nn.Sequential(
            nn.ConvTranspose2d(dim, 512, kernel_size=4),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
        )
        self.block2 = cnn_layer(512, 256)
        self.block3 = cnn_layer(256, 256)
        self.block4 = cnn_layer(256, 128)
        self.block5 = cnn_layer(128, 128)
        self.block6 = cnn_layer(128, 64)
        self.block7 = nn.Sequential(
            nn.ConvTranspose2d(64, out_channel, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )

        self.up = nn.UpsamplingNearest2d(scale_factor=2)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(self.up(x))
        x = self.block3(self.up(x))
        x = self.block4(self.up(x))
        x = self.block5(self.up(x))
        x = self.block6(self.up(x))
        x = self.block7(self.up(x))
        return x

# ====================================================== #

class cnn_layer3d(nn.Module):
    def __init__(self, nin, nout):
        super(cnn_layer3d, self).__init__()
        self.main = nn.Sequential(
            nn.Conv3d(nin, nout, 4, (1, 2, 2), 1),
            nn.BatchNorm3d(nout),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.main(x)
class cnn_layer3d_up(nn.Module):
    def __init__(self, nin, nout):
        super(cnn_layer3d_up, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose3d(nin, nout, 4, (1, 2, 2), 1),
            nn.BatchNorm3d(nout),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.main(x)

# ====================================================== #

class cnn_encoder3d_256(nn.Module):
    def __init__(self, in_channel, dim):
        super(cnn_encoder3d_256, self).__init__()
        self.block1 = cnn_layer3d(in_channel, 64)
        self.block2 = cnn_layer3d(64, 128)
        self.block3 = cnn_layer3d(128, 128)
        self.block4 = cnn_layer3d(128, 256)
        self.block5 = cnn_layer3d(256, 256)
        self.block6 = cnn_layer3d(256, 512)
        self.block7 = nn.Sequential(
            nn.Conv3d(512, dim, kernel_size=4),
            nn.BatchNorm3d(dim),
            nn.Tanh(),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        return x
    
class cnn_decoder3d_256(nn.Module):
    def __init__(self, out_channel, dim):
        super(cnn_decoder3d_256, self).__init__()
        self.block1 = nn.Sequential(
            nn.ConvTranspose3d(dim, 512, kernel_size=4),
            nn.BatchNorm3d(512),
            nn.LeakyReLU(),
        )
        self.block2 = cnn_layer3d_up(512, 256)
        self.block3 = cnn_layer3d_up(256, 256)
        self.block4 = cnn_layer3d_up(256, 128)
        self.block5 = cnn_layer3d_up(128, 128)
        self.block6 = cnn_layer3d_up(128, 64)
        self.block7 = nn.Sequential(
            nn.ConvTranspose3d(64, out_channel, 4, (1, 2, 2), 1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        return x.permute(0, 2, 1, 3, 4)


class AutoEncoder_CNN(nn.Module):
    def __init__(self, channel=3, dim=32, spatial=True):
        super(AutoEncoder_CNN, self).__init__()

        if spatial:
            self.encoder = cnn_encoder3d_256(in_channel=channel, dim=dim)
            self.decoder = cnn_decoder3d_256(out_channel=channel, dim=dim)
        else:
            self.encoder = cnn_encoder_256(in_channel=channel, dim=dim)
            self.decoder = cnn_decoder_256(out_channel=channel, dim=dim)

    def forward(self, x):
        x = self.encoder(x)
        # print('[encoder] shape', x.shape)

        x = self.decoder(x)
        # print('[decocer] shape', x.shape)
        # input()
        return x