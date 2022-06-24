from torchvision.models.resnet import resnet50
import torch.nn.functional as F
import torch.nn as nn
import torch
import torchvision.models as models
from torchvision.models.resnet import Bottleneck


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.network = ResNet(pretrained = True)
        self.connect_patches_feature = nn.Linear(1152, 128)

    def return_reduced_image_features_1(self,original):
        resnet_output = self.network(original)
        # original_features = self.projection_original_features(resnet_output)
        return resnet_output


    def return_reduced_image_patches_features(self, original, patches):
        original_features = self.return_reduced_image_features_1(original)
        patches_features = []
        for i, patch in enumerate(patches):
            patch_features = self.return_reduced_image_features_1(patch)
            patches_features.append(patch_features)
        patches_features = torch.cat(patches_features, axis=1)
        patches_features = self.connect_patches_feature(patches_features)
        return original_features, patches_features

    def forward(self, images=None, patches=None, mode=0):
        '''
        mode 0: get 128 feature for image,
        mode 1: get 128 feature for image and patches
        '''
        if mode == 0:
            return self.return_reduced_image_features_1(images)
        if mode == 1:
            return self.return_reduced_image_patches_features(images, patches)



class ResNet(nn.Module):
    def __init__(self,pretrained=True):
        self.inplanes = 64
        super(ResNet, self).__init__()
        resnet = models.resnet50(pretrained=True)
        if pretrained == True:
            print("loading is good")
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.mu = nn.Linear(2048, 128)

    def forward(self, x, **kwargs):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        z = self.layer4(x)
        x = self.avgpool(z)

        z = x.view(-1,2048)
        latent = self.mu(z)

        return latent


class Decoder(nn.Module):
    def __init__(self, block=Bottleneck):
        super(Decoder, self).__init__()

        self.active = nn.Tanh()
        self.layer_1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128,out_channels=128,kernel_size=(10,10),bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2,True)
        )
        self.layer_2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, stride=2, kernel_size=(8, 8), bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True)
        )
        self.layer_3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, stride=2, kernel_size=(6, 6), bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, True)
        )
        self.layer_4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=3, stride=4, kernel_size=(4, 4), bias=False),
            nn.BatchNorm2d(3),
            nn.LeakyReLU(0.2, True)
        )

    def forward(self, z):
        z = z.view(-1,128,1,1)
        x = self.layer_1(z)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.active(x)
        return x


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = Decoder().to(device)
    a = torch.rand(1,128).to(device)
    b= net(a)
    #print(b.shape)
    print(b.shape)