import torch
import torch.nn as nn
import torch.nn.functional as F
import math



class DilNetLRDisp(nn.Module):
    """Learning Deep CNN Denoiser Prior for Image Restoration. Zhang et al. ICCV 2017"""

    def __init__(self, input_nbr, label_nbr):
        """Init fields."""
        super(DilNetLRDisp, self).__init__()

        nb_filters=128

        # convolutions
        self.conv0 = nn.Conv2d(input_nbr, int(nb_filters/2), kernel_size=3, padding=1, dilation=1)
        self.conv1 = nn.Conv2d(int(nb_filters/2), nb_filters, kernel_size=3, padding=1, dilation=1, stride=2)
        self.conv2 = nn.Conv2d(nb_filters, nb_filters, kernel_size=3, padding=2, dilation=2)
        self.conv3 = nn.Conv2d(nb_filters, nb_filters, kernel_size=3, padding=4, dilation=4)
        self.conv4 = nn.Conv2d(nb_filters, nb_filters, kernel_size=3, padding=8, dilation=8)
        self.conv5 = nn.Conv2d(nb_filters, nb_filters, kernel_size=3, padding=12, dilation=12)
        self.conv6 = nn.Conv2d(nb_filters, nb_filters, kernel_size=3, padding=8, dilation=8)
        self.conv7 = nn.Conv2d(nb_filters, nb_filters, kernel_size=3, padding=4, dilation=4)
        self.conv8 = nn.Conv2d(nb_filters, nb_filters, kernel_size=3, padding=2, dilation=2)
        self.conv9 = nn.ConvTranspose2d(nb_filters, nb_filters, kernel_size=3, padding=1, output_padding=1, stride=2)
        self.conv10 = nn.Conv2d(int(nb_filters * 3/2), nb_filters, kernel_size=3, padding=1, dilation=1)
        self.conv11 = nn.Conv2d(nb_filters + input_nbr, nb_filters, kernel_size=3, padding=1, dilation=1)
        self.conv12 = nn.Conv2d(nb_filters, nb_filters, kernel_size=3, padding=1, dilation=1)
        self.conv121 = nn.Conv2d(nb_filters, int(nb_filters/2), kernel_size=3, padding=1, dilation=1)
        self.conv122 = nn.Conv2d(int(nb_filters/2), int(nb_filters/4), kernel_size=3, padding=1, dilation=1)
        self.conv13 = nn.Conv2d(int(nb_filters/4), label_nbr, kernel_size=1)
        

        # init the weights
        self.init_weights()


    def init_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()


    def forward(self, in_img):
        """Forward method."""

        # img = torch.cat((in_img[0],in_img[1],in_img[2]),1)

        img = torch.cat((in_img[:]),1)

        # Stage 1
        x0 = F.relu(self.conv0(img))
        x = F.relu(self.conv1(x0))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        x = torch.cat((x0,x),1)
        x = F.relu(self.conv10(x))
        x = torch.cat((img,x),1)
        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))
        x = F.relu(self.conv121(x))
        x = F.relu(self.conv122(x))
        x = self.conv13(x)
        
        # print(x.shape)

        # x = torch.nn.functional.interpolate(x, size=self.out_size, mode='bilinear', align_corners=False)
        
        # print(x.shape)
        
        return x

