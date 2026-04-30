
import torch
import torchvision.models
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
from tqdm import tqdm
from torchsummary import summary



print(torch.__version__)
print(torch.version.cuda)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


######################### for student Model#################################
class DRGA_With_ChannelAttention(nn.Module):
    def __init__(self, channels, kernel_size=3, reduction=16):
        super(DRGA_With_ChannelAttention, self).__init__()
        self.kernel_size = kernel_size
        self.avg_pool = nn.AvgPool2d(kernel_size, stride=1, padding=kernel_size // 2)

        # Channel Attention (SE-style)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, 1, bias=False)
        self.sigmoid_channel = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.size()

        # --- Spatial Attention based on local std ---
        mean = self.avg_pool(x)
        sq_diff = (x - mean) ** 2
        std = torch.sqrt(self.avg_pool(sq_diff) + 1e-6)  # shape: [B, C, H, W]

        # Normalize std to [0, 1]
        std_min = std.view(B, C, -1).min(dim=2, keepdim=True)[0].unsqueeze(-1)
        std_max = std.view(B, C, -1).max(dim=2, keepdim=True)[0].unsqueeze(-1)
        norm_std = (std - std_min) / (std_max - std_min + 1e-6)

        spatial_attention = torch.sigmoid(norm_std)  # shape: [B, C, H, W]

        # --- Channel Attention ---
        channel_avg = self.global_pool(x)              # shape: [B, C, 1, 1]
        ca = self.fc1(channel_avg)
        ca = self.relu(ca)
        ca = self.fc2(ca)
        channel_attention = self.sigmoid_channel(ca)   # shape: [B, C, 1, 1]

        # --- Combine both ---
        x = x * spatial_attention * channel_attention

        return x

class firstConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(firstConv, self).__init__()
        self.convLayer=nn.Conv2d(in_channels,out_channels,kernel_size=3, padding=1, stride=2, bias=False)
        self.bn=nn.BatchNorm2d(out_channels)
        self.act=nn.ReLU6()

    def forward(self, x):
        x = self.convLayer(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class finalConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(finalConv, self).__init__()
        self.convLayer = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU6()
    def forward(self, x):
        x = self.convLayer(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class expensionLayer(nn.Module):
    def __init__(self, in_channels,exp_factor):
        super(expensionLayer, self).__init__()
        self.convLayer = nn.Conv2d(in_channels=in_channels, out_channels=in_channels*exp_factor, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(in_channels*exp_factor)
        self.act = nn.ReLU6()
    def forward(self, x):
        x = self.convLayer(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class separableConvLayer(nn.Module):
    def __init__(self, in_channels):
        super(separableConvLayer, self).__init__()
        self.DW_ConvLayer = nn.Conv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=3,stride=2,padding=1,groups=in_channels,bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.act = nn.ReLU6()
        self.attentionLayer=DRGA_With_ChannelAttention(channels=in_channels)

        self.PW_ConvLayer=nn.Conv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=1,stride=1,bias=False)
        self.bn2=nn.BatchNorm2d(in_channels)

    def forward(self, x):
        x = self.DW_ConvLayer(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.attentionLayer(x)
        x = self.PW_ConvLayer(x)
        x = self.bn2(x)

        return x

class invertedBottleNeck_residual(nn.Module):
    def __init__(self, in_channels,exp_factor):
        super(invertedBottleNeck_residual, self).__init__()
        self.convLayer1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels*exp_factor, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels*exp_factor)
        self.act1 = nn.ReLU6()

        self.DW_ConvLayer=nn.Conv2d(in_channels=in_channels*exp_factor, out_channels=in_channels*exp_factor,kernel_size=3,stride=1,padding=1, groups=in_channels*exp_factor,bias=False)
        self.bn2=nn.BatchNorm2d(in_channels*exp_factor)
        self.act2=nn.ReLU6()
        self.attentionLayer=DRGA_With_ChannelAttention(channels=in_channels*exp_factor)



        self.PW_ConLayer=nn.Conv2d(in_channels=in_channels*exp_factor, out_channels=in_channels,kernel_size=1,stride=1, bias=False)
        self.bn3=nn.ReLU6()


    def forward(self, x):
        originalFeature = x
        x = self.convLayer1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.DW_ConvLayer(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.attentionLayer(x)
        x = self.PW_ConLayer(x)
        x = self.bn3(x)
        out = originalFeature+x
        return out

class classifier(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(classifier, self).__init__()
        self.dropoutLayer=nn.Dropout(p=0.2,inplace=False)
        self.FC=nn.Linear(in_features=in_channels, out_features=out_channels)
    def forward(self, x):
        x=self.dropoutLayer(x)
        x=self.FC(x)

        return x




class st_Net(nn.Module):
    def __init__(self):
        super(st_Net, self).__init__()
        self.firstLayer_=firstConv(in_channels=3,out_channels=24)
        self.expension_stage1 = expensionLayer(in_channels=24,exp_factor=2)
        self.separable_stage1 = separableConvLayer(in_channels=48)

        self.ibr_stage1_a = invertedBottleNeck_residual(in_channels=48,exp_factor=2)
        self.ibr_stage1_b = invertedBottleNeck_residual(in_channels=48, exp_factor=2)

        self.expension_stage2 = expensionLayer(in_channels=48,exp_factor=2)
        self.separable_stage2 = separableConvLayer(in_channels=96)
        self.ibr_stage2_a = invertedBottleNeck_residual(in_channels=96,exp_factor=2)
        self.ibr_stage2_b = invertedBottleNeck_residual(in_channels=96, exp_factor=2)

        self.expension_stage3 = expensionLayer(in_channels=96,exp_factor=2)
        self.separable_stage3 = separableConvLayer(in_channels=192)
        self.ibr_stage3_a = invertedBottleNeck_residual(in_channels=192,exp_factor=2)
        self.ibr_stage3_b = invertedBottleNeck_residual(in_channels=192, exp_factor=2)
        self.ibr_stage3_c = invertedBottleNeck_residual(in_channels=192, exp_factor=2)
        self.ibr_stage3_d = invertedBottleNeck_residual(in_channels=192, exp_factor=2)
        self.ibr_stage3_e = invertedBottleNeck_residual(in_channels=192, exp_factor=2)
        self.ibr_stage3_f = invertedBottleNeck_residual(in_channels=192, exp_factor=2)

        self.expension_stage4 = expensionLayer(in_channels=192, exp_factor=2)
        self.separable_stage4 = separableConvLayer(in_channels=384)
        self.ibr_stage4_a = invertedBottleNeck_residual(in_channels=384,exp_factor=2)
        self.ibr_stage4_b = invertedBottleNeck_residual(in_channels=384,exp_factor=2)
        self.ibr_stage4_c = invertedBottleNeck_residual(in_channels=384, exp_factor=2)
        self.ibr_stage4_d = invertedBottleNeck_residual(in_channels=384, exp_factor=2)
        self.ibr_stage4_e = invertedBottleNeck_residual(in_channels=384, exp_factor=2)


        self.finalLayer_=finalConv(in_channels=384,out_channels=768)
        self.gap = torch.nn.AdaptiveAvgPool2d(1)
        self.classifier_=classifier(in_channels=768, out_channels=38)

    def forward(self, x):


        x = self.firstLayer_(x)
        x = self.expension_stage1(x)
        x = self.separable_stage1(x)
        x = self.ibr_stage1_a(x)
        x = self.ibr_stage1_b(x)

        x = self.expension_stage2(x)
        x = self.separable_stage2(x)
        x = self.ibr_stage2_a(x)
        x = self.ibr_stage2_b(x)

        x = self.expension_stage3(x)
        x = self.separable_stage3(x)
        x = self.ibr_stage3_a(x)
        x = self.ibr_stage3_b(x)
        x = self.ibr_stage3_c(x)
        x = self.ibr_stage3_d(x)
        x = self.ibr_stage3_e(x)
        x = self.ibr_stage3_f(x)

        x = self.expension_stage4(x)
        x = self.separable_stage4(x)
        x = self.ibr_stage4_a(x)
        x = self.ibr_stage4_b(x)
        x = self.ibr_stage4_c(x)
        x = self.ibr_stage4_d(x)
        x = self.ibr_stage4_e(x)

        x = self.finalLayer_(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.classifier_(x)
        return x



st_model=st_Net()
print(st_model)
st_model.to(device)
summary(st_model,(3,224,224))