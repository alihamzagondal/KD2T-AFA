import torch
import torchvision.models
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
from tqdm import tqdm


print(torch.__version__)
print(torch.version.cuda)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


#################################################################################

def inf_Diver_Stat_Att(feat, lambda1=0.50, lambda2=0.25, lambda3=0.25):
    B, C, H, W = feat.shape

    # Global average pooling
    gap = F.adaptive_avg_pool2d(feat, 1).view(B, C)  # [B, C]
    gap_mean = gap.mean(dim=1)  # [B]

    # Channel-wise std
    std_channel = torch.std(gap, dim=1)  # [B]

    # Spatial entropy
    feat_flat = torch.abs(feat).view(B, -1)
    p = feat_flat / (feat_flat.sum(dim=1, keepdim=True) + 1e-8)
    entropy = -(p * torch.log(p + 1e-8)).sum(dim=1)  # [B]

    # Weighted combination
    score = lambda1 * gap_mean + lambda2 * std_channel + lambda3 * entropy
    return score  # [B]

class adapFeaDistSwitch(nn.Module):
    def __init__(self, in_channels1, in_channels2, out_channels):
        super(adapFeaDistSwitch, self).__init__()

        self.layer1=nn.Conv2d(in_channels=in_channels1, out_channels=out_channels, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.GELU()
        self.layer2 = nn.Conv2d(in_channels=in_channels2, out_channels=out_channels, kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.SiLU()

    def forward(self, x1,x2):
        originalF1 = x1
        originalF2 = x2
        out1 = self.act1(self.bn1(self.layer1(x1)))
        out2 = self.act2(self.bn2(self.layer2(x2)))
        score1 = inf_Diver_Stat_Att(out1).mean()  # [B]
        score2 = inf_Diver_Stat_Att(out2).mean()  # [B]

        if score1>=score2:
            out=originalF1
        else:
            out = originalF2

        return out




class FeatureTransformation(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FeatureTransformation, self).__init__()
        # 1x1 Convolution to match the feature dimensions
        self.transform = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1).to(device)

    def forward(self, x):
        return self.transform(x)



# Student Feature transformation for Teacher 1
transform_st_feat1_for_T1 = FeatureTransformation(48, 96)  # 56x56x24 -> 56x56x96 (Teacher 1)
transform_st_feat2_for_T1 = FeatureTransformation(96, 192)  # 28x28x32 -> 28x28x192 (Teacher 1)
transform_st_feat3_for_T1 = FeatureTransformation(192, 384)  # 14x14x96 -> 14x14x384 (Teacher 1)
transform_st_feat4_for_T1 = FeatureTransformation(768, 768)  # 7x7x1280 -> 7x7x768 (Teacher 1)

# Student Feature transformation for Teacher 2
transform_st_feat1_for_T2 = FeatureTransformation(48, 24)  # 28x28x32 -> 28x28x112 (Teacher 2)
transform_st_feat2_for_T2 = FeatureTransformation(96, 40)  # 14x14x96 -> 14x14x1280 (Teacher 2)
transform_st_feat3_for_T2 = FeatureTransformation(192, 112)  # 28x28x32 -> 28x28x112 (Teacher 2)
transform_st_feat4_for_T2 = FeatureTransformation(768, 1280)  # 14x14x96 -> 14x14x1280 (Teacher 2)




########################################################################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
from tqdm import tqdm


#
class DCBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DCBlock, self).__init__()

        # Define the structure for each parallel layer
        self.layer1 = self._create_layer(in_channels, out_channels, dilation_rate=1, padding=1)
        self.layer2 = self._create_layer(in_channels, out_channels, dilation_rate=3, padding=3)
        self.layer3 = self._create_layer(in_channels, out_channels, dilation_rate=5, padding=5)

    def _create_layer(self, in_channels, out_channels, dilation_rate, padding):
        # Use LayerNorm instead of BatchNorm2d
        layer_norm = LayerNorm(normalized_shape=out_channels)

        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=dilation_rate, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

    def forward(self, x):
        # Forward pass through each parallel layer
        out1 = self.layer1(x)
        out2 = self.layer2(x)
        out3 = self.layer3(x)

        # Concatenate the outputs along the channel dimension
        out = torch.cat([out1, out2, out3], dim=1)


        return out

class CustomBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CustomBlock, self).__init__()

        self.Layerconv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.BacthNorm=nn.BatchNorm2d(out_channels)
        self.gelu = nn.GELU()
        self.Layerconv1x1 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        # Forward pass through the block
        originalFeature = x
        x = self.Layerconv3x3(x)
        x=self.BacthNorm(x)
        x = self.gelu(x)
        x1 = self.Layerconv1x1(x)
        x2 = self.sigmoid(x1)

        outputs = x2 * x
        return outputs


class ResBlock_PA(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock_PA, self).__init__()

        self.conv_1x1_a = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=1,stride=stride, padding=0)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU6()

        self.conv_3x3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU6()

        self.conv_1x1_b = nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu3 = nn.ReLU6()

        self.PA_Conv1x1= nn.Conv2d(in_channels,out_channels,kernel_size=1)
        self.PA_Sigmoid= nn.Sigmoid()


    def forward(self, x):
        inputFeature = x

        x= self.conv_1x1_a(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv_3x3(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv_1x1_b(x)
        x= self.bn3(x)
        x = self.relu3(x)

        residualResults= x+inputFeature

        PA_Conv1x1_Out= self.PA_Conv1x1(residualResults)
        PA_Sigmoid_Out= self.PA_Sigmoid(PA_Conv1x1_Out)

        residualPA = PA_Sigmoid_Out*residualResults

        return residualPA

class ParalleConv_PA(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ParalleConv_PA, self).__init__()

        self.conv_1x1_PA = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,stride=stride, padding=0)
        self.sigmoid1=nn.Sigmoid()

        self.conv_3x3 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels//3, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels//3)
        self.gelu1 = nn.GELU()

        self.conv_5x5 = nn.Conv2d(in_channels=in_channels, out_channels = in_channels//3 , kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(in_channels//3)
        self.gelu2 = nn.GELU()

        self.Pooling= nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.sizeMatching= nn.Conv2d(in_channels, in_channels//3, kernel_size=1)




    def forward(self, x):
        inputFeature = x

        conv1x1_out=self.conv_1x1_PA(x)
        sigmoidOut= self.sigmoid1(conv1x1_out)

        PA= sigmoidOut*inputFeature

        x1= self.conv_3x3(x)
        x1= self.bn1(x1)
        x1= self.gelu1(x1)


        x2=self.conv_5x5(x)
        x2=self.bn2(x2)
        x2=self.gelu2(x2)


        x3=self.Pooling(x)
        x3=self.sizeMatching(x3)

        outputs = [x1,x2,x3]
        out1=torch.cat(outputs, 1)

        out= out1+ PA



        return out

class PixelAttentoin(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PixelAttentoin, self).__init__()

        self.Layerconv1x1 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        # Forward pass through the block
        inputFeature = x
        x1 = self.Layerconv1x1(x)
        x2 = self.sigmoid(x1)
        outputs= x2 * inputFeature
        return outputs


class Block(nn.Module):
 

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class ConvNeXt(nn.Module):
   

    def __init__(self, in_chans=3, num_classes=1000,
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0.,
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 ):
        super().__init__()

        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            if i == 0:
                stage.add_module('Dilation Block 1', DCBlock(in_channels=96, out_channels=32))

            elif i==1:
                stage.add_module('ConvBlockPixelAttention Block2',CustomBlock(in_channels=192, out_channels=192))

            elif i==2:
                stage.add_module('Parallel Convolution Bloc',ParalleConv_PA(in_channels=384,out_channels=384))

            elif i==3:
                stage.add_module('PixelAttention', PixelAttentoin(in_channels=768, out_channels=768))



            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1]))  # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


class LayerNorm(nn.Module):
  

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


model_urls = {
    
    "convnext_small_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",
    
}




@register_model
def convnext_small(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        url = model_urls['convnext_small_22k'] if in_22k else model_urls['convnext_small_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model



teacher_model_1 = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768])
teacher_model_1.head = nn.Linear(768, 38)
saved_model_path1 = ''  #### path of weight for DPA-Net on PlantVillage
teacher_model_1.load_state_dict(torch.load(saved_model_path1))
print(teacher_model_1)


teacher_model_2=torchvision.models.efficientnet_b1(pretrained=False)
saved_model_path2 = ''    #### path of weights for EfficientNet-b1 on PlantVilalge  
teacher_model_2.classifier[-1] = nn.Linear(teacher_model_2.classifier[-1].in_features, 38)
teacher_model_2.load_state_dict(torch.load(saved_model_path2))
print(teacher_model_2)






class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super(SEBlock, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // self.reduction, kernel_size=1, stride=1, padding=0)
        self.fc2 = nn.Conv2d(in_channels // self.reduction, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        z = self.avg_pool(x)
        z = F.relu(self.fc1(z))
        z = torch.sigmoid(self.fc2(z))

        return x * z  # Element-wise multiplication to recalibrate channels


class MultiScale_AttentionFusion(nn.Module):
    def __init__(self,in_channels, out_channels):
        super(MultiScale_AttentionFusion,self).__init__()
        self.dwConv1 = nn.Conv2d(in_channels=in_channels // 2, out_channels=out_channels // 2, kernel_size=3, stride=1,padding=1, groups=in_channels // 2)
        self.bn1 = nn.BatchNorm2d(in_channels // 2)
        self.act1 = nn.ReLU6()
        self.pw1 = nn.Conv2d(in_channels=in_channels // 2, out_channels=in_channels // 2, kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm2d(in_channels // 2)
        self.act2 = nn.ReLU6()
        self.dwConv2 = nn.Conv2d(in_channels=in_channels // 2, out_channels=out_channels // 2, kernel_size=5, stride=1,padding=2, groups=in_channels // 2)
        self.bn3 = nn.BatchNorm2d(in_channels // 2)
        self.act3 = nn.ReLU6()
        self.pw2 = nn.Conv2d(in_channels=in_channels // 2, out_channels=in_channels // 2, kernel_size=1, stride=1)
        self.bn4 = nn.BatchNorm2d(in_channels // 2)
        self.act4 = nn.ReLU6()

        self.SE_attention=SEBlock(in_channels=in_channels//2)




    def forward(self,x):
        origFeat = x
        channels = x.size(1)
        x1, x2 = torch.split(x, channels // 2, dim=1)
        x1 = self.dwConv1(x1)
        x1 = self.bn1(x1)
        x1 = self.act1(x1)
        x1 = self.pw1(x1)
        x1 = self.bn2(x1)
        x1 = self.act2(x1)

        x2 = self.dwConv2(x2)
        x2 = self.bn3(x2)
        x2 = self.act3(x2)
        x2 = self.pw2(x2)
        x2 = self.bn4(x2)
        x2= self.act4(x2)

        x = x1+x2

        att = self.SE_attention(x)
        x1_att = x1*att
        x2_att = x2*att
        out = torch.concat([x1_att, x2_att],dim=1)

        return out

teacher_model_1.to(device)
teacher_model_2.to(device)
for param in teacher_model_1.parameters():
    param.requires_grad = False
for param in teacher_model_2.parameters():
    param.requires_grad = False
teacher_model_1.eval()
teacher_model_2.eval()



######################### for student Model#################################

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super(SEBlock, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // self.reduction, kernel_size=1, stride=1, padding=0)
        self.fc2 = nn.Conv2d(in_channels // self.reduction, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        z = self.avg_pool(x)
        z = F.relu(self.fc1(z))
        z = torch.sigmoid(self.fc2(z))

        return x * z  # Element-wise multiplication to recalibrate channels


class MultiScale_AttentionFusion(nn.Module):
    def __init__(self,in_channels, out_channels):
        super(MultiScale_AttentionFusion,self).__init__()
        self.dwConv1 = nn.Conv2d(in_channels=in_channels // 2, out_channels=out_channels // 2, kernel_size=3, stride=1,padding=1, groups=in_channels // 2)
        self.bn1 = nn.BatchNorm2d(in_channels // 2)
        self.act1 = nn.ReLU6()
        self.pw1 = nn.Conv2d(in_channels=in_channels // 2, out_channels=in_channels // 2, kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm2d(in_channels // 2)
        self.act2 = nn.ReLU6()
        self.dwConv2 = nn.Conv2d(in_channels=in_channels // 2, out_channels=out_channels // 2, kernel_size=5, stride=1,padding=2, groups=in_channels // 2)
        self.bn3 = nn.BatchNorm2d(in_channels // 2)
        self.act3 = nn.ReLU6()
        self.pw2 = nn.Conv2d(in_channels=in_channels // 2, out_channels=in_channels // 2, kernel_size=1, stride=1)
        self.bn4 = nn.BatchNorm2d(in_channels // 2)
        self.act4 = nn.ReLU6()

        self.SE_attention=SEBlock(in_channels=in_channels//2)




    def forward(self,x):
        origFeat = x
        channels = x.size(1)
        x1, x2 = torch.split(x, channels // 2, dim=1)
        x1 = self.dwConv1(x1)
        x1 = self.bn1(x1)
        x1 = self.act1(x1)
        x1 = self.pw1(x1)
        x1 = self.bn2(x1)
        x1 = self.act2(x1)

        x2 = self.dwConv2(x2)
        x2 = self.bn3(x2)
        x2 = self.act3(x2)
        x2 = self.pw2(x2)
        x2 = self.bn4(x2)
        x2= self.act4(x2)

        x = x1+x2

        att = self.SE_attention(x)
        x1_att = x1*att
        x2_att = x2*att
        out = torch.concat([x1_att, x2_att],dim=1)

        return out



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
        for param in teacher_model_1.parameters():
            param.requires_grad = False
        for param in teacher_model_2.parameters():
            param.requires_grad = False

        self.downsample_layers = teacher_model_1.downsample_layers
        self.stages = teacher_model_1.stages
        self.efficientLyer = teacher_model_2.features

        self.firstLayer_ = firstConv(in_channels=3, out_channels=24)
        self.expension_stage1 = expensionLayer(in_channels=24, exp_factor=2)
        self.separable_stage1 = separableConvLayer(in_channels=48)

        self.ibr_stage1_a = invertedBottleNeck_residual(in_channels=48, exp_factor=2)
        self.ibr_stage1_b = invertedBottleNeck_residual(in_channels=48, exp_factor=2)

        self.expension_stage2 = expensionLayer(in_channels=48, exp_factor=2)
        self.separable_stage2 = separableConvLayer(in_channels=96)
        self.ibr_stage2_a = invertedBottleNeck_residual(in_channels=96, exp_factor=2)
        self.ibr_stage2_b = invertedBottleNeck_residual(in_channels=96, exp_factor=2)

        self.expension_stage3 = expensionLayer(in_channels=96, exp_factor=2)
        self.separable_stage3 = separableConvLayer(in_channels=192)
        self.ibr_stage3_a = invertedBottleNeck_residual(in_channels=192, exp_factor=2)
        self.ibr_stage3_b = invertedBottleNeck_residual(in_channels=192, exp_factor=2)
        self.ibr_stage3_c = invertedBottleNeck_residual(in_channels=192, exp_factor=2)
        self.ibr_stage3_d = invertedBottleNeck_residual(in_channels=192, exp_factor=2)
        self.ibr_stage3_e = invertedBottleNeck_residual(in_channels=192, exp_factor=2)
        self.ibr_stage3_f = invertedBottleNeck_residual(in_channels=192, exp_factor=2)

        self.expension_stage4 = expensionLayer(in_channels=192, exp_factor=2)
        self.separable_stage4 = separableConvLayer(in_channels=384)
        self.ibr_stage4_a = invertedBottleNeck_residual(in_channels=384, exp_factor=2)
        self.ibr_stage4_b = invertedBottleNeck_residual(in_channels=384, exp_factor=2)
        self.ibr_stage4_c = invertedBottleNeck_residual(in_channels=384, exp_factor=2)
        self.ibr_stage4_d = invertedBottleNeck_residual(in_channels=384, exp_factor=2)
        self.ibr_stage4_e = invertedBottleNeck_residual(in_channels=384, exp_factor=2)

        self.finalLayer_ = finalConv(in_channels=384, out_channels=768)
        self.gap = torch.nn.AdaptiveAvgPool2d(1)
        self.classifier_ = classifier(in_channels=768, out_channels=38)



    def forward(self, x, stage=None, use_teacher=False):
        teacher1_Features = []
        teacher2_Features = []
        student_Features = []
        if use_teacher:
            features1 = []
            teacher_x1 = x
            teacher_x2 = x
            for i in range(4):
                teacher_x1 = self.downsample_layers[i](teacher_x1)
                teacher_x1 = self.stages[i](teacher_x1)
                features1.append(teacher_x1)

            teacher1_Features.append(features1[0])
            teacher1_Features.append(features1[1])
            teacher1_Features.append(features1[2])
            teacher1_Features.append(features1[3])

            for idx, layer in enumerate(self.efficientLyer):
                teacher_x2 = layer(teacher_x2)
                if idx == 2:
                    te_feat1 = teacher_x2
                elif idx == 3:
                    te_feat2 = teacher_x2
                elif idx == 5:
                    te_feat3 = teacher_x2
                elif idx == 8:
                    te_feat4 = teacher_x2

            teacher2_Features.append(te_feat1)
            teacher2_Features.append(te_feat2)
            teacher2_Features.append(te_feat3)
            teacher2_Features.append(te_feat4)

        student_x = x
        x = self.firstLayer_(x)
        x = self.expension_stage1(x)
        x = self.separable_stage1(x)
        x = self.ibr_stage1_a(x)
        x = self.ibr_stage1_b(x)

        st_feat1 = x

        x = self.expension_stage2(x)
        x = self.separable_stage2(x)
        x = self.ibr_stage2_a(x)
        x = self.ibr_stage2_b(x)
        st_feat2 = x

        x = self.expension_stage3(x)
        x = self.separable_stage3(x)
        x = self.ibr_stage3_a(x)
        x = self.ibr_stage3_b(x)
        x = self.ibr_stage3_c(x)
        x = self.ibr_stage3_d(x)
        x = self.ibr_stage3_e(x)
        x = self.ibr_stage3_f(x)

        st_feat3 = x

        x = self.expension_stage4(x)
        x = self.separable_stage4(x)
        x = self.ibr_stage4_a(x)
        x = self.ibr_stage4_b(x)
        x = self.ibr_stage4_c(x)
        x = self.ibr_stage4_d(x)
        x = self.ibr_stage4_e(x)


        x = self.finalLayer_(x)
        st_feat4 = x


        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.classifier_(x)
        student_logits = x

        student_Features.append(st_feat1)
        student_Features.append(st_feat2)
        student_Features.append(st_feat3)
        student_Features.append(st_feat4)
        return student_logits, student_Features, teacher1_Features, teacher2_Features if use_teacher else None


st_model=st_Net()
st_model.to(device)
print(st_model)








