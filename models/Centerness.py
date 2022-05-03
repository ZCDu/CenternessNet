import torch
import torch.nn as nn
from .DCNv2.dcn_v2 import DCN


class BasicBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0):
        super(BasicBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding)
        self.norm = nn.GroupNorm(num_groups=32, num_channels=256)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        norm = self.norm(x)
        relu = self.relu(norm)
        return relu

    # init
    #def _init_weight(self):
    #    for m in self.children():
    #        if isinstance(m, nn.Conv2d):
    #            torch.nn.init.kaiming_init(m.weight.data)
    #            if m.bias is not None:
    #                m.bias.data.zero_()
    #        elif isinstance(m, nn.GroupNorm):
    #            pass
    #        elif isinstance(m, nn.Linear):
    #            torch.nn.init.normal(m.weight.data, 0,0.01)
    #            m.bias.data.zero_()


class CenterNessNet(nn.Module):
    def __init__(self,
                 in_channels=256,
                 feat_channels=256,
                 stacked_convs=4,
                 dcn_on_last_conv=False):
        super(CenterNessNet, self).__init__()
        self.stacked_convs = stacked_convs
        self.dcn_on_last_conv = dcn_on_last_conv
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self._init_layers()
        self.init_weight()

    def _init_layers(self):
        self._init_centerness_convs()
        self._init_centerness_predict()

    def normal_init(self, module, mean=0, std=1, bias=0):
        nn.init.normal_(module.weight, mean, std)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)

    def init_weight(self):
        for m in self.centerness_convs.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
                elif isinstance(m, nn.GroupNorm):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    torch.nn.normal(m.weight.data, 0, 0.01)
                    m.bias.zero_()
        #self.normal_init(self.centerness_predict, std=0.01)
        for m in self.centerness_predict.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def _init_centerness_convs(self):
        self.centerness_convs = nn.Sequential()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            if self.dcn_on_last_conv and i == self.stacked_convs - 1:
                conv_cfg = DCN(chn,
                               self.feat_channels,
                               kernel_size=(3, 3),
                               stride=1,
                               padding=1,
                               deformable_groups=1)
            else:
                conv_cfg = BasicBlock(chn, self.feat_channels, 3, 1, 1)
            self.centerness_convs.add_module(
                ("centerness_" + str({0})).format(i), conv_cfg) # 参数

    def _init_centerness_predict(self):
        #self.centerness_predict = nn.Sequential(nn.Conv2d)
        # 在fcos中只对cneterness使用了卷积操作，是否合理
        self.centerness_predict = nn.Sequential()
        predict = nn.Conv2d(self.feat_channels, 1, 3, padding=1)
        self.centerness_predict.add_module(("centerness_predict"), predict)

    def forward(self, x):
        # error
        convs = self.centerness_convs(x)
        predict = self.centerness_predict(convs)
        #for param, meter in self.centerness_convs.named_parameters():
        #    print('centerness_convs convs:', param)

        return predict
