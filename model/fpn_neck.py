import torch.nn as nn
import torch.nn.functional as F
from model.asff import ASFF, AFM

#特征融合FPN
class FPN(nn.Module):
    def __init__(self, fpn_size, features=256):
        super(FPN, self).__init__()
        self.prj_5 = nn.Conv2d(fpn_size[3], features, kernel_size=1)
        self.prj_4 = nn.Conv2d(fpn_size[2], features, kernel_size=1)
        self.prj_3 = nn.Conv2d(fpn_size[1], features, kernel_size=1)
        self.prj_2 = nn.Conv2d(fpn_size[0], features, kernel_size=1)
        self.conv_5 = nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.conv_4 = nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.conv_3 = nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.conv_2 = nn.Conv2d(features, features, kernel_size=3, padding=1)

        self.apply(self.init_conv_kaiming)

    def upsamplelike(self, inputs):
        src, target = inputs
        return F.interpolate(src, size=(target.shape[2], target.shape[3]), mode='nearest')

    def init_conv_kaiming(self, module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_uniform_(module.weight, a=1)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        C2, C3, C4, C5 = x
        P5 = self.prj_5(C5)
        P4 = self.prj_4(C4)
        P3 = self.prj_3(C3)
        P2 = self.prj_2(C2)

        P4 = P4 + self.upsamplelike([P5, C4])
        P3 = P3 + self.upsamplelike([P4, C3])
        P2 = P2 + self.upsamplelike([P3, C2])

        P2 = self.conv_2(P2)
        P3 = self.conv_3(P3)
        # P4 = self.conv_4(P4)
        # P5 = self.conv_4(P5)

        return [P2, P3]

#FPN+自适应空间特征融合
class FPN_FRM(nn.Module):
    def __init__(self, fpn_size, features=128):
        super(FPN_FRM, self).__init__()

        self.prj_4 = nn.Conv2d(fpn_size[3], features, kernel_size=1)
        self.prj_3 = nn.Conv2d(fpn_size[2], features, kernel_size=1)
        self.prj_2 = nn.Conv2d(fpn_size[1], features, kernel_size=1)
        self.prj_1 = nn.Conv2d(fpn_size[0], features, kernel_size=1)

        self.conv_3 = nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.conv_2 = nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(features, features, kernel_size=3, padding=1)

        self.asff3 = ASFF(level=1, dim=[features,features,features], features=features)
        self.asff2 = ASFF(level=1, dim=[features,features,features], features=features)

    def upsamplelike(self, inputs):
        src, target = inputs
        return F.interpolate(src, size=(target.shape[2], target.shape[3]), mode='nearest')

    def forward(self, x):
        C1, C2, C3, C4 = x

        # Conv 1x1
        F4 = self.prj_4(C4)
        F3 = self.prj_3(C3)
        F2 = self.prj_2(C2)
        F1 = self.prj_1(C1)

        # FPN
        F3 = F3 + self.upsamplelike([F4, C3])
        F2 = F2 + self.upsamplelike([F3, C2])
        F1 = F1 + self.upsamplelike([F2, C1])

        F3 = self.conv_3(F3)
        F2 = self.conv_2(F2)
        F1 = self.conv_1(F1)

        P2 = self.asff2([F3, F2, F1])
        P3 = self.asff3([F4, F3, F2])

        return [P2, P3]

#自适应空间特征融合
class ASFF_FPN(nn.Module):
    def __init__(self, fpn_size, features=128):
        super(ASFF_FPN, self).__init__()

        self.asff3 = ASFF(level=1, dim=[fpn_size[3],fpn_size[2],fpn_size[1]], features=features)
        self.asff2 = ASFF(level=1, dim=[fpn_size[2],fpn_size[1],fpn_size[0]], features=features)

    def forward(self, x):
        C1, C2, C3, C4 = x

        P2 = self.asff2([C3, C2, C1])
        P3 = self.asff3([C4, C3, C2])

        return [P2, P3]

#特征细化
class AFM_FPN(nn.Module):
    def __init__(self, fpn_size, features=256):
        super(AFM_FPN, self).__init__()

        self.aff4 = AFM(level=0, dim=[fpn_size[2],fpn_size[1],fpn_size[0]], features=features)
        self.aff3 = AFM(level=1, dim=[fpn_size[2],fpn_size[1],fpn_size[0]], features=features)
        self.aff2 = AFM(level=2, dim=[fpn_size[2],fpn_size[1],fpn_size[0]], features=features)

    def forward(self, x):
        C2, C3, C4 ,C5= x

        P2 = self.aff2([C4, C3, C2])
        P3 = self.aff3([C4, C3, C2])
        P4 = self.aff4([C4, C3, C2])

        return [P2, P3, P4]


