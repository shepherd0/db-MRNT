import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv64(nn.Module):
    """4-layers-conv"""

    def __init__(self):
        super(Conv64, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU())

    def forward(self, x):
        out1 = self.layer1(x)  # 42
        out1 = self.layer2(out1)  # 42
        out = self.layer3(out1)  # 21
        out2 = self.layer4(out)  # 21
        return out1, out2


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, kernel=3, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        if kernel == 1:
            self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        elif kernel == 3:
            self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        if kernel == 1:
            self.conv3 = nn.Conv2d(planes, planes, kernel_size=1, bias=False)
        elif kernel == 3:
            self.conv3 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet12(nn.Module):

    def __init__(self, block, layers, kernel=3):
        super(ResNet12, self).__init__()

        self.inplanes = 64
        self.kernel = kernel
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # self.nFeat = 512 * block.expansion

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, self.kernel, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.kernel))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)  # (64,64,84,84)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)  # (64,64,42,42)
        x = self.layer2(x)  # (64,128,21,21)
        x = self.layer3(x)  # (64,256,21,21)
        x = self.layer4(x)  # (64,256,6,6)

        return x


class MultyGenerator(nn.Module):
    """Multy feature generator"""
    def __init__(self):
        super(MultyGenerator, self).__init__()
        # self.layer_scale2 = nn.MaxPool2d(2)
        self.layer_scale2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(2)
        )
        # self.layer_scale3 = nn.Sequential(
        #     nn.Conv2d(64, 64, 5, padding=0, dilation=2),
        #     nn.BatchNorm2d(64),
        #     nn.LeakyReLU(),
        # )
        self.layer_scale3 = nn.Sequential(
            nn.Conv2d(64, 64, 3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
        )
        self.layer_scale4 = nn.Sequential(
            nn.Conv2d(64, 64, (1, 7)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, (7, 1)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU()
        )
        self.layer_scale5 = nn.Sequential(
            nn.Conv2d(64, 64, 5),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
        )

    def forward(self, x):

        out1 = self.layer_scale2(x)  # out2=(80,64,10,10)
        out2 = self.layer_scale3(x)  # out3=(80,64,19,19)
        out3 = self.layer_scale4(x)  # out4=(80,64,15,15)
        out4 = self.layer_scale5(x)  # out2=(80,64,17,17)

        return out1, out2, out3, out4


def l2normalize(x):
    return F.normalize(x, p=2, dim=1)


class KronRelationNets(nn.Module):
    def __init__(self):
        super(KronRelationNets, self).__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(64, 64, stride=1, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer1 = nn.Sequential(
            nn.Conv2d(64 * 4, 64, kernel_size=3, padding=0),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=0),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc1 = nn.Linear(64, 8)
        self.fc2 = nn.Linear(8, 1)
        self.global_pooling = nn.AdaptiveAvgPool2d((10, 10))

    def kron_matching(self, *inputs):
        assert len(inputs) == 2
        assert inputs[0].dim() == 4 and inputs[1].dim() == 4
        assert inputs[0].size() == inputs[1].size()
        N, C, H, W = inputs[0].size()  # (375,64,10,10)
        # 0是查询，1是支持
        w = inputs[0].permute(0, 2, 3, 1).contiguous().view(-1, C, 1, 1)  # (37500,64,1,1)
        x = inputs[1].view(1, N * C, H, W)  # (1,375*64,10,10)
        x = F.conv2d(x, w, groups=N)
        x = x.view(N, H, W, H, W)  # (375,10,10,10,10)
        return x

    def forward(self, support, query):
        x = self.global_pooling(query)  # (375,64,10,10)
        y = self.global_pooling(support)  # (375,64,10,10)
        b, c, h, w = x.size()  # 375 64 10 10
        x = l2normalize(x.view(b, c, h * w)).view(b, c, h, w)  # 每个局部位置/64
        y = l2normalize(y.view(b, c, h * w)).view(b, c, h, w)  # (375,64,10,10)
        kron_feature = self.kron_matching(x, y).view(b, h * w, h * w).max(2)[0].view(b, 1, h, w).repeat(1, 64, 1, 1)
        kron_feature_ = self.kron_matching(y, x).view(b, h * w, h * w).max(2)[0].view(b, 1, h, w).repeat(1, 64, 1, 1)
        x_feature = self.layer0(x)  # (375,64,10,10)
        y_feature = self.layer0(y)  # (375,64,10,10)
        out = torch.cat((x_feature, kron_feature, kron_feature_, y_feature), 1)
        out = self.layer1(out)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out).view(-1, 5)
        return out


class FPNRelationNet(nn.Module):
    def __init__(self, num_f=2, padset=1):
        super(FPNRelationNet, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(64 * num_f, 64, kernel_size=3, padding=padset, stride=1),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=0, stride=1),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(64, 8)
        self.fc2 = nn.Linear(8, 1)

    def forward(self, x):  # (375,64,21,21)
        out = self.layer1(x)  # (375,64,21,21),(10,10),(5,5)
        out1 = self.layer2(out)  # (375,64,9,9),(4,4),(1,1)
        out = self.avg(out1)  # (375,64,1,1)
        out = out.view(out.size(0), -1)  # (375,64)
        out = F.relu(self.fc1(out))
        out = self.fc2(out).view(-1, 5)
        return out, out1


class MsCRN(nn.Module):
    def __init__(self, num_class, num_support, num_query, embedding_net="Conv64"):
        super(MsCRN, self).__init__()

        self.num_class = num_class
        self.num_support = num_support
        self.num_query = num_query
        self.embedding_net = embedding_net

        if self.embedding_net == "Conv64":
            self.feature_encoder = Conv64()
        elif self.embedding_net == "ResNet12":
            self.feature_encoder = ResNet12(BasicBlock, [1, 1, 1, 1], kernel=3)

        self.multy_feature_generator = MultyGenerator()

        # self.global_pooling = nn.AdaptiveAvgPool2d((10, 10))

        self.relationNet = KronRelationNets()
        self.relation1 = FPNRelationNet()
        self.relation2 = FPNRelationNet(3)
        self.relation3 = FPNRelationNet(3)

        # self.relation2 = FPNRelationNet(num_f=3, padset=0)
        # self.relation3 = FPNRelationNet(num_f=3)

        # self.multi_w = nn.Parameter(torch.ones(5))
        # self.wfpn = nn.Parameter(torch.ones(3))

        # self.global_pooling = nn.AdaptiveAvgPool2d((10, 10))
        self.unsample1 = nn.Upsample(21)
        self.unsample2 = nn.Upsample(10)
        # self.conv1 = nn.Conv2d(64, 64, 1)
        # self.conv2 = nn.Conv2d(64, 64, 1)
        # self.conv3 = nn.Conv2d(64, 64, 1)

    def forward(self, support_x, query_x, class_num, class_query, feature_dim):
        """
            3*3/2+1           5*5-2               1*7 7*1           5*5
            sf1=(5,64,10,10)  sf2=(5,64,13,13)  sf3=(5,64,15,15)  sf4=(5,64,17,17)
            qf1=(75,64,10,10) qf2=(75,64,13,13) qf3=(75,64,15,15) qf4=(75,64,17,17)
        """
        # support_x=(5,3,84,84)  query_x=(75,3,84,84)

        # feature_encoding
        # (75,64,42,42)
        support_feature0, support_features = self.feature_encoder(support_x)  # (42)(5,64,21,21)
        query_feature0, query_features = self.feature_encoder(query_x)  # (42)(75,64,21,21)

        # multy scale feature
        s_fea_s1, s_fea_s2, s_fea_s3, s_fea_s4 = self.multy_feature_generator(support_features)
        q_fea_s1, q_fea_s2, q_fea_s3, q_fea_s4 = self.multy_feature_generator(query_features)

        # FPN-low
        support_feature0 = support_feature0.unsqueeze(0).repeat(class_num * class_query, 1, 1, 1, 1)  # (75,5,64,42,42)
        query_feature0 = query_feature0.unsqueeze(1).repeat(1, class_num, 1, 1, 1)  # (75,5,64,42,42)
        support_feature0 = support_feature0.view(-1, feature_dim, 42, 42)
        query_feature0 = query_feature0.view(-1, feature_dim, 42, 42)

        # FPN-mid
        support_features = support_features.unsqueeze(0).repeat(class_num * class_query, 1, 1, 1, 1)  # (75,5,64,21,21)
        query_features = query_features.unsqueeze(1).repeat(1, class_num, 1, 1, 1)  # (75,5,64,21,21)
        support_features = support_features.view(-1, feature_dim, 21, 21)
        query_features = query_features.view(-1, feature_dim, 21, 21)

        # FPN-top
        s_fea_s1 = s_fea_s1.unsqueeze(0).repeat(class_num * class_query, 1, 1, 1, 1)  # (75,5,64,10,10)
        q_fea_s1 = q_fea_s1.unsqueeze(1).repeat(1, class_num, 1, 1, 1)  # (75,5,64,10,10)
        s_fea_s1 = s_fea_s1.view(-1, feature_dim, 10, 10)
        q_fea_s1 = q_fea_s1.view(-1, feature_dim, 10, 10)

        # FPN-relation
        sq_cat = torch.cat((support_feature0, query_feature0), 1)  # (375,128,42,42)
        fpn_score1, out1 = self.relation1(sq_cat)  # score, 9
        out1 = self.unsample1(out1)  # 21

        sq_cat = torch.cat((support_features, out1, query_features), 1)  # (375,128,21,21)
        fpn_score2, out2 = self.relation2(sq_cat)  # 21
        out2 = self.unsample2(out2)  # 10

        sq_cat = torch.cat((s_fea_s1, out2, q_fea_s1), 1)  # (375,128,10,10)
        fpn_score3, _ = self.relation3(sq_cat)  # 10

        # multy-scale
        s_fea_s2 = s_fea_s2.unsqueeze(0).repeat(class_num * class_query, 1, 1, 1, 1)  # (75,5,64,13,13)
        q_fea_s2 = q_fea_s2.unsqueeze(1).repeat(1, class_num, 1, 1, 1)  # (75,5,64,13,13)
        s_fea_s2 = s_fea_s2.view(-1, feature_dim, 19, 19)
        q_fea_s2 = q_fea_s2.view(-1, feature_dim, 19, 19)

        s_fea_s3 = s_fea_s3.unsqueeze(0).repeat(class_num * class_query, 1, 1, 1, 1)  # (75,5,64,15,15)
        q_fea_s3 = q_fea_s3.unsqueeze(1).repeat(1, class_num, 1, 1, 1)  # (75,5,64,15,15)
        s_fea_s3 = s_fea_s3.view(-1, feature_dim, 15, 15)
        q_fea_s3 = q_fea_s3.view(-1, feature_dim, 15, 15)

        s_fea_s4 = s_fea_s4.unsqueeze(0).repeat(class_num * class_query, 1, 1, 1, 1)  # (75,5,64,17,17)
        q_fea_s4 = q_fea_s4.unsqueeze(1).repeat(1, class_num, 1, 1, 1)  # (75,5,64,17,17)
        s_fea_s4 = s_fea_s4.view(-1, feature_dim, 17, 17)
        q_fea_s4 = q_fea_s4.view(-1, feature_dim, 17, 17)

        # KP_attention_relation_net
        scores_0 = self.relationNet(support_features, query_features)
        scores_1 = self.relationNet(s_fea_s1, q_fea_s1)
        scores_2 = self.relationNet(s_fea_s2, q_fea_s2)
        scores_3 = self.relationNet(s_fea_s3, q_fea_s3)
        scores_4 = self.relationNet(s_fea_s4, q_fea_s4)

        # FPN_up_sample
        # s_fea_s1_up = self.unsample1(s_fea_s1)  # (5,64,21,21)
        # q_fea_s1_up = self.unsample1(q_fea_s1)  # (75,64,21,21)
        #
        # fpn_mid_s = support_features + s_fea_s1_up  # (5,64,21,21)
        # fpn_mid_q = query_features + q_fea_s1_up  # (75,64,21,21)
        #
        # s_fea_mid_up = self.unsample2(fpn_mid_s)  # (5,64,42,42)
        # q_fea_mid_up = self.unsample2(fpn_mid_q)  # (75,64,42,42)
        #
        # fpn_low_s = s_fea_mid_up + support_feature0  # (5,64,42,42)
        # fpn_low_q = q_fea_mid_up + query_feature0  # (75,64,42,42)

        return fpn_score1, fpn_score2, fpn_score3, scores_0, scores_1, scores_2, scores_3, scores_4


"""
network_dil2:
description: FPN第一层为最浅的一层，关系传递
"""
