import torchvision.models as models
import torch.nn as nn


class PEPX(nn.Module):
    def __init__(self, ni, no):
        super(Discriminator, self).__init__()
        n1 = ni // 2
        n2 = ni * 2

        self.model = nn.Sequential(
            nn.Conv2d(ni, n1, 1, 1, bias=False),
            nn.BatchNorm2d(nc),
            nn.ReLU(inplace=True)

            nn.Conv2d(n1, n2, 1, 1, bias=False),
            nn.BatchNorm2d(nc),
            nn.ReLU(inplace=True)

            nn.Conv2d(n2, n2, 3, 1, 2, bias=False, groups=nc),
            nn.BatchNorm2d(n2),
            nn.ReLU(inplace=True)

            nn.Conv2d(n2, n1, 1, 1, bias=False),
            nn.BatchNorm2d(ni),
            nn.ReLU(inplace=True)

            nn.Conv2d(n1, no, 1, 1, bias=False),
            nn.BatchNorm2d(no),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.model(input)


class CovidDNN(nn.Module):
    """model architecture for CovidDNN"""

    def __init__(self):
        super(CovidDNN, self).__init__()

        self.conv = nn.Conv2d(3, 64, kernel_size=7,
                              stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv1 = nn.Conv2d(64, 256, kernel_size=1,
                               stride=2, padding=0, bias=False)
        self.conv2 = nn.Conv2d(256 * 4, 512, kernel_size=1,
                               stride=2, padding=0, bias=False)
        self.conv3 = nn.Conv2d(512 * 5, 1024, kernel_size=1,
                               stride=2, padding=0, bias=False)
        self.conv4 = nn.Conv2d(1024 * 7, 2048, kernel_size=1,
                               stride=2, padding=0, bias=False)

        self.pepx1_1 = PEPX(64, 256)
        self.pepx1_2 = PEPX(256 * 2, 256)
        self.pepx1_3 = PEPX(256 * 3, 256)

        self.pepx2_1 = PEPX(256 * 4, 512)
        self.pepx2_2 = PEPX(512 * 2, 512)
        self.pepx2_3 = PEPX(512 * 3, 512)
        self.pepx2_4 = PEPX(512 * 4, 512)

        self.pepx3_1 = PEPX(512 * 5, 1024)
        self.pepx3_2 = PEPX(1024 * 2, 1024)
        self.pepx3_3 = PEPX(1024 * 3, 1024)
        self.pepx3_4 = PEPX(1024 * 4, 1024)
        self.pepx3_5 = PEPX(1024 * 5, 1024)
        self.pepx3_6 = PEPX(1024 * 6, 1024)

        self.pepx4_1 = PEPX(1024 * 7, 2048)
        self.pepx4_2 = PEPX(2048 * 2, 2048)
        self.pepx4_3 = PEPX(2048 * 3, 2048)

        self.conv5 = nn.Conv2d(2048 * 4, 2048, kernel_size=1,
                               stride=1, padding=0, bias=False)
        self.fc = nn.sequential(nn.Linear(2048, 1024),
                                nn.Linear(1024, 256), nn.Linear(256, 4))

    def forward(self, input):
        y1 = self.conv(input)
        y_conv_1 = self.conv1(y1)

        y1_1 = self.pepx1_1(y)
        y1_1f = torch.cat((y1_1, y_conv_1), dim=1)

        y1_2 = self.pepx1_2(y1_1f)
        y1_2f = torch.cat((y1_2, y1_1, y_conv_1), dim=1)

        y1_3 = self.pepx1_3(y1_2f)
        y1_3f = torch.cat((y1_3, y1_2, y1_1, y_conv_1), dim=1)

        y_conv_1f = torch.cat((y_conv_1, y1_1, y1_2, y1_3), dim=1)
        y_conv_2 = self.conv2(y)

        y2_1 = self.pepx2_1(y1_3f)
        y2_1f = torch.cat((y2_1, y_conv_2), dim=1)

        y2_2 = self.pepx2_2(y2_1f)
        y2_2f = torch.cat((y2_2, y2_1, y_conv_2), dim=1)

        y2_3 = self.pepx2_3(y2_2f)
        y2_3f = torch.cat((y2_3, y2_2, y2_1, y_conv_2), dim=1)

        y2_4 = self.pepx2_4(y2_3f)
        y2_4f = torch.cat((y2_4, y2_3, y2_2, y2_1, y_conv_2), dim=1)

        y_conv_2f = torch.cat((y_conv_2, y2_1, y2_2, y2_3, y2_4), dim=1)
        y_conv_3 = self.conv3(y_conv_2f)

        y3_1 = self.pepx3_1(y2_4f)
        y3_1f = torch.cat((y3_1, y_conv_3), dim=1)

        y3_2 = self.pepx3_2(y3_1f)
        y3_2f = torch.cat((y3_2, y3_1, y_conv_3), dim=1)

        y3_3 = self.pepx3_3(y3_2f)
        y3_3f = torch.cat((y3_3, y3_2, y3_1, y_conv_3), dim=1)

        y3_4 = self.pepx3_4(y2_3f)
        y3_4f = torch.cat((y3_4, y3_3, y3_2, y3_1, y_conv_3), dim=1)

        y3_5 = self.pepx3_2(y3_4f)
        y3_5f = torch.cat((y3_5, y3_4, y3_3, y3_2, y3_1, y_conv_3), dim=1)

        y3_6 = self.pepx3_6(y3_5f)
        y3_6f = torch.cat(
            (y3_6, y3_5, y3_4, y3_3, y3_2, y3_1, y_conv_3), dim=1)

        y_conv_3f = torch.cat(
            (y_conv_3, y3_1, y3_2, y3_3, y3_4, y3_5, y3_6), dim=1)
        y_conv_4 = self.conv4(y_conv_3f)

        y4_1 = self.pepx4_1(y3_6f)
        y4_1f = torch.cat((y4_1, y_conv_4), dim=1)

        y4_2 = self.pepx4_2(y4_1f)
        y4_2f = torch.cat((y4_2, y4_1, y_conv_4), dim=1)

        y4_3 = self.pepx4_3(y4_2f)
        y4_3f = torch.cat((y4_3, y4_2, y4_1, y_conv_4), dim=1)

        y_conv_4f = torch.cat((y_conv_4, y4_1, y4_2, y4_3), dim=1)
        y_conv_5 = self.conv5(y_conv_4f)

        y = self.fc(y_conv_5)
        return y
