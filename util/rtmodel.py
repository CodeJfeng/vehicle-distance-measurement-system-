import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class DecomNet(nn.Module):
    def __init__(self, channel=64, kernel_size=3):
        super(DecomNet, self).__init__()
        # Shallow feature extraction 卷积网路提取图像特征
        self.net1_conv0 = nn.Conv2d(4, channel, kernel_size * 3,
                                    padding=4, padding_mode='replicate')
        # Activated layers!
        self.net1_convs = nn.Sequential(nn.Conv2d(channel, channel, kernel_size, padding=1, padding_mode='replicate'),
                                        nn.ReLU(),
                                        nn.Conv2d(channel, channel, kernel_size,
                                                  padding=1, padding_mode='replicate'),
                                        nn.ReLU(),
                                        nn.Conv2d(channel, channel, kernel_size,
                                                  padding=1, padding_mode='replicate'),
                                        nn.ReLU(),
                                        nn.Conv2d(channel, channel, kernel_size,
                                                  padding=1, padding_mode='replicate'),
                                        nn.ReLU(),
                                        nn.Conv2d(channel, channel, kernel_size,
                                                  padding=1, padding_mode='replicate'),
                                        nn.ReLU())
        # Final recon layer
        self.net1_recon = nn.Conv2d(channel, 4, kernel_size,
                                    padding=1, padding_mode='replicate')

    def forward(self, input_im):
        input_max = torch.max(input_im, dim=1, keepdim=True)[0]
        input_img = torch.cat((input_max, input_im), dim=1)
        feats0 = self.net1_conv0(input_img)  # 一层conv
        featss = self.net1_convs(feats0)  # 更高维度5层次的 conv+rule 用于调整模块
        outs = self.net1_recon(featss)  # 最后一个conv进行收尾
        R = torch.sigmoid(outs[:, 0:3, :, :])  # 进行sigmod收尾
        L = torch.sigmoid(outs[:, 3:4, :, :])
        return R, L


class RelightNet(nn.Module):
    def __init__(self, channel=64, kernel_size=3):
        super(RelightNet, self).__init__()

        self.relu = nn.ReLU()
        self.net2_conv0_1 = nn.Conv2d(4, channel, kernel_size,
                                      padding=1, padding_mode='replicate')

        self.net2_conv1_1 = nn.Conv2d(channel, channel, kernel_size, stride=2,
                                      padding=1, padding_mode='replicate')
        self.net2_conv1_2 = nn.Conv2d(channel, channel, kernel_size, stride=2,
                                      padding=1, padding_mode='replicate')
        self.net2_conv1_3 = nn.Conv2d(channel, channel, kernel_size, stride=2,
                                      padding=1, padding_mode='replicate')

        self.net2_deconv1_1 = nn.Conv2d(channel * 2, channel, kernel_size,
                                        padding=1, padding_mode='replicate')
        self.net2_deconv1_2 = nn.Conv2d(channel * 2, channel, kernel_size,
                                        padding=1, padding_mode='replicate')
        self.net2_deconv1_3 = nn.Conv2d(channel * 2, channel, kernel_size,
                                        padding=1, padding_mode='replicate')

        self.net2_fusion = nn.Conv2d(channel * 3, channel, kernel_size=1,
                                     padding=1, padding_mode='replicate')
        self.net2_output = nn.Conv2d(channel, 1, kernel_size=3, padding=0)

    def forward(self, input_L, input_R):
        input_img = torch.cat((input_R, input_L), dim=1)  # 调整模块
        out0 = self.net2_conv0_1(input_img)
        out1 = self.relu(self.net2_conv1_1(out0))
        out2 = self.relu(self.net2_conv1_2(out1))
        out3 = self.relu(self.net2_conv1_3(out2))  # 提取特征

        out3_up = F.interpolate(out3, size=(out2.size()[2], out2.size()[3]))  # 进行下采样
        deconv1 = self.relu(self.net2_deconv1_1(torch.cat((out3_up, out2), dim=1)))  # 第一层 重建模块
        deconv1_up = F.interpolate(deconv1, size=(out1.size()[2], out1.size()[3]))  # 进行采样
        deconv2 = self.relu(self.net2_deconv1_2(torch.cat((deconv1_up, out1), dim=1)))
        deconv2_up = F.interpolate(deconv2, size=(out0.size()[2], out0.size()[3]))
        deconv3 = self.relu(self.net2_deconv1_3(torch.cat((deconv2_up, out0), dim=1)))  # 重复上一操作

        deconv1_rs = F.interpolate(deconv1, size=(input_R.size()[2], input_R.size()[3]))  # 得出resize的特征模型 再次采样
        deconv2_rs = F.interpolate(deconv2, size=(input_R.size()[2], input_R.size()[3]))  #
        feats_all = torch.cat((deconv1_rs, deconv2_rs, deconv3), dim=1)  # 拼接特征
        feats_fus = self.net2_fusion(feats_all)  # 进行卷积Conv2d
        output = self.net2_output(feats_fus)  # 输出模型特征
        return output

class RetinexNet(nn.Module):
    def __init__(self):
        super(RetinexNet, self).__init__()
        self.DecomNet = DecomNet()
        self.RelightNet = RelightNet()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, input_low, input_high):
        # Forward DecompNet
        input_low = Variable(torch.FloatTensor(torch.from_numpy(input_low))).to(self.device)
        R_low, I_low = self.DecomNet(input_low)  # 得到低光照的两个图像分解

        # Forward RelightNet 低光照下传播网络模型
        I_delta = self.RelightNet(I_low, R_low)

        # Other variables
        I_low_3 = torch.cat((I_low, I_low, I_low), dim=1)
        I_delta_3 = torch.cat((I_delta, I_delta, I_delta), dim=1)

        self.output_R_low = R_low.detach().cpu()
        self.output_I_low = I_low_3.detach().cpu()
        self.output_I_delta = I_delta_3.detach().cpu()
        self.output_S = R_low.detach().cpu() * I_delta_3.detach().cpu()

    def gradient(self, input_tensor, direction):
        self.smooth_kernel_x = torch.FloatTensor([[0, 0], [-1, 1]]).view((1, 1, 2, 2)).to(self.device)
        self.smooth_kernel_y = torch.transpose(self.smooth_kernel_x, 2, 3)

        if direction == "x":
            kernel = self.smooth_kernel_x
        elif direction == "y":
            kernel = self.smooth_kernel_y
        grad_out = torch.abs(F.conv2d(input_tensor, kernel,
                                      stride=1, padding=1))
        return grad_out

    def ave_gradient(self, input_tensor, direction):
        return F.avg_pool2d(self.gradient(input_tensor, direction),
                            kernel_size=3, stride=1, padding=1)

    def smooth(self, input_I, input_R):
        input_R = 0.299 * input_R[:, 0, :, :] + 0.587 * input_R[:, 1, :, :] + 0.114 * input_R[:, 2, :, :]
        input_R = torch.unsqueeze(input_R, dim=1)
        return torch.mean(self.gradient(input_I, "x") * torch.exp(-10 * self.ave_gradient(input_R, "x")) +
                          self.gradient(input_I, "y") * torch.exp(-10 * self.ave_gradient(input_R, "y")))

    def predict_load(self):  # 先预测一张
        Decom_tar = './RetinexNet/RetinexNet_PyTorch-master/ckpts4/Decom/6600.tar'
        ckpt_dict = torch.load(Decom_tar, map_location=self.device)
        self.DecomNet.load_state_dict(ckpt_dict)
        Relight_tar = './RetinexNet/RetinexNet_PyTorch-master/ckpts4/Relight/6600.tar'
        ckpt_dict = torch.load(Relight_tar, map_location=self.device)
        self.RelightNet.load_state_dict(ckpt_dict)

    def predict(self, image, isContact = True):
        # 预测图像
        # start = time.time()
        image = np.array(image, dtype='float32') / 255.0
        image = np.transpose(image, (2, 0, 1))
        input_low_test = np.expand_dims(image, axis=0)
        self.forward(input_low_test, input_low_test)
        input = np.squeeze(input_low_test)
        result_4 = self.output_S
        input = np.squeeze(input)
        result_4 = np.squeeze(result_4)
        if isContact:
            cat_image = np.concatenate([input, result_4], axis=2)
        else:
            cat_image = np.concatenate([result_4], axis=2)
        cat_image = np.transpose(cat_image, (1, 2, 0))
        # print(time.time()-start)  # 时间在0.3-0.1之间 FPS大概在1-2每张图像
        cat_image = np.clip(cat_image * 255.0, 0, 255.0).astype('uint8')
        return cat_image