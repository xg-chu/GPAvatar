import torch
import torch.nn as nn
import torchvision

# Learned perceptual metric
class PerceptualLoss(nn.Module):
    def __init__(
            self, net='alex', lpips=True, spatial=False, 
            use_dropout=True, model_path=None, eval_mode=True, verbose=True
        ):
        super().__init__()
        if net == 'vgg':
            self.net = VGG16()
            self.chns = [64,128,256,512,512]
            self.mean = [0.48235, 0.45882, 0.40784]
            self.std = [0.00392156862745098, 0.00392156862745098, 0.00392156862745098]
        elif net == 'alex':
            self.net = AlexNet()
            self.chns = [64,192,384,256,256]
            self.mean = [0.485, 0.456, 0.406]
            self.std = [0.229, 0.224, 0.225]
        else:
            raise NotImplementedError
        self.L = len(self.chns)
        self.norm = torchvision.transforms.Normalize(mean=self.mean, std=self.std, inplace=False)
        self.eval()

    def forward(self, img0, img1, normalize=False):
        if normalize: # turn on this flag if input is [-1, +1] so it can be adjusted to [0, 1]
            img0 = 0.5 * img0 + 0.5
            img1 = 0.5 * img1 + 0.5
        outs0, outs1 = self.net.forward(self.norm(img0)), self.net.forward(self.norm(img1))
        final_loss, losses = 0, {}
        for lid in range(self.L):
            feats0, feats1 = self.normalize_tensor(outs0[lid]), self.normalize_tensor(outs1[lid])
            losses[lid] = torch.nn.functional.mse_loss(feats0, feats1, reduction='none').sum(dim=1,keepdim=True).mean()
            final_loss += losses[lid]
        return final_loss

    @staticmethod
    def normalize_tensor(in_feat,eps=1e-10):
        norm_factor = torch.sqrt(torch.sum(in_feat**2,dim=1,keepdim=True))
        return in_feat/(norm_factor+eps)


class AlexNet(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super().__init__()
        alexnet_layers = torchvision.models.alexnet(weights="DEFAULT").features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(2):
            self.slice1.add_module(str(x), alexnet_layers[x])
        for x in range(2, 5):
            self.slice2.add_module(str(x), alexnet_layers[x])
        for x in range(5, 8):
            self.slice3.add_module(str(x), alexnet_layers[x])
        for x in range(8, 10):
            self.slice4.add_module(str(x), alexnet_layers[x])
        for x in range(10, 12):
            self.slice5.add_module(str(x), alexnet_layers[x])
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1 = h
        h = self.slice2(h)
        h_relu2 = h
        h = self.slice3(h)
        h_relu3 = h
        h = self.slice4(h)
        h_relu4 = h
        h = self.slice5(h)
        h_relu5 = h
        return (h_relu1, h_relu2, h_relu3, h_relu4, h_relu5)


class VGG16(torch.nn.Module):
    def __init__(self, ):
        super().__init__()
        vgg_layers = torchvision.models.vgg16(weights="DEFAULT").features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_layers[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_layers[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_layers[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_layers[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_layers[x])
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        return (h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)


class ColorLoss(torch.nn.Module):
    def __init__(self, ):
        super().__init__()
        self.pool = torch.nn.AvgPool2d(16, stride=16)
    
    def forward(self, img0, img1):
        color_loss = torch.mean((self.pool(img0) - self.pool(img1)).abs())
        return color_loss


if __name__ == '__main__':
    lfn = PerceptualLoss(net='alex')
    img0 = torch.rand(1,3,64,64) # image should be RGB, IMPORTANT: normalized to [-1,1]
    img1 = torch.rand(1,3,64,64)
    d = lfn(img0, img1)
    print(d)
