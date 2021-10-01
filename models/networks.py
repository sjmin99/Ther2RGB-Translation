import torch
import torch.nn as nn
import functools
from torch.autograd import Variable
import numpy as np

###############################################################################
# Functions
###############################################################################
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def define_G(input_nc, output_nc, ngf, netG, n_downsample_global=3, n_blocks_global=9, n_local_enhancers=1, 
             n_blocks_local=3, norm='instance', use_noise=False, gpu_ids=[]):    
    norm_layer = get_norm_layer(norm_type=norm)     

    netG = SIGGRAPHGenerator(input_nc, output_nc,norm_layer, use_noise = use_noise, use_tanh=True)

    print(netG)
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())   
        netG.cuda(gpu_ids[0])
    netG.apply(weights_init)
    return netG

def define_D(input_nc, ndf, n_layers_D, norm='instance', use_sigmoid=False, num_D=1, getIntermFeat=False, gpu_ids=[]):        
    norm_layer = get_norm_layer(norm_type=norm)   
    netD = MultiscaleDiscriminator(input_nc, ndf, n_layers_D, norm_layer, use_sigmoid, num_D, getIntermFeat)   
    print(netD)
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        netD.cuda(gpu_ids[0])
    netD.apply(weights_init)
    return netD

def print_network(net):
    if isinstance(net, list):
        net = net[0]
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

##############################################################################
# Losses
##############################################################################
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss =  nn.BCEWithLogitsLoss() 

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        else:            
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)

class VGGLoss(nn.Module):
    def __init__(self, gpu_ids):
        super(VGGLoss, self).__init__()        
        self.vgg = Vgg19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]        

    def forward(self, x, y):     
        if x.shape[1]==1:
            x=torch.cat([x,x,x],dim=1)
        if y.shape[1]==1:
            y=torch.cat([y,y,y],dim=1)
        
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())        
        return loss

class HuberLoss(nn.Module):
    def __init__(self, delta=.01):
        super(HuberLoss, self).__init__()
        self.delta=delta

    def __call__(self, in0, in1):
        mask = torch.zeros_like(in0)
        mann = torch.abs(in0-in1)
        eucl = .5 * (mann**2)
        mask[...] = mann < self.delta

        # loss = eucl*mask + self.delta*(mann-.5*self.delta)*(1-mask)
        loss = eucl*mask/self.delta + (mann-.5*self.delta)*(1-mask)
        return torch.sum(loss,dim=1,keepdim=True)
##############################################################################
# Generator
##############################################################################
class SIGGRAPHGenerator(nn.Module):
    def __init__(self, input_nc, output_nc ,norm_layer=nn.BatchNorm2d, use_noise=False, use_tanh=True):
        super(SIGGRAPHGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.use_noise = use_noise
        use_bias = True
        
        # Conv1
        # model1=[nn.ReflectionPad2d(1),]
        ###백본 코드가 아무래도 사용자의 point label값도 사용하다보니, input 채널이 original input 1+user input 2 + mask? 1 해서 4채널임.
        model1=[nn.Conv2d(input_nc, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        # model1+=[norm_layer(64),]
        model1+=[nn.ReLU(True),]
        # model1+=[nn.ReflectionPad2d(1),]
        model1+=[nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),]

        model1+=[nn.ReLU(True),]
        model1+=[norm_layer(64),]
        # add a subsampling operation

        # Conv2
        # model2=[nn.ReflectionPad2d(1),]
        model2=[nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=use_bias),]
        # model2+=[norm_layer(128),]
        model2+=[nn.ReLU(True),]
        # model2+=[nn.ReflectionPad2d(1),]
        model2+=[nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=use_bias),]

        model2+=[nn.ReLU(True),]
        model2+=[norm_layer(128),]
        # add a subsampling layer operation

        # Conv3
        # model3=[nn.ReflectionPad2d(1),]
        model3=[nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=use_bias),]
        # model3+=[norm_layer(256),]
        model3+=[nn.ReLU(True),]
        # model3+=[nn.ReflectionPad2d(1),]
        model3+=[nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        # model3+=[norm_layer(256),]
        model3+=[nn.ReLU(True),]
        # model3+=[nn.ReflectionPad2d(1),]
        model3+=[nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=use_bias),]

        model3+=[nn.ReLU(True),]
        model3+=[norm_layer(256),]
        # add a subsampling layer operation

        # Conv4
        # model47=[nn.ReflectionPad2d(1),]
        model4=[nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=use_bias),]
        # model4+=[norm_layer(512),]
        model4+=[nn.ReLU(True),]
        # model4+=[nn.ReflectionPad2d(1),]
        model4+=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        # model4+=[norm_layer(512),]
        model4+=[nn.ReLU(True),]
        # model4+=[nn.ReflectionPad2d(1),]
        model4+=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        model4+=[nn.ReLU(True),]
        model4+=[norm_layer(512),]

        # Conv5
        # model47+=[nn.ReflectionPad2d(2),]
        model5=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=use_bias),]
        # model5+=[norm_layer(512),]
        model5+=[nn.ReLU(True),]
        # model5+=[nn.ReflectionPad2d(2),]
        model5+=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=use_bias),]
        # model5+=[norm_layer(512),]
        model5+=[nn.ReLU(True),]
        # model5+=[nn.ReflectionPad2d(2),]
        model5+=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=use_bias),]
        model5+=[nn.ReLU(True),]
        model5+=[norm_layer(512),]

        # Conv6
        # model6+=[nn.ReflectionPad2d(2),]
        model6=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=use_bias),]
        # model6+=[norm_layer(512),]
        model6+=[nn.ReLU(True),]
        # model6+=[nn.ReflectionPad2d(2),]
        model6+=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=use_bias),]
        # model6+=[norm_layer(512),]
        model6+=[nn.ReLU(True),]
        # model6+=[nn.ReflectionPad2d(2),]
        model6+=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=use_bias),]
        model6+=[nn.ReLU(True),]
        model6+=[norm_layer(512),]

        # Conv7
        # model47+=[nn.ReflectionPad2d(1),]
        model7=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        # model7+=[norm_layer(512),]
        model7+=[nn.ReLU(True),]
        # model7+=[nn.ReflectionPad2d(1),]
        model7+=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        # model7+=[norm_layer(512),]
        model7+=[nn.ReLU(True),]
        # model7+=[nn.ReflectionPad2d(1),]
        model7+=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        model7+=[nn.ReLU(True),]
        model7+=[norm_layer(512),]

        # Conv7
        model8up=[nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=use_bias)]

        # model3short8=[nn.ReflectionPad2d(1),]
        model3short8=[nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=use_bias),]

        # model47+=[norm_layer(256),]
        model8=[nn.ReLU(True),]
        # model8+=[nn.ReflectionPad2d(1),]
        model8+=[nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        # model8+=[norm_layer(256),]
        model8+=[nn.ReLU(True),]
        # model8+=[nn.ReflectionPad2d(1),]
        model8+=[nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=use_bias),]

        model8+=[nn.ReLU(True),]
        model8+=[norm_layer(256),]

        # Conv9
        model9up=[nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=use_bias),]

        # model2short9=[nn.ReflectionPad2d(1),]
        model2short9=[nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        # add the two feature maps above        

        # model9=[norm_layer(128),]
        model9=[nn.ReLU(True),]
        # model9+=[nn.ReflectionPad2d(1),]
        model9+=[nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=use_bias),]

        model9+=[nn.ReLU(True),]
        model9+=[norm_layer(128),]

        # Conv10
        model10up=[nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1, bias=use_bias),]

        # model1short10=[nn.ReflectionPad2d(1),]
        model1short10=[nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        # add the two feature maps above

        # model10=[norm_layer(128),]
        model10=[nn.ReLU(True),]
        # model10+=[nn.ReflectionPad2d(1),]
        model10+=[nn.Conv2d(128, 128, kernel_size=3, dilation=1, stride=1, padding=1, bias=use_bias),]
        model10+=[nn.LeakyReLU(negative_slope=.2),]

        # regression output
        model_out=[nn.Conv2d(128, self.output_nc, kernel_size=1, padding=0, dilation=1, stride=1, bias=use_bias),]
        if(use_tanh):
            model_out+=[nn.Tanh()]

        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)
        self.model4 = nn.Sequential(*model4)
        self.model5 = nn.Sequential(*model5)
        self.model6 = nn.Sequential(*model6)
        self.model7 = nn.Sequential(*model7)
        self.model8up = nn.Sequential(*model8up)
        self.model8 = nn.Sequential(*model8)
        self.model9up = nn.Sequential(*model9up)
        self.model9 = nn.Sequential(*model9)
        self.model10up = nn.Sequential(*model10up)
        self.model10 = nn.Sequential(*model10)
        self.model3short8 = nn.Sequential(*model3short8)
        self.model2short9 = nn.Sequential(*model2short9)
        self.model1short10 = nn.Sequential(*model1short10)

        self.model_out = nn.Sequential(*model_out)

    def forward(self, input_A):
        
        # conv1_2 = self.model1(torch.cat((input_A,input_B,mask_B),dim=1))
        
        conv1_2 = self.model1(input_A)#(3,256,256) -> (64,256,256)
        conv2_2 = self.model2(conv1_2)#(64,256,256) -> (128,128,128)
        conv3_3 = self.model3(conv2_2)#(128,128,128) -> (256,64,64)
        conv4_3 = self.model4(conv3_3)#(256,64,64) -> (512,32,32)
        conv5_3 = self.model5(conv4_3)#(512,32,32) -> (512,32,32)
        conv6_3 = self.model6(conv5_3)#(512,32,32) -> (512,32,32)
        conv7_3 = self.model7(conv6_3)#(512,32,32) -> (512,32,32)

        conv8_up = self.model8up(conv7_3) + self.model3short8(conv3_3) 
        conv8_3 = self.model8(conv8_up)

        conv9_up = self.model9up(conv8_3) + self.model2short9(conv2_2)
        conv9_3 = self.model9(conv9_up)
        conv10_up = self.model10up(conv9_3) + self.model1short10(conv1_2)
        conv10_2 = self.model10(conv10_up)
        
        out_reg = self.model_out(conv10_2)

        return out_reg

# Define a resnet block
class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, 
                 use_sigmoid=False, num_D=3, getIntermFeat=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat
     
        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)
            if getIntermFeat:                                
                for j in range(n_layers+2):
                    setattr(self, 'scale'+str(i)+'_layer'+str(j), getattr(netD, 'model'+str(j)))                                   
            else:
                setattr(self, 'layer'+str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):        
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale'+str(num_D-1-i)+'_layer'+str(j)) for j in range(self.n_layers+2)]
            else:
                model = getattr(self, 'layer'+str(num_D-1-i))

            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D-1):
                input_downsampled = self.downsample(input_downsampled)
        return result
        
# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)        

from torchvision import models
class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)        
        h_relu3 = self.slice3(h_relu2)        
        h_relu4 = self.slice4(h_relu3)        
        h_relu5 = self.slice5(h_relu4)                
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out
