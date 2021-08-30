import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, c_in, c_out,is_downsample=False):
        super(BasicBlock,self).__init__()
        self.is_downsample = is_downsample
        if is_downsample:
            self.conv1 = nn.Conv2d(c_in, c_out, 3, stride=2, padding=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(c_in, c_out, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(c_out,c_out,3,stride=1,padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        if is_downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride=2, bias=False),
                nn.BatchNorm2d(c_out)
            )
        elif c_in != c_out:
            self.downsample = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride=1, bias=False),
                nn.BatchNorm2d(c_out)
            )
            self.is_downsample = True

    def forward(self,x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.bn2(y)
        if self.is_downsample:
            x = self.downsample(x)
        return F.relu(x.add(y),True)
    
    def _quant_forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.bn2(y)
        if self.is_downsample:
            x = self.downsample(x)
        x = self.dequant(x)
        y = self.dequant(y)
        x = x.add(y)
        x = self.requant(x)
        return F.relu(x,True)
    
    def quantize(self):
        self.dequant = torch.quantization.DeQuantStub()
        self.requant = torch.quantization.QuantStub()


def make_layers(c_in,c_out,repeat_times, is_downsample=False):
    blocks = []
    for i in range(repeat_times):
        if i ==0:
            blocks += [BasicBlock(c_in,c_out, is_downsample=is_downsample),]
        else:
            blocks += [BasicBlock(c_out,c_out),]
    return nn.Sequential(*blocks)

class Net(nn.Module):
    def __init__(self, num_classes=751 ,reid=False):
        super(Net,self).__init__()
        # 3 128 64
        self.conv = nn.Sequential(
            nn.Conv2d(3,24,3,stride=1,padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            # nn.Conv2d(32,32,3,stride=1,padding=1),
            # nn.BatchNorm2d(32),
            # nn.ReLU(inplace=True),
            nn.MaxPool2d(3,2,padding=1),
        )
        # 32 64 32
        self.layer1 = make_layers(24,24,2,False)
        # 32 64 32
        self.layer2 = make_layers(24,32,2,True)
        # 64 32 16
        self.layer3 = make_layers(32,48,2,True)
        # 128 16 8
        self.layer4 = make_layers(48,64,2,True)
        # 256 8 4
        self.avgpool = nn.AvgPool2d((8,4),1)
        # 256 1 1 
        self.reid = reid
        self.classifier = nn.Sequential(
            nn.Linear(64, 72),
            nn.BatchNorm1d(72),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(72, num_classes),
        )
    
    def forward(self, x):
        x = self.conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0),-1)
        # B x 128
        if self.reid:
            x = x.div(x.norm(p=2,dim=1,keepdim=True))
            return x
        # classifier
        x = self.classifier(x)
        return x

    def _quant_forward(self, x):
        x = self.quant(x)
        x = self.conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0),-1)
        # B x 128
        if self.reid:
            x = x.div(x.norm(p=2,dim=1,keepdim=True))
            return x
        # classifier
        x = self.classifier(x)
        x = self.dequant(x)
        return x

    def fuse_modules(self):
        conv = nn.ModuleList([layer for layer in self.conv.children()])
        conv = torch.quantization.fuse_modules(conv, [['0', '1', '2']], inplace=True)  # 0, 1, 2 -> conv-bn-relu
        self.conv = nn.Sequential(*conv)

        layer1 = self._fuse_basicblock(self.layer1)
        layer2 = self._fuse_basicblock(self.layer2)
        layer3 = self._fuse_basicblock(self.layer3)
        layer4 = self._fuse_basicblock(self.layer4)

        self.layer1 = nn.Sequential(*layer1)
        self.layer2 = nn.Sequential(*layer2)
        self.layer3 = nn.Sequential(*layer3)
        self.layer4 = nn.Sequential(*layer4)

        classifier = nn.ModuleList([layer for layer in self.classifier.children()])
        classifier = torch.quantization.fuse_modules(classifier, [['0', '1']], inplace=True)  # 0, 1 -> linear-bn
        self.classifier = nn.Sequential(*classifier)

    def _fuse_basicblock(self, layer):
        sublayers = nn.ModuleList([layer for layer in layer.children()])
        for basicblock in sublayers:
            if type(basicblock) != BasicBlock:
                continue
            basicblock = torch.quantization.fuse_modules(basicblock, [['conv1', 'bn1', 'relu'], ['conv2', 'bn2']], inplace=True)
            if basicblock.is_downsample:
                downsample = nn.ModuleList([layer for layer in basicblock.downsample.children()])
                downsample = torch.quantization.fuse_modules(downsample, [['0', '1']], inplace=True)  # 0, 1 -> conv-bn
                basicblock.downsample = nn.Sequential(*downsample)
        return sublayers

    '''
        activation_dataset -> torch.Tensor([B, 3, 128, 64]) -> (B is mandatory)
    '''
    def quantize(self, activation_dataset):
        self.eval()

        self.qconfig = torch.quantization.get_default_qconfig('qnnpack')
        
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        self.forward = self._quant_forward

        layer1 = self._quant_basicblock(self.layer1)
        layer2 = self._quant_basicblock(self.layer2)
        layer3 = self._quant_basicblock(self.layer3)
        layer4 = self._quant_basicblock(self.layer4)

        self.layer1 = nn.Sequential(*layer1)
        self.layer2 = nn.Sequential(*layer2)
        self.layer3 = nn.Sequential(*layer3)
        self.layer4 = nn.Sequential(*layer4)

        prepared_model = torch.quantization.prepare(self)
        prepared_model(activation_dataset)  # activate model

        return torch.quantization.convert(prepared_model)

    def _quant_basicblock(self, layer):
        sublayers = nn.ModuleList([layer for layer in layer.children()])
        for basicblock in sublayers:
            if type(basicblock) != BasicBlock:
                continue
            basicblock.quantize()
            basicblock.forward = basicblock._quant_forward
        return sublayers

if __name__ == '__main__':
    # for inference:
    # torch.backends.quantized.engine = 'qnnpack'

    model = Net()
    model.eval()
    
    state_dict = torch.load('/root/submission-test/solution/deep_sort/deep_sort/deep/checkpoint/ckpt.t7', map_location=torch.device('cpu'))['net_dict']
    model.load_state_dict(state_dict)
    model.fuse_modules()

    # calibrate the prepared model to determine quantization parameters for activations
    # in a real world setting, the calibration would be done with a representative dataset
    def generate_activation_market1501():
        from glob import glob
        import os
        import numpy as np
        import cv2
        import random
        
        root_dir=r'C:\Dataset\Market-1501-v15.09.15\train'
        all_files = list(glob(os.path.join(root_dir, '*', '*')))
        random.shuffle(all_files)
        
        top_n = 32
        top_n_files = all_files[:top_n]
        
        means = np.expand_dims(np.expand_dims(np.array([0.485, 0.456, 0.406]), 0), 0)
        stds = np.expand_dims(np.expand_dims(np.array([0.229, 0.224, 0.225]), 0), 0)
        size = (64, 128)
        
        image_batch = []
        for file in top_n_files:
            image = cv2.imread(file)
            image = cv2.resize(image, size)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = (image / 255.0).astype(np.float32)
            image -= means
            image /= stds
            image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
            image_batch.append(image)

        image_batch = torch.cat(image_batch, dim=0)
        print(image_batch.shape)
        
        return image_batch

    input_fp32 = generate_activation_market1501()
    model_int8 = model.quantize(input_fp32)

    # run the model, relevant calculations will happen in int8
    res = model_int8(input_fp32)

    int8_model = torch.jit.trace(model_int8, input_fp32)
    torch.jit.save(int8_model, 'deep_int8_model.torchscript')
