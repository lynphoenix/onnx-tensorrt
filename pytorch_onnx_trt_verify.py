import os
import onnx
import torch
import torch.onnx
import torchvision
import argparse
import math
from PIL import Image
from torch.autograd import Variable
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
 
import onnx_tensorrt.backend as backend
import subprocess


class BidirectionalLSTM(nn.Module):
    # Inputs hidden units Out
    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output

    
class LeNet_A(nn.Module):
    
    def __init__(self, init_weights=False):
        super(LeNet_A, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)


        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class LeNet_B(nn.Module):
    
    def __init__(self, init_weights=False):
        super(LeNet_B, self).__init__()
        nh = 256
        nclass = 10
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 16*5*5)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

                
class LeNet_C(nn.Module):
    
    def __init__(self, init_weights=False):
        super(LeNet_C, self).__init__()
        nh = 256
        nclass = 10
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.rnn = nn.Sequential(
            BidirectionalLSTM(400, nh, nh),
            # BidirectionalLSTM(nh, nh, nclass)
        )

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 1, 16*5*5)
        print(x.size())
        x = self.rnn(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

                
def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trainroot', required=True, help='path to dataset')
    parser.add_argument('--valroot', required=True, help='path to dataset')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    args = parser.parse_args()
    return args


def create_pytorch_model():
    model = LeNet_C(init_weights=True).cuda()
    print(model)
    return model


def pytorch2trace(image, model):
    traced_script_module = torch.jit.trace(model, image)
    traced_script_module.save('lenet.pt')


def pytorch_infer(image, model):
    model.eval()
    preds = model(image)
    print(preds)
    return preds

def onnx_infer(image, model_path):
    model = onnx.load(model_path)
    engine = backend.prepare(model, device='CUDA:1')
    # input_data = np.random.random(size=(32, 3, 224, 224)).astype(np.float32)
    output_data = engine.run(image)[0]
    print(output_data)
    print(output_data.shape)

    
def pytorch2onnx(image, pytorch_model, model_path):
    torch_out = torch.onnx._export(pytorch_model,  # model being run
                               image,  # model input (or a tuple for multiple inputs)
                               model_path,  # where to save the model
                               export_params=True)

    return torch_out


def onnx2tensorrt(onnx_model_path, trt_model_path):
    cmd_str = './build/onnx2trt ' + onnx_model_path + ' -o ' + trt_model_path
    print(os.system(cmd_str))


if __name__ == '__main__':

    # 创建pytorch模型
    pytorch_model = create_pytorch_model()
    # pytorch模型推理
    image_data = torch.randn(1, 3, 32, 32)
    image = Variable(image_data).cuda()
    preds = pytorch_infer(image, pytorch_model)

    # 转ONNX模型
    onnx_model_path = 'model.onnx'
    onnx_model = pytorch2onnx(image, pytorch_model, onnx_model_path)
    # ONNX模型推理
    onnx_infer(image.cpu().numpy(), onnx_model_path)
    
    # ONNX模型转TensorRT模型
    trt_model_path = 'model.trt'
    trt_model = onnx2tensorrt(onnx_model_path, trt_model_path)
    
