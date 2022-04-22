import torch.nn as nn
import torch
import os
from .utils import  create_dir
from torchvision import models as tm
import torch.optim as optim


def get_net(num_classes):
    #return MyModel(torchvision.models.resnet101(pretrained = True))
    model = tm.resnext50_32x4d(pretrained=True)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    model.fc = nn.Linear(2048, num_classes)
    model.cuda()
    return model
def accuracy(output, label, topk=(1,)):

    with torch.no_grad():

        batch_size = label.size(0)

        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()

        _,target=label.topk(1, 1, True, True)
        target=target.t()
        correct = pred.eq(target)

        correct_k = correct[:1].view(-1).float().sum(0, keepdim=True)


        return correct_k.mul_(100.0 / batch_size)
class ClassificationModel(nn.Module):

    def __init__(self, model_name,name, config):
        super(ClassificationModel, self).__init__()
        model = get_net(num_classes=2)
        if len(config.GPU) > 1:
            model = nn.DataParallel(model, config.GPU)
        self.add_module('model', model)

        self.optimizer = optim.Adam(
            params=model.parameters(),
            lr=float(config.LR),
            betas=(config.BETA1, config.BETA2)
        )


        self.name = name
        self.config = config
        self.iteration = 0
        create_dir(os.path.join(config.PATH, model_name))
        self.weights_path = os.path.join(config.PATH, os.path.join(model_name, name + '.pth'))

        self.criterion = nn.BCELoss().cuda()   #nn.CrossEntropyLoss().cuda()


    def load(self):
        if os.path.exists(self.weights_path):
            print('Loading %s generator...' % self.name)

            if torch.cuda.is_available():
                data = torch.load(self.weights_path)
            else:
                data = torch.load(self.weights_path, map_location=lambda storage, loc: storage)

            self.model.load_state_dict(data['model'])
            self.iteration = data['iteration']

    def save(self):
        print('\nsaving %s...\n' % self.name)
        torch.save({
            'iteration': self.iteration,
            'model': self.model.state_dict()
        }, self.weights_path)

    def process(self, images,labels):
        self.iteration += 1


        # zero optimizers
        self.optimizer.zero_grad()


        # process outputs
        outputs = self(images)

        loss = self.criterion(outputs, labels)

        precision = accuracy(outputs, labels, topk=(1))[0]

        # create logs
        logs = [
            ("loss", loss.item()),
            ("precision",precision.item())
        ]

        return outputs,loss, logs,precision

    def forward(self,images):
        input=images
        outputs = self.model(input)
        m=nn.Sigmoid()
        outputs=m(outputs)
        return outputs

    def backward(self, loss=None):
        if loss is not None:
            loss.backward()
        self.optimizer.step()
