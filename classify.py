
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import pretrainedmodels
import dataset
from torch.utils.data import DataLoader
import math
import resnet
from tensorboardX import SummaryWriter
import random
from densenet import DenseNet169_change_avg, DenseNet121_change_avg
#from tuils.lrs_scheduler import WarmRestart, warm_restart, AdamW, RAdam
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


class WarmRestart(lr_scheduler.CosineAnnealingLR):
    """This class implements Stochastic Gradient Descent with Warm Restarts(SGDR): https://arxiv.org/abs/1608.03983.

    Set the learning rate of each parameter group using a cosine annealing schedule, When last_epoch=-1, sets initial lr as lr.
    This can't support scheduler.step(epoch). please keep epoch=None.
    """

    def __init__(self, optimizer, T_max=10, T_mult=2, eta_min=0, last_epoch=-1):
        """implements SGDR

        Parameters:
        ----------
        T_max : int
            Maximum number of epochs.
        T_mult : int
            Multiplicative factor of T_max.
        eta_min : int
            Minimum learning rate. Default: 0.
        last_epoch : int
            The index of last epoch. Default: -1.
        """
        self.T_mult = T_mult
        super().__init__(optimizer, T_max, eta_min, last_epoch)

    def get_lr(self):
        if self.last_epoch == self.T_max:
            self.last_epoch = 0
            self.T_max *= self.T_mult
        return [self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2 for
                base_lr in self.base_lrs]


def train_model(model, loader, val_loader, criterion, optimizer, scheduler, dataset_sizes, num_epochs=30):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    global_step = 0
    val_global_step = 0
    logname = 'log_fold1'
    if not os.path.exists(logname):
        os.mkdir(logname)
    sw = SummaryWriter(logname)
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train','val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            if phase =='train':
                for step, (inputs, labels) in enumerate(loader):
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                   # print(labels.size())
                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    global_step += 1
                    sw.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=global_step)

            else:
                for step, (inputs, labels) in enumerate(val_loader):
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    # print(labels.size())
                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                    val_global_step+=1
                    #sw.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=val_global_step)

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            if phase == 'train':
                sw.add_scalars('loss', {'training_loss': epoch_loss}, global_step=epoch+1)
                sw.add_scalars('accuracy', {'training_acc': epoch_acc}, global_step=epoch + 1)
            else:
                sw.add_scalars('loss', {'validation_loss': epoch_loss}, global_step=epoch+1)
                sw.add_scalars('accuracy', {'validation_acc': epoch_acc}, global_step=epoch + 1)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())





    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


if __name__ == '__main__':
    # Data augmentation and normalization for training
    # Just normalization for validation
    torch.manual_seed(1992)
    torch.cuda.manual_seed(1992)
    np.random.seed(1992)
    random.seed(1992)
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    data_dir = 'data'
    cfg = dataset.Config(datapath='.',
                         savepath='./out', mode='train_2',
                         batch=20)
    data = dataset.Data(cfg)
    loader = DataLoader(data, batch_size=cfg.batch, shuffle=True)
    cfg = dataset.Config(datapath='.', savepath='./out', mode='val_2')
    valdata = dataset.Data(cfg)
    valoader = DataLoader(valdata, batch_size=20, shuffle=False)
    dataset_sizes={'train':len(data), 'val':len(valdata)}
    print(len(data))
    #class_names = image_datasets['train'].classes
    #print(class_names)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Get a batch of training data
    #inputs, classes = next(iter(dataloaders['train']))
    ###########ResNet50############
    model_ft = models.resnet50(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 6)
    #######se_resnext101_32x4d##########
    #model_ft = pretrainedmodels.__dict__['se_resnext50_32x4d'](num_classes=1000, pretrained='imagenet')
    #num_ftrs = model_ft.last_linear.in_features
    #model_ft.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
    #model_ft.last_linear = nn.Sequential(nn.Linear(num_ftrs, 6, bias=True))
    ######gct_resnet########
    #model_ft = resnet.ResNet([])
    #checkpoint = torch.load('gct_resnet50.pth')
    #model_dict = model_ft.state_dict()
    #pretrained_dict = {k[6:]: v for k, v in checkpoint['state_dict'].items()
    #                   if k[6:] in model_dict.keys()}
    #model_dict.update(pretrained_dict)
    #model_ft.load_state_dict(model_dict)
    #model_ft.fc = nn.Sequential(nn.Linear(512 * 4, 6, bias=True))
    ######DenseNet169_change_avg#######
    #model_ft = DenseNet169_change_avg()
    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
   # optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    #exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    optimizer_ft = torch.optim.Adam(model_ft.parameters(), lr=0.0005, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.00002)
    exp_lr_scheduler = WarmRestart(optimizer_ft, T_max=5, T_mult=1, eta_min=1e-5)
    torch.save(model_ft.state_dict(),
               os.path.join('init_state.pth'))
    model_ft = train_model(model_ft, loader, valoader, criterion, optimizer_ft, exp_lr_scheduler, dataset_sizes,
                           num_epochs=40)
    torch.save(model_ft.state_dict(),
               os.path.join('best_state.pth'))
