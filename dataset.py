import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from PIL import Image
import albumentations
import random
import math
random.seed(42)
########################### Data Augmentation ###########################
class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, mask=None):
        image = (image - self.mean) / self.std
        #mask /= 255
        return image#, mask


class RandomCrop(object):
    def __call__(self, image, mask=None):
        H, W, _ = image.shape
        randw = np.random.randint(W / 8)
        randh = np.random.randint(H / 8)
        offseth = 0 if randh == 0 else np.random.randint(randh)
        offsetw = 0 if randw == 0 else np.random.randint(randw)
        p0, p1, p2, p3 = offseth, H + offseth - randh, offsetw, W + offsetw - randw
        return image[p0:p1, p2:p3, :]#, mask[p0:p1, p2:p3]


class RandomFlip(object):
    def __call__(self, image, mask=None):
        if np.random.randint(2) == 0:
            return image[:, ::-1, :]#, mask[:, ::-1]
        else:
            return image#, mask


class Resize(object):
    def __init__(self, H, W):
        self.H = H
        self.W = W

    def __call__(self, image, mask=None):
        image = cv2.resize(image, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        #mask = cv2.resize(mask, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        return image#, mask


class ToTensor(object):
    def __call__(self, image, mask=None):
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1)
        #mask = torch.from_numpy(mask)
        return image#, mask


########################### Config File ###########################
class Config(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.mean = np.array([[[124.55, 118.90, 102.94]]])
        self.std = np.array([[[56.77, 55.97, 57.50]]])
        print('\nParameters...')
        for k, v in self.kwargs.items():
            print('%-10s: %s' % (k, v))

    def __getattr__(self, name):
        if name in self.kwargs:
            return self.kwargs[name]
        else:
            return None


########################### Dataset Class ###########################
class Data(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.normalize = Normalize(mean=cfg.mean, std=cfg.std)
        self.randomcrop = RandomCrop()
        self.randomflip = RandomFlip()
        self.resize = Resize(352, 352)
        self.totensor = ToTensor()
        with open(cfg.datapath + '/' + cfg.mode + '.txt', 'r') as lines:
            self.samples = []
            for line in lines:
                self.samples.append(line.strip())
        self.dict= {'epidural':0, 'healthy':1, 'intraparenchymal':2 ,'intraventricular':3, 'subarachnoid':4, 'subdural':5}
        self.data_transforms = {
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
        IMAGENET_SIZE = 256

        self.train_transform = albumentations.Compose([
            albumentations.Resize(IMAGENET_SIZE, IMAGENET_SIZE),
            albumentations.Normalize(mean=(0.456, 0.456, 0.456), std=(0.224, 0.224, 0.224), max_pixel_value=255.0,
                                     p=1.0)
        ])

        self.val_transform = albumentations.Compose([
            albumentations.Resize(IMAGENET_SIZE, IMAGENET_SIZE),
            albumentations.Normalize(mean=(0.456, 0.456, 0.456), std=(0.224, 0.224, 0.224), max_pixel_value=255.0,
                                     p=1.0)
        ])
    def __getitem__(self, idx):
        name = self.samples[idx]
        classname = name.split('\\')[-2]
        image = cv2.imread(name,0).astype(np.float32)
        image_cat = np.concatenate([image[:, :, np.newaxis], image[:, :, np.newaxis], image[:, :, np.newaxis]],
                                   2)
        #img = Image.open(name)
        #image = img.convert('RGB')
        y = np.zeros((1,1))
        id = self.dict[classname]
        y[0,0]=id
        #mask = cv2.imread(self.cfg.datapath + '/mask/' + name + '.png', 0).astype(np.float32)
        #shape = mask.shape

        if 'train' in self.cfg.mode:
            #image = self.normalize(image)
            #image = self.randomcrop(image)
            #image = self.randomflip(image)
            #image = cv2.resize(image, dsize=(224, 224), interpolation=cv2.INTER_LINEAR)
            #image = torch.from_numpy(np.stack(image, axis=0)).permute(0, 3, 1, 2).type(torch.FloatTensor)
            if random.random() < 0.5:
                image_cat = cv2.cvtColor(image_cat, cv2.COLOR_BGR2RGB)

            image_cat = aug_image(image_cat, is_infer=False)

            augmented = self.train_transform(image=image_cat)
            image_cat = augmented['image'].transpose(2, 0, 1)
            #image = self.data_transforms['train'](image)
            y = torch.from_numpy(np.stack(y, axis=0)).squeeze().type(torch.LongTensor)
            return image_cat, y
        else:
            #image = self.normalize(image)
            #image = self.randomcrop(image)
            #image = cv2.resize(image, dsize=(224, 224), interpolation=cv2.INTER_LINEAR)
            #image = torch.from_numpy(np.stack(image, axis=0)).permute(0, 3, 1, 2).type(torch.FloatTensor)
            #image = self.data_transforms['val'](image)
            image_cat = aug_image(image_cat, is_infer=True)

            augmented = self.val_transform(image=image_cat)
            image_cat = augmented['image'].transpose(2, 0, 1)
            y = torch.from_numpy(np.stack(y, axis=0)).squeeze().type(torch.LongTensor)
            return image_cat, y

    def collate(self, batch):
        size = [224, 256, 288, 320, 352][np.random.randint(0, 5)]
        image, y = [list(item) for item in zip(*batch)]
        for i in range(len(batch)):
            image[i] = cv2.resize(image[i], dsize=(size, size), interpolation=cv2.INTER_LINEAR)
            #mask[i] = cv2.resize(mask[i], dsize=(size, size), interpolation=cv2.INTER_LINEAR)
        image = torch.from_numpy(np.stack(image, axis=0)).permute(0, 3, 1, 2).type(torch.FloatTensor)
        y = torch.from_numpy(np.stack(y, axis=0)).squeeze().type(torch.LongTensor)
        return image, y

    def __len__(self):
        return len(self.samples)

def randomHorizontalFlip(image, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 1)
    return image

def randomVerticleFlip(image, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 0)
    return image

def randomRotate90(image, u=0.5):
    if np.random.random() < u:
        image[:,:,0:3] = np.rot90(image[:,:,0:3])
    return image

#===================================================origin=============================================================
def random_cropping(image, ratio=0.8, is_random = True):
    height, width, _ = image.shape
    target_h = int(height*ratio)
    target_w = int(width*ratio)

    if is_random:
        start_x = random.randint(0, width - target_w)
        start_y = random.randint(0, height - target_h)
    else:
        start_x = ( width - target_w ) // 2
        start_y = ( height - target_h ) // 2

    zeros = image[start_y:start_y+target_h,start_x:start_x+target_w,:]
    zeros = cv2.resize(zeros ,(width,height))
    return zeros

def cropping(image, ratio=0.8, code = 0):
    height, width, _ = image.shape
    target_h = int(height*ratio)
    target_w = int(width*ratio)

    if code==0:
        start_x = ( width - target_w ) // 2
        start_y = ( height - target_h ) // 2

    elif code == 1:
        start_x = 0
        start_y = 0

    elif code == 2:
        start_x = width - target_w
        start_y = 0

    elif code == 3:
        start_x = 0
        start_y = height - target_h

    elif code == 4:
        start_x = width - target_w
        start_y = height - target_h

    elif code == -1:
        return image

    zeros = image[start_y:start_y+target_h,start_x:start_x+target_w,:]
    zeros = cv2.resize(zeros ,(width,height))
    return zeros

def random_erasing(img, probability=0.5, sl=0.02, sh=0.4, r1=0.3):
    if random.uniform(0, 1) > probability:
        return img

    for attempt in range(100):
        area = img.shape[0] * img.shape[1]

        target_area = random.uniform(sl, sh) * area
        aspect_ratio = random.uniform(r1, 1 / r1)

        h = int(round(math.sqrt(target_area * aspect_ratio)))
        w = int(round(math.sqrt(target_area / aspect_ratio)))

        if w < img.shape[1] and h < img.shape[0]:
            x1 = random.randint(0, img.shape[0] - h)
            y1 = random.randint(0, img.shape[1] - w)
            if img.shape[2] == 3:
                img[x1:x1 + h, y1:y1 + w, :] = 0.0
            else:
                print('!!!!!!!! random_erasing dim wrong!!!!!!!!!!!')
                return

            return img
    return img

def randomShiftScaleRotate(image,
                           shift_limit=(-0.0, 0.0),
                           scale_limit=(-0.0, 0.0),
                           rotate_limit=(-0.0, 0.0),
                           aspect_limit=(-0.0, 0.0),
                           borderMode=cv2.BORDER_CONSTANT, u=0.5):

    if np.random.random() < u:
        height, width, channel = image.shape

        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                    borderValue=(
                                        0, 0,
                                        0,))
    return image
def aug_image(image, is_infer=False):
    if is_infer:
        image = randomHorizontalFlip(image, u=0)
        image = np.asarray(image)
        image = cropping(image, ratio=0.8, code=0)
        return image

    else:
        image = randomHorizontalFlip(image)
        height, width, _ = image.shape
        image = randomShiftScaleRotate(image,
                                       shift_limit=(-0.1, 0.1),
                                       scale_limit=(-0.1, 0.1),
                                       aspect_limit=(-0.1, 0.1),
                                       rotate_limit=(-30, 30))

        image = cv2.resize(image, (width, height))
        image = random_erasing(image, probability=0.5, sl=0.02, sh=0.4, r1=0.3)

        ratio = random.uniform(0.6,0.99)
        image = random_cropping(image, ratio=ratio, is_random=True)
        return image
