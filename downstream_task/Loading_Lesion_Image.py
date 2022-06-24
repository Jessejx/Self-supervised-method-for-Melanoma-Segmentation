import torch
import glob
import os
import cv2
import numpy as np
##### Data_Augmentation #####
import torchvision.transforms as transforms
from PIL import Image
import albumentations as A
import imgaug.augmenters as iaa
from albumentations.pytorch import ToTensorV2

def data_loader(mode):
    # Data Flag Check

    # Mode Flag Check
    if mode == 'train':
        shuffle = True
        dataset = TrainSet()
    elif mode == 'valid':
        shuffle = True
        dataset = ValidSet()
    elif mode == 'test':
        shuffle = False
        print("mode is False")
        # dataset = ValidSet(args)
    else:
        raise ValueError('data_loader mode ERROR')

    dataloader = torch.utils.data.DataLoader(dataset,
                            batch_size=16,
                            num_workers=12,
                            shuffle=shuffle,
                            drop_last=True)
    return dataloader


class TrainSet(torch.utils.data.Dataset):
    def __init__(self ):
        self.img_root = "to/isic2017/train"
        self.label_root = "to/isic2017/ISIC-2017_Training_Part1_GroundTruth"
        self.to_tensor = transforms.ToTensor()
        self.transform = A.Compose([
            A.Resize(224, 224),
            # A.HorizontalFlip(p=1),
            # A.RandomRotate90(p=1),
            # A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
            ])
        # get all images paths
        self.img_files = []
        self.img_files = sorted(glob.glob(os.path.join(self.img_root, "*.jpg")))
        # get all mask paths
        self.label_files = []
        self.label_files = sorted(glob.glob(os.path.join(self.label_root, "*.png")))

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img_path = self.img_files[index]
        mask_path = self.label_files[index]
        # img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        transformed = self.transform(image=img, mask=mask)
        img = transformed["image"]
        mask = transformed["mask"]
        mask_fore = np.expand_dims(np.array(mask), 0)
        # mask = mask.astype(np.float32)
        mask_back = np.absolute(mask_fore - 1)

        mask_fore = np.transpose(mask_fore, (1, 2, 0))
        mask_back = np.transpose(mask_back, (1, 2, 0))
        img = self.to_tensor(img)
        mask_fore = self.to_tensor(mask_fore)
        mask_back = self.to_tensor(mask_back)
        return img, mask_fore, mask_back


class ValidSet(torch.utils.data.Dataset):
    def __init__(self ):
        self.img_root = "to/isic2017/test"
        self.label_root = "to/isic2017/ISIC-2017_Test_v2_Part1_GroundTruth"
        self.to_tensor = transforms.ToTensor()
        self.transform = A.Compose([
            A.Resize(224, 224),
            ])
        # get all images paths
        self.img_files = []
        self.img_files = sorted(glob.glob(os.path.join(self.img_root, "*.jpg")))
        # get all mask paths
        self.label_files = []
        self.label_files = sorted(glob.glob(os.path.join(self.label_root, "*.png")))

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img_path = self.img_files[index]
        mask_path = self.label_files[index]
        # img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        transformed = self.transform(image=img, mask=mask)
        img = transformed["image"]
        mask = transformed["mask"]

        mask = np.expand_dims(np.array(mask), 0)
        mask = np.transpose(mask, (1, 2, 0))
        img = self.to_tensor(img)
        mask = self.to_tensor(mask)
        return img, mask, img_path

