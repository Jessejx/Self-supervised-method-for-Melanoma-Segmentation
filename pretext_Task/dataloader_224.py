import random
import torchvision
from torchvision.datasets import DatasetFolder
from utils import pil_loader
import cv2

class JigsawLoader(DatasetFolder):
    def __init__(self, root_dir):
        super(JigsawLoader, self).__init__(root_dir, pil_loader, extensions=('jpg'))
        self.root_dir = root_dir
        self.color_transform = torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4)
        self.flips = [torchvision.transforms.RandomHorizontalFlip(), torchvision.transforms.RandomVerticalFlip()]
        self.normalize = torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, _ = self.samples[index]
        original = self.loader(path)
        image = torchvision.transforms.Resize((224, 224))(original)
        sample = torchvision.transforms.RandomCrop((255, 255))(original)

        crop_areas = [(i*85, j*85, (i+1)*85, (j+1)*85) for i in range(3) for j in range(3)]
        samples = [sample.crop(crop_area) for crop_area in crop_areas]
        samples = [torchvision.transforms.RandomCrop((64, 64))(patch) for patch in samples]
        # augmentation collor jitter
        image = self.color_transform(image)
        samples = [self.color_transform(patch) for patch in samples]
        # augmentation - flips
        # image = self.flips[0](image)
        # image = self.flips[1](image)
        # to tensor
        image = torchvision.transforms.functional.to_tensor(image)
        samples = [torchvision.transforms.functional.to_tensor(patch) for patch in samples]
        # normalize
        # image = self.normalize(image)

        # samples = [self.normalize(patch) for patch in samples]
        random.shuffle(samples)

        return {'original': image, 'patches': samples, 'index': index}