from pathlib import Path
from random import randint

import PIL

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.datasets import ImageFolder, FakeData
from pytorch_lightning import LightningDataModule
import torch
from typing import Any, Tuple

import webdataset as wds

from PIL import Image
from io import BytesIO

#To prevent truncated error
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def web_dataset_helper(path):
    if Path(path).is_dir():
        DATASET = [str(p) for p in Path(path).glob("**/*") if ".tar" in str(p).lower()] # .name
        assert len(DATASET) > 0, 'The directory ({}) does not contain any WebDataset/.tar files.'.format(path)
        print('Found {} WebDataset .tar(.gz) file(s) under given path {}!'.format(len(DATASET), path))
    elif ('http://' in path.lower()) | ('https://' in path.lower()):
        DATASET = f"pipe:curl -L -s {path} || true"
        print('Found {} http(s) link under given path!'.format(len(DATASET), path))
    elif 'gs://' in path.lower():
        DATASET = f"pipe:gsutil cat {path} || true"
        print('Found {} GCS link under given path!'.format(len(DATASET), path))
    elif '.tar' in path:
        DATASET = path
        print('Found WebDataset .tar(.gz) file under given path {}!'.format(path))
    else:
        raise Exception('No folder, no .tar(.gz) and no url pointing to tar files provided under {}.'.format(path))
    return DATASET

def identity(x):
    return x

class Grayscale2RGB:
    def __init__(self):  
        pass  
    def __call__(self, img):
        if img.mode != 'RGB':
            return img.convert('RGB') 
        else:
            return img
    def __repr__(self):
        return self.__class__.__name__ + '()'        




class ImageDataModule(LightningDataModule):

    def __init__(self, train_dir, val_dir, batch_size, num_workers, img_size, resize_ratio=0.75, 
                fake_data=False, web_dataset=False, world_size = 1, dataset_size = [int(1e9)]):
        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.fake_data = fake_data
        self.img_size = img_size
        self.web_dataset = web_dataset
        if len(dataset_size) == 1:
            self.train_dataset_size = dataset_size[0]  
            self.val_dataset_size = dataset_size[0]
        else:
            self.train_dataset_size = dataset_size[0]  
            self.val_dataset_size = dataset_size[1] 
        self.world_size = world_size
        self.transform_train = T.Compose([
                            Grayscale2RGB(),
                            T.RandomResizedCrop(img_size,
                                    scale=(resize_ratio, 1.),ratio=(1., 1.)),
                            T.ToTensor(),
                            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),                            
                            ])
        self.transform_val = T.Compose([
                                    Grayscale2RGB(),
                                    T.Resize(img_size),
                                    T.CenterCrop(img_size),
                                    T.ToTensor(),
                                    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),                                
                                    ])
    def imagetransform(self, b):
        return Image.open(BytesIO(b))

    def dummy(self, s):
        return torch.zeros(1)

    def setup(self, stage=None):
        if self.fake_data:
            self.train_dataset = FakeData(12000000, (3, self.img_size, self.img_size), 1000, self.transform_train)
            self.val_dataset = FakeData(50000, (3, self.img_size, self.img_size), 1000, self.transform_val)
            self.transform_train = None
            self.transform_val = None            
        else:
            if self.web_dataset:
                DATASET_TRAIN = web_dataset_helper(self.train_dir)
                DATASET_VAL = web_dataset_helper(self.val_dir)
                
                
                self.train_dataset = (
                    wds.WebDataset(DATASET_TRAIN, handler=wds.warn_and_continue)   
                    .shuffle(1000, handler=wds.warn_and_continue) 
                    .decode("pil", handler=wds.warn_and_continue)   
                    .to_tuple("jpg;png;jpeg", handler=wds.warn_and_continue)   
                    .map_tuple(self.transform_train, handler=wds.warn_and_continue)   
                    .batched(self.batch_size, partial=False) # It is good to avoid partial batches when using Distributed training
                    )  

                self.val_dataset = (
                    wds.WebDataset(DATASET_VAL, handler=wds.warn_and_continue)
                    .decode("pil", handler=wds.warn_and_continue)   
                    .to_tuple("jpg;png;jpeg", handler=wds.warn_and_continue)   
                    .map_tuple(self.transform_val, handler=wds.warn_and_continue)   
                    .batched(self.batch_size, partial=False) # It is good to avoid partial batches when using Distributed training
                    )  

            else:
                self.train_dataset = ImageDataset(self.train_dir, self.transform_train)
                self.val_dataset = ImageDataset(self.val_dir, self.transform_val)
  

    def train_dataloader(self):
        if self.web_dataset:
            dl = wds.WebLoader(self.train_dataset, batch_size=None, num_workers=self.num_workers)
            number_of_batches = self.train_dataset_size // (self.batch_size * self.world_size)
            dl = dl.repeat(9999999999).slice(number_of_batches)
            dl.length = number_of_batches
            return dl
        else:
            return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        if self.web_dataset:
            dl = wds.WebLoader(self.val_dataset, batch_size=None, num_workers=self.num_workers)
            number_of_batches = self.val_dataset_size // (self.batch_size * self.world_size)
            dl = dl.repeat(9999999999).slice(number_of_batches)
            dl.length = number_of_batches
            return dl
        else:
            return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        #simply reuse val_dataloader for test.
        if self.web_dataset:
            dl = wds.WebLoader(self.val_dataset, batch_size=None, num_workers=self.num_workers)
            number_of_batches = self.val_dataset_size // (self.batch_size * self.world_size)
            dl = dl.repeat(9999999999).slice(number_of_batches)
            dl.length = number_of_batches
            return dl
        else:
            return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

class ImageDataset(ImageFolder):
    def random_sample(self):
        return self.__getitem__(randint(0, self.__len__() - 1))

    def sequential_sample(self, ind):
        if ind >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(ind + 1)

    def skip_sample(self, ind):
        return self.random_sample()


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        try:
            path, target = self.samples[index]
            sample = self.loader(path)
        except (PIL.UnidentifiedImageError, OSError) as corrupt_image_exceptions:
            print(corrupt_image_exceptions)
            print(f"An exception occurred trying to load file {path}.")
            print(f"Skipping index {index}")
            return self.skip_sample(index)     
                   
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


class ImageDataset2(Dataset):
    def __init__(self,
                 folder,
                 transform=None,
                 shuffle=False,
                 ):
        """
        @param folder: Folder containing images and text files matched by their paths' respective "stem"
        @param truncate_captions: Rather than throw an exception, captions which are too long will be truncated.
        """
        super().__init__()
        self.shuffle = shuffle
        path = Path(folder)

        self.image_files = [
            *path.glob('**/*.png'), *path.glob('**/*.jpg'),
            *path.glob('**/*.jpeg'), *path.glob('**/*.bmp')
        ]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def random_sample(self):
        return self.__getitem__(randint(0, self.__len__() - 1))

    def sequential_sample(self, ind):
        if ind >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(ind + 1)

    def skip_sample(self, ind):
        if self.shuffle:
            return self.random_sample()
        return self.sequential_sample(ind=ind)
        
    def __getitem__(self, ind):
        try:
            image_tensor = self.transform(PIL.Image.open(self.image_files[ind]))
        except (PIL.UnidentifiedImageError, OSError) as corrupt_image_exceptions:
            print(corrupt_image_exceptions)            
            print(f"An exception occurred trying to load file {self.image_files[ind]}.")
            print(f"Skipping index {ind}")
            return self.skip_sample(ind)      
        return image_tensor  