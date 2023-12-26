import PIL
import os
import torch
import numpy as np

class ImageTrainDataSet(torch.utils.data.Dataset):
    def __init__(self,  folder_with_images, transform=None, aug=True, color_transform=None):

        self.folder_with_images = folder_with_images
        self.list_of_names = os.listdir(self.folder_with_images)
        self.transform = transform
        self.color_transform = color_transform
        self.aug = aug

    def __len__(self):
        return len(self.list_of_names)

    def __getitem__(self, i):

        name = self.list_of_names[i]

        xy = PIL.Image.open(os.path.join(self.folder_with_images, name))

        if self.color_transform is not None:
            xy = self.color_transform(xy)

        width, height = xy.size

        x = xy.crop((width // 2, 0, width, height))
        y = xy.crop((0, 0, width // 2, height))
        if self.aug and np.random.rand() <= 0.5:
            x = x.transpose(PIL.Image.FLIP_LEFT_RIGHT)
            y = y.transpose(PIL.Image.FLIP_LEFT_RIGHT)

        if self.transform is not None:
            x = self.transform(x)
            y = self.transform(y)
        return x,y
