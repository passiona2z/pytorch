
# import some packages you need here
import tarfile
import glob 
import os

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from skimage import io


class MNIST(Dataset):
    """ MNIST dataset

        To write custom datasets, refer to
        https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

    Args:
        data_dir: directory path containing images

    Note:
        1) Each image should be preprocessed as follows:
            - First, all values should be in a range of [0,1]
            - Substract mean of 0.1307, and divide by std 0.3081
            - These preprocessing can be implemented using torchvision.transforms
        2) Labels can be obtained from filenames: {number}_{label}.png
    """

    def __init__(self, data_dir):

        # unzip tar file
        if not os.path.exists(f'{data_dir}') :
            tar = tarfile.open(f"{data_dir}.tar")
            tar.extractall('../data')
            tar.close()

        # get file
        file = glob.glob(f'{data_dir}/*')
     

        image_li = []
 
        for image_root in file :
            image = io.imread(image_root)
            image_li.append(torch.Tensor(image))


        # tensor image
        self.X = torch.stack(image_li)
  

        # preprocessing
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.Normalize(mean=0.1307, std=0.3081)
        ])

        # transform = transforms.Normalize(mean=0.1307, std=0.3081)
        self.X = transform(self.X/255)
     
        # print(f'After preprocessing / traing image mean : {(self.train).mean():.4f}, std {(self.train).std():.4f}')

        # label
        self.y = torch.LongTensor([int(i.split('_')[1][0]) for i in file])


    def __len__(self):

        return len(self.X)
    

    def __getitem__(self, idx):

        img   = self.X[idx]
        label = self.y[idx]

        return img, label
    

if __name__ == '__main__':

    # write test codes to verify your implementations
    dataset = MNIST("../data/train")
    print(f"self.X {dataset.X.shape}")
    print(f"After preprocessing / traing image mean : {dataset.X.mean():.3f}, std {dataset.X.std():.3f}")
    print(f"self.y {dataset.y.shape}")
  