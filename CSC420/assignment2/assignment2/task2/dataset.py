

from torch.utils.data import Dataset
import torchvision.transforms as transforms

class DogBreedDataset(Dataset):
    def __init__(self, ds, model_number):
        self.ds = ds
        if model_number ==1:
            size = (224,224)
        if model_number ==2:
            size = (256,256)
        self.transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
        #    transforms.Normalize(*imagenet_stats, inplace=True)
        ])
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        img, label = self.ds[idx]
        if self.transform:
            img = self.transform(img)  
            return img, label

class TrainDogBreedDataset(DogBreedDataset):
    def __init__(self, ds, model_number):
        super().__init__(ds, model_number)

        if model_number ==1:
            resize = (256,256)
            crop=224
        if model_number ==2:
            resize = (300, 300)
            crop = 256

        self.transform = transforms.Compose([
            transforms.Resize(resize),
            transforms.RandomCrop(crop, padding=4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomRotation(degrees=30),
            transforms.ToTensor(),
            
        ])    
    def __getitem__(self, idx):
        return super().__getitem__(idx)

