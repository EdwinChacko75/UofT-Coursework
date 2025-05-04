

from torch.utils.data import Dataset
import torchvision.transforms as transforms

class DogBreedDataset(Dataset):
    def __init__(self, ds):
        self.ds = ds
        self.transform = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor(),
        ])
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        img, label = self.ds[idx]
        if self.transform:
            img = self.transform(img)  
            return img, label

class TrainDogBreedDataset(DogBreedDataset):
    def __init__(self, ds):
        super().__init__(ds)

        self.transform = transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.RandomCrop(256, padding=4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomRotation(degrees=30),
            transforms.ToTensor(),
            
        ])    
    def __getitem__(self, idx):
        return super().__getitem__(idx)

