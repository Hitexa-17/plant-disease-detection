from PIL import Image
from torch.utils.data import Dataset


class PlantDataset(Dataset):

    def __init__(self, dataframe, class_to_idx, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]

        image = Image.open(row["filepath"]).convert("RGB")
        label = self.class_to_idx[row["class"]]

        if self.transform:
            image = self.transform(image)

        return image, label