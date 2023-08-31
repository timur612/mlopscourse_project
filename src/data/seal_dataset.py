from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path


class SealDataset(Dataset):
    def __init__(self, images_direction, transform=None):
        self.paths = list(Path(images_direction).glob("*/*.jpg"))
        self.transform = transform

        self.classes, self.class_to_idx = (
            ["no_seal", "seal"],
            {"no_seal": 0, "seal": 1},
        )

    def load_image(self, index: int) -> Image.Image:
        """load image from path
        return:
           image
        """

        image_path = self.paths[index]
        return Image.open(image_path)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = self.load_image(idx)
        class_name = self.paths[idx].parent.name
        class_idx = self.class_to_idx[class_name]

        if self.transform:
            return self.transform(img), class_idx
        else:
            return img, class_idx
