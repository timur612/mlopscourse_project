from pytorch_lightning import Trainer
from torchvision import transforms
from torch.utils.data import DataLoader

train_transforms = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
    ]
)

test_transforms = transforms.Compose(
    [transforms.Resize((64, 64)), transforms.ToTensor()]
)

train_data = SealDataset("./interim/train", transform=train_transforms)
test_data = SealDataset("./interim/test", transform=test_transforms)
eval_data = SealDataset("./interim/eval", transform=test_transforms)

model = SealClassificationModel()
# data = SealDatasetPL()
trainer = Trainer(max_epochs=5)

train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
eval_dataloader = DataLoader(eval_data, batch_size=64, shuffle=False)

trainer.fit(model, train_dataloader, eval_dataloader)
