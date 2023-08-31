import torch
from torchvision import models
import pytorch_lightning as pl
from torchmetrics import F1Score


class SealClassificationModel(pl.LightningModule):
    def __init__(self, learning_rate=3e-4):
        super().__init__()
        self.learning_rate = learning_rate
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.neural_net = models.resnet50(pretrained=True)
        self.neural_net.fc = torch.nn.Linear(2048, 2)

        self.f1_train = F1Score(task="binary").to("cuda")
        self.f1_eval = F1Score(task="binary").to("cuda")
        self.f1_test = F1Score(task="binary").to("cuda")

    def forward(self, x):
        return self.neural_net(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=2, verbose=False
        )
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_prediction = self(x)
        loss = self.loss_func(y_prediction, y)
        f1score = self.f1_train(torch.argmax(y_prediction, dim=1), y)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_F1", f1score, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_prediction = self(x)
        loss = self.loss_func(y_prediction, y)
        f1score = self.f1_eval(torch.argmax(y_prediction, dim=1), y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_F1", f1score, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_prediction = self(x)
        loss = self.loss_func(y_prediction, y)
        f1score = self.f1_test(torch.argmax(y_prediction, dim=1), y)
        self.log("test_loss: ", loss, prog_bar=True)
        self.log("test_F1: ", f1score, prog_bar=True)
        return loss
