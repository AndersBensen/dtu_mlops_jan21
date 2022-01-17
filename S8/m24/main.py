import torchvision
import torch
import pytorch_lightning as pl

from model import Classifier
from dataloader import OurDataModule

if __name__ == '__main__':
    datamodule = OurDataModule()

    feature_extractor = torchvision.models.resnet18(pretrained=True)
    feature_extractor.fc = torch.nn.Identity()
    model = Classifier(feature_extractor)
    trainer = pl.Trainer()
    trainer.fit(model, OurDataModule())
