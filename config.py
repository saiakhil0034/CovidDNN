import torch.nn as nn
import pickle

from torchvision import transforms
from models import covidDNN

baseDataPath = './data'
data_path = f"{baseDataPath}/file_names_jules.csv"

size = (224, 224)

transform = transforms.Compose([transforms.Resize(size), transforms.RandomHorizontalFlip(),
                                transforms.ToTensor()])


args = {
    'file_path_csv': data_path,

    'epochs': 500,

    'batch_size': 2,

    'num_workers': 4,

    'nc': 3,

    'ngpu': 1,

    'model_path': './results/covidDNN_1',

    'model': CovidDNN,

    'loss_criterion': nn.CrossEntropyLoss(),

    'learning_rate': 2e-4,

    'beta': 0.5,

    'transforms': transform,
}
