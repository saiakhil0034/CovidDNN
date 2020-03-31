import torch.nn as nn
import pickle

from torchvision import transforms
from models.architectures import CovidDNN

baseDataPath = './data'

train_csv_path = "req_files/train_split_v2.txt"
test_csv_path = "req_files/test_split_v2.txt"


size = (224, 224)

transform = transforms.Compose([transforms.Resize(size), transforms.RandomHorizontalFlip(),
                                transforms.ToTensor()])


args = {
    'root' : baseDataPath,
    
    'train_csv_path': train_csv_path,
    
    'test_csv_path': test_csv_path,
    
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
