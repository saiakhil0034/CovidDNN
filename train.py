#!/usr/bin/env python
# coding: utf-8


import nntools as nt
from models import architectures
import torch
import torch.nn as nn
from config import args
import os
from dataloader import get_loader


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

model = architectures.CovidDNN().to(device)


model.apply(weights_init)

criterion = args['loss_criterion']

params = list(model.parameters())

optimizer = torch.optim.SGD(params, lr=args['learning_rate'])#, betas=(args['beta'], 0.999))

stats_manager= nt.StatsManager()


exp1 = nt.Experiment(model, device, criterion, optimizer, stats_manager,
                     output_dir=args['model_path'])


if __name__ == "__main__":
	exp1.run(num_epochs=args['epochs'])
