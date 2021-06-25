""" 추론 코드
"""

import os
import random
import torch
from torch.utils.data import DataLoader
import numpy as np
from model.model import Summarizer
from modules.dataset import CustomDataset
from modules.trainer import Trainer
from modules.metrics import Hitrate

if __name__ == '__main__':


    # Set random seed
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(42)
    random.seed(42)
    DATA_DIR = './data'
    TRAINED_MODEL_PATH = ...
    BATCH_SIZE = 4
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load dataset & dataloader
    test_dataset = CustomDataset(data_dir=DATA_DIR, mode='test')
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Load Model
    model = Summarizer().to(device)
    model.load_state_dict(torch.load(TRAINED_MODEL_PATH)['model'])

    # Set metrics & Loss function
    metric_fn = Hitrate
    loss_fn = torch.nn.BCELoss(reduction='none')

    # Set trainer
    trainer = Trainer(model, device, loss_fn, metric_fn)

    # Predict
    trainer.validate_epoch(test_dataloader, epoch_index=0)


