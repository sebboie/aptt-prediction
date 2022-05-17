import pandas as pd
import torch
from torch_datamodule import DataModule
import pytorch_lightning as pl
from model import Model
import random
random.seed(1337)
import numpy as np

torch.set_default_dtype(torch.double)

model_parameters = { "hidden_size": 5, "dropout": 0.1,
    "num_layers": 1,
    "bidirectional": True,
    "lr": 1e-3
}

training_parameters = {"max_epochs": 1200, "accumulate_grad_batches": 16,
                       "auto_lr_find": True}

name = (
    f"l1_loss,"
    f"AdamW,"
    f"weight_decay=0.1,"
    f"hidden_size={model_parameters['hidden_size']},"
    f"dropout={model_parameters['dropout']},"
    f"num_layers={model_parameters['num_layers']},"
    f"bidrectional={model_parameters['bidirectional']},"
    f"learning_rate={model_parameters['lr']},"
    f"max_epochs={training_parameters['max_epochs']},"
    f"accumulate_grad_batches={training_parameters['accumulate_grad_batches']},"
    f"modified_model"
)

data_module = DataModule()
data_module.prepare_data()
data_module.setup()

model = Model(**model_parameters)
model = Model.load_from_checkpoint(
    f"checkpoints/{name}.ckpt", **model_parameters
)
model.eval()

results = {"y": [], "y_hat": []}
for i, datum in enumerate(data_module.test_dataloader()):
    y = datum[1][0][0].detach().numpy()
    y_hat = model.forward(datum[0]).detach().numpy()
    results["y"].append(y)
    results["y_hat"].append(y_hat[0][0])
