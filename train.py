import torch
from torch_datamodule import DataModule
import pytorch_lightning as pl
from model import Model
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

torch.set_default_dtype(torch.double)

model_parameters = {
    "hidden_size": 5,
    "dropout": 0.2,
    "num_layers": 1,
    "bidirectional": True,
    "lr": 1e-3,
}

training_parameters = {"max_epochs": 1200, "accumulate_grad_batches": 16}

name = (
    f"l1_loss,"
    f"AdamW,"
    f"weight_decay=0.2,"
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

logger = TensorBoardLogger("./logs", name=name)
model = Model(**model_parameters)
checkpoint_callback = ModelCheckpoint(
    monitor="loss/val",
    dirpath="checkpoints/",
    filename=f"{name}",
    save_top_k=1,
    mode="min",
)


trainer = pl.Trainer(
    logger=logger,
    **training_parameters,
    callbacks=[checkpoint_callback],
    progress_bar_refresh_rate=10000,
    gpus=1
)
trainer.fit(model, data_module)
