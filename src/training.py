import random

import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader


from .models import Recommender
from .data_processing import map_column, MASK, Dataset


def train(
    data_path: str,
    log_dir: str = "training_logs",
    model_dir: str = "models",
    batch_size: int = 32,
    epochs: int = 2000,
    history_size: int = 120,
):
    delimiter = "::"
    column_names = ["userId","movieId","rating","timestamp"]
    data = pd.read_csv(data_path, 
                       delimiter=delimiter, 
                       header=None, 
                       names=column_names)

    data.sort_values(by="timestamp", inplace=True)

    data, mapping, inverse_mapping = map_column(data, col_name="movieId")

    grp_by_train = data.groupby(by="userId")

    groups = list(grp_by_train.groups)

    train_data = Dataset(
        groups=groups,
        grp_by=grp_by_train,
        split="train",
        history_size=history_size,
    )
    val_data = Dataset(
        groups=groups,
        grp_by=grp_by_train,
        split="val",
        history_size=history_size,
    )

    print("len(train_data)", len(train_data))
    print("len(val_data)", len(val_data))

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        num_workers=10,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        num_workers=10,
        shuffle=False,
    )

    model = Recommender(
        vocab_size=len(mapping) + 2,
        lr=1e-4,
        dropout=0.3,
    )

    logger = TensorBoardLogger(
        save_dir=log_dir,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="valid_loss",
        mode="min",
        dirpath=model_dir,
        filename="recommender",
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator='cpu',
        logger=logger,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model, train_loader, val_loader)

    result_val = trainer.test(dataloaders=val_loader)

    output_json = {
        "val_loss": result_val[0]["test_loss"],
        "best_model_path": checkpoint_callback.best_model_path,
    }

    print(output_json)

    return output_json


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path")
    parser.add_argument("--epochs", type=int, default=500)
    args = parser.parse_args()

    train(
        data_path=args.data_path,
        epochs=args.epochs,
    )
