""" This is the main training script for the VIIRS vessel feedback CNN.
"""
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from nets import NightLightsNet
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision.utils import make_grid
from viirs_dataset import VIIRSVesselDataset

from utils import log_val_predictions

VAL_SIZE = 0.1
N_EPOCHS = 10
TRAIN_BATCH_SIZE = 24
VAL_BATCH_SIZE = 400
SGD_MOMENTUM = 0.8
LEARNING_RATE = 0.0001
NUM_BATCHES_TO_LOG = 10

MODEL = NightLightsNet()
optimizer = optim.SGD(MODEL.parameters(), lr=LEARNING_RATE, momentum=SGD_MOMENTUM)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

TRAIN_DATA_PATH = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "vvd_annotations"
)

train_dataset = VIIRSVesselDataset(root_dir=TRAIN_DATA_PATH)
MODEL_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "model.pt")

if "WANDB_API_KEY" not in os.environ:
    wandb.init(project="VIIRS-Feedback-Model", mode="disabled")
else:
    wandb.init(project="VIIRS-Feedback-Model")

train_indices, val_indices, _, _ = train_test_split(
    range(len(train_dataset)),
    train_dataset.targets,
    stratify=train_dataset.targets,
    test_size=VAL_SIZE,
    random_state=42,
)

# generate subset based on indices
train_split = Subset(train_dataset, train_indices)
val_split = Subset(train_dataset, val_indices)

# create batches
train_batches = DataLoader(train_split, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
val_batches = DataLoader(val_split, batch_size=VAL_BATCH_SIZE, shuffle=False)


criterion = nn.CrossEntropyLoss()
for epoch in range(N_EPOCHS):
    for i, (inputs, labels) in enumerate(train_batches, 0):
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = MODEL(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        wandb.log({"train_loss": loss})

        # ✨ W&B: Create a Table to store predictions for each test step
    columns = [
        "id",
        "image",
        "prediction",
        "ground_truth",
        "score_correct",
        "score_incorrect",
    ]
    test_table = wandb.Table(columns=columns)

    MODEL.eval()
    with torch.no_grad():
        log_counter = 0
        for i, vdata in enumerate(val_batches):
            vinputs, vlabels = vdata
            voutputs = MODEL(vinputs)
            vloss = criterion(voutputs, vlabels)
            wandb.log({"validation_loss": vloss})

            p = torch.nn.functional.softmax(voutputs.detach(), dim=1)

            # print(p)
            val_predictions = MODEL(vinputs)
            ground_truth_class_ids = vlabels

            top_pred_ids = val_predictions.argmax(axis=1)

            wandb.log(
                {
                    "conf_mat": wandb.plot.confusion_matrix(
                        probs=p.numpy(),
                        y_true=vlabels.numpy(),
                        class_names=["Incorrect", "Correct"],
                    )
                }
            )

            wandb.log(
                {
                    "roc": wandb.plot.roc_curve(
                        ground_truth_class_ids.numpy(),
                        val_predictions.detach().numpy(),
                        labels=["Incorrect", "Correct"],
                        classes_to_plot=None,
                    )
                }
            )

            if log_counter < NUM_BATCHES_TO_LOG:
                images = np.expand_dims(np.clip(vinputs[:, 0, :, :], 0, 100), axis=1)
                log_val_predictions(
                    images,
                    ground_truth_class_ids.numpy(),
                    voutputs,
                    top_pred_ids,
                    test_table,
                    log_counter,
                )
                log_counter += 1

            wandb.log({"val_predictions": test_table})

            images = wandb.Image(
                make_grid(
                    torch.from_numpy(
                        np.expand_dims(np.clip(vinputs[0:50, 0, :, :], 0, 100), axis=1)
                    ),
                    nrow=10,
                    padding=2,
                    pad_value=100,
                    normalize=False,
                )
            )
            wandb.log(
                {
                    "pr": wandb.plot.pr_curve(
                        ground_truth_class_ids.numpy(),
                        val_predictions.detach().numpy(),
                        labels=None,
                        classes_to_plot=None,
                    )
                }
            )

            wandb.log(
                {
                    "f1": f1_score(
                        ground_truth_class_ids.numpy(), top_pred_ids.detach().numpy()
                    )
                }
            )

            wandb.log({"validation_examples": images})

artifact = wandb.Artifact("model_weights", type="model")
artifact.add_file(MODEL_PATH)
wandb.log_artifact(artifact)

torch.save(MODEL.state_dict(), MODEL_PATH)
