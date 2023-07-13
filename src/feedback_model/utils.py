""" Utility functions for the feedback model """
import numpy as np
import torch.nn.functional as F
from torch import Tensor

import wandb

NUM_IMAGES_PER_BATCH = 50


def log_val_predictions(
    log_images: np.ndarray,
    log_labels: np.ndarray,
    outputs: Tensor,
    log_preds: np.ndarray,
    test_table: wandb.Table,
    log_counter: int,
) -> None:
    """Populate the a table with images and their gt/predictions

    Parameters
    ----------
    log_images : np.ndarray
    log_labels : np.ndarray
    outputs : Tensor
    log_preds : np.ndarray
    test_table : wandb.Table
    log_counter : int
    """
    # obtain confidence scores for all classes
    scores = F.softmax(outputs.data, dim=1)
    log_scores = scores
    # adding ids based on the order of the images
    _id = 0
    for img, label, prediction, scores in zip(
        log_images, log_labels, log_preds, log_scores
    ):
        img_id = str(_id) + "_" + str(log_counter)
        test_table.add_data(img_id, wandb.Image(img), prediction, label, *scores)
        _id += 1
        if _id == NUM_IMAGES_PER_BATCH:
            break
