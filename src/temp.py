import os
import shutil
from pathlib import Path

import pandas as pd

for file in Path(".").rglob("detections.csv"):
    if len(pd.read_csv(file)):
        new_filename = Path(file.resolve().parent) / (
            str(file.resolve().parent).rsplit("/")[-1] + "_" + file.stem + ".csv"
        )
        # npy_file = list(Path(file[0].resolve().parent).glob("*.npy"))[0]
        print(new_filename)
        file.rename(new_filename)
