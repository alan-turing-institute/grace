import mrcfile
import starfile


import os
from os import listdir
from os.path import isfile, join

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset


class MRCImageDataset(Dataset):
    """Creating a Torch dataset from a local folder of MRC,
    and corresponding star files.

    Parameters
    ----------
    transform = transformation added to the images
    target_transform = the same but for the targets
    filaments = are the objects in the images filaments?
    mrcpath = base path to the mrc files
    starpath = base path to star files
    type_star = which folder of starfiles, assuming different coordinates available.
    """

    def __init__(
        self,
        type_star=None,
        transform=None,
        target_transform=None,
        filaments=False,
        mrcpath=None,
        starpath=None,
    ):
        df = pd.DataFrame(
            columns=["mrcfolder", "mrcfile", "starfolder", "starfile"]
        )
        if mrcpath is None:
            mrcpath = "/bask/homes/t/ttef2338/vjgo8416-ms-img-pc/bea/TF1/Micrographs/"
            starpath = "/bask/homes/t/ttef2338/vjgo8416-ms-img-pc/bea/TF1/"
        if type_star is None:
            type_star = ""

        starpath = starpath + type_star + "/"

        onlymrc = [f for f in listdir(mrcpath) if isfile(join(mrcpath, f))]
        onlystar = [f for f in listdir(starpath) if isfile(join(starpath, f))]
        for i in range(0, len(onlymrc)):
            if onlymrc[i].split(".")[0] + ".star" in onlystar:
                df2 = pd.DataFrame(
                    [
                        [
                            mrcpath,
                            onlymrc[i],
                            starpath,
                            onlymrc[i].split(".")[0] + ".star",
                        ]
                    ],
                    columns=["mrcfolder", "mrcfile", "starfolder", "starfile"],
                )
                df = pd.concat([df, df2], axis=0, ignore_index=True)

        self.full_dirs = df
        self.transform = transform
        self.target_transform = target_transform
        self.filaments = filaments
        self.nodes_dict = pd.DataFrame(columns=["image_id", "x", "y", "box"])
        self.box_id = 0

    def __len__(self):
        return len(self.full_dirs)

    def __getitem__(self, idx):
        # Get image
        img_path = os.path.join(
            self.full_dirs.iloc[idx, 0], self.full_dirs.iloc[idx, 1]
        )
        # image = torch.tensor(np.repeat
        # (mrcfile.read(img_path)[np.newaxis, :, :], 3, axis=0))
        image = torch.tensor(mrcfile.read(img_path), dtype=torch.float32)
        # Get target, aka, bouding boxes from starfile filament description
        target_path = os.path.join(
            self.full_dirs.iloc[idx, 2], self.full_dirs.iloc[idx, 3]
        )
        star_coords = starfile.read(target_path)
        boxes = get_boxes(star_coords, self.filaments)

        self.nodes_dict = get_locations(
            self.nodes_dict, star_coords, image, self.full_dirs.iloc[idx, 1]
        )

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((len(boxes),), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        # target["image_id"] = [self.full_dirs.iloc[idx, 0]]
        target["image_id"] = torch.tensor([idx])

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            target = self.target_transform(target)

        return image, target

    def return_locations(self):
        """Returns a dictionary of the detections/nodes.

        Keys:
        image_id: name of the file where it originated
        x: x coordinate
        y: y coordinate
        box: cut image with box_buffer size around each (x,y) centre.
        """
        return self.nodes_dict


def get_locations(nodes_dict, star_coords, image, image_id):
    """Add the locations of the detections to the nodes."""
    box_buffer = 112
    for i in range(0, star_coords.shape[0]):
        x = star_coords.iloc[i].rlnCoordinateX
        y = star_coords.iloc[i].rlnCoordinateY
        temp_image = image[
            round(x - box_buffer) : round(x + box_buffer),
            round(y - box_buffer) : round(y + box_buffer),
        ]
        df2 = pd.DataFrame(
            [[image_id, x, y, temp_image]],
            columns=["image_id", "x", "y", "box"],
        )
        nodes_dict = pd.concat([nodes_dict, df2], axis=0, ignore_index=True)

    return nodes_dict


def get_boxes(star_coords, filaments, box_buffer=None):
    """Create bounding boxes from a centre coordinate."""
    i = 1
    if filaments:
        filaments = []
        while i < star_coords.shape[0]:
            start_x = star_coords.iloc[i - 1].rlnCoordinateX
            start_y = star_coords.iloc[i - 1].rlnCoordinateY
            end_x = star_coords.iloc[i].rlnCoordinateX
            end_y = star_coords.iloc[i].rlnCoordinateY
            filaments.append([start_x, start_y, end_x, end_y])
            i = i + 2

        boxes = []
        if box_buffer is None:
            box_buffer = 32
        for filament in filaments:
            # n_points = int(np.sqrt((filament[2] - filament[0]) **
            # 2 + (filament[3] - filament[1]) ** 2) / 0.02)
            n_points = int(
                np.sqrt(
                    abs(filament[2] - filament[0])
                    + abs(filament[3] - filament[1])
                )
                / 2
            )
            # n_points = 20
            dx = np.linspace(filament[0], filament[2], n_points)
            dy = np.linspace(filament[1], filament[3], n_points)

            for i in range(0, len(dx)):
                temp_box = [
                    dx[i] - box_buffer,
                    dy[i] - box_buffer,
                    dx[i] + box_buffer,
                    dy[i] + box_buffer,
                ]
                boxes.append(temp_box)
    else:
        boxes = []
        if box_buffer is None:
            box_buffer = 112
        for i in range(0, star_coords.shape[0]):
            x = star_coords.iloc[i].rlnCoordinateX
            y = star_coords.iloc[i].rlnCoordinateY
            temp_box = [
                x - box_buffer,
                y - box_buffer,
                x + box_buffer,
                y + box_buffer,
            ]
            boxes.append(temp_box)

    return boxes
