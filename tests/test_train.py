import torch
import pytest

from grace.base import Annotation
from grace.models.train import edge_criterion
from grace.models.classifier import GCN

def test_edge_criterion_masks_unknown_ground_truth():

    masked_class = Annotation.UNKNOWN

    embed = torch.tensor([
        [-1.,-1.],
        [1.,1.],
    ], dtype=torch.float32)

    target = torch.tensor([
        Annotation.TRUE_NEGATIVE,
        Annotation.TRUE_POSITIVE,
        Annotation.TRUE_NEGATIVE,
        Annotation.UNKNOWN,
    ], dtype=torch.float32)

    edge_index = torch.tensor([
        [0,0,0,0],
        [1,1,1,1],
    ], dtype=torch.int32)

    assert (
        edge_criterion(embed, target, edge_index, masked_class)
        == edge_criterion(embed, target[:-1], edge_index[:,:-1], masked_class)
    )
    assert (
        edge_criterion(embed, target, edge_index, masked_class)
        != edge_criterion(embed, target[1:], edge_index[:,1:], masked_class)
    )
