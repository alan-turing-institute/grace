from typing import List

import torch
import torch_geometric
from torch_geometric.loader import DataLoader

from torch.utils.tensorboard import SummaryWriter

def train_model(
    model: torch.nn.Module,
    dataset: List[torch_geometric.data.Data],
    *,
    epochs: int = 100,
    batch_size: int = 64,
):
    """Train the pytorch model."""
    writer = SummaryWriter()

    train_dataset = dataset[: round(0.7 * len(dataset))]
    test_dataset = dataset[round(0.7 * len(dataset)) :]

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.01,
        # weight_decay=5e-4),
    )

    criterion = torch.nn.CrossEntropyLoss()

    def train():
        model.train()

        for data in train_loader:
            out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    def test(loader):
        model.eval()

        correct = 0
        for data in loader:
            out = model(data.x, data.edge_index, data.batch)
            pred = out.argmax(dim=1)  # Use the class with highest probability.
            correct += int((pred == data.y).sum())
        return correct / len(loader.dataset)

    for epoch in range(1, epochs):
        train()
        train_acc = test(train_loader)
        test_acc = test(test_loader)
        print(
            f"Epoch: {epoch:03d}, Train Acc: {train_acc:.4f},"
            f" Test Acc: {test_acc:.4f}"
        )
