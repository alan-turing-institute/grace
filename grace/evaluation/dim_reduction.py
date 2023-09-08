import torch


def drop_linear_layers_from_model(
    model: torch.nn.Module,
) -> torch.nn.Sequential:
    """Chops off last 2 Linear layers from the classifier to
    access node embeddings learnt by the GCN classifier."""

    modules = list(model.children())[:-2]
    node_emb_extractor = torch.nn.Sequential(*modules)
    for p in node_emb_extractor.parameters():
        p.requires_grad = False

    return node_emb_extractor
