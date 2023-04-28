from grace.models.classifier import GCN

import pytest


@pytest.mark.parametrize("input_channels", [1, 2, 4])
@pytest.mark.parametrize("hidden_channels", [16, 32, 64])
@pytest.mark.parametrize("output_classes", [1, 2, 4])
def test_model_building(input_channels, hidden_channels, output_classes):
    """Test building the model with different dimension."""

    model = GCN(
        input_channels=input_channels,
        hidden_channels=hidden_channels,
        output_classes=output_classes,
    )

    assert model.conv1.in_channels == input_channels
    assert model.linear.in_features == hidden_channels
    assert model.linear.out_features == output_classes
