from grace.models.classifier import GCN

import pytest


@pytest.mark.parametrize("input_dims", [1, 2, 4])
@pytest.mark.parametrize("embedding_dims", [16, 32, 64])
@pytest.mark.parametrize("output_dims", [1, 2, 4])
def test_model_building(input_dims, embedding_dims, output_dims):
    """Test building the model with different dimension."""

    model = GCN(
        input_dims=input_dims,
        embedding_dims=embedding_dims,
        output_dims=output_dims,
    )

    assert model.conv1.in_channels == input_dims
    assert model.linear.in_features == embedding_dims
    assert model.linear.out_features == output_dims
