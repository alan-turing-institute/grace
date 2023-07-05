import pytest
import numpy as np

from grace.simulator.simulate_graph import random_graph
from grace.simulator.simulate_image import synthesize_image_from_graph

from conftest import simple_graph

@pytest.mark.parametrize("n_motifs", [3])
@pytest.mark.parametrize("n_chaff", [100])
@pytest.mark.parametrize("density", [0.02])
@pytest.mark.parametrize("motif", ["lines"])

class TestGraph:
    @pytest.fixture
    def graph(
        self,
        n_motifs, 
        n_chaff, 
        density, 
        motif,
    ):
        return random_graph(
            n_motifs=n_motifs, 
            n_chaff=n_chaff, 
            density=density, 
            motif=motif
        )
    def test_num_nodes_greater_than_n_chaff(
        self,
        graph,
        n_chaff
    ):
        assert graph.number_of_nodes() > n_chaff

    

# Arange , act - do, assert - check expectations == observed


@pytest.mark.parametrize("n_motifs", [3])
@pytest.mark.parametrize("n_chaff", [100])
@pytest.mark.parametrize("density", [0.02])
@pytest.mark.parametrize("motif", ["lines"])

@pytest.mark.parametrize("image_value", [0, 0.5, 1, 227])
@pytest.mark.parametrize("image_shape", [(3500, 3500)])
@pytest.mark.parametrize("crop_shape", [(224, 224)])
@pytest.mark.parametrize("mask_shape", [(112, 112)])

class TestImage:
    @pytest.fixture  # first create a graph
    def graph(
        self,
        n_motifs, 
        n_chaff, 
        density, 
        motif,
    ):
        return random_graph(
            n_motifs=n_motifs, 
            n_chaff=n_chaff, 
            density=density, 
            motif=motif
        )
    
    @pytest.fixture  # then create a corresponding image
    def image(
        self,
        graph,
        image_value,
        image_shape,
        crop_shape,
        mask_shape,
    ):
        return synthesize_image_from_graph(
            graph,
            image_value,
            image_shape,
            crop_shape,
            mask_shape,
        )[0]
       
    def test_three_unique_pixel_values(
        self,
        # graph,
        image,
    ):
        print (image.shape)
        print (np.unique(image))
        assert len(np.unique(image)) == 3
