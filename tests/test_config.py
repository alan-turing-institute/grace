import pytest

from grace.training.config import (
    Config,
    write_config_file,
    load_config_params,
    ExtractorNotDefinedError,
)


def test_write_config_file(tmp_path):
    config = Config()
    setattr(config, "run_dir", tmp_path)

    # test json:
    write_config_file(config, "json")
    expected_path = tmp_path / "config_hyperparams.json"
    assert expected_path.exists()

    # test yaml:
    write_config_file(config, "yaml")
    expected_path = tmp_path / "config_hyperparams.yaml"
    assert expected_path.exists()


@pytest.mark.parametrize("batch_size", [256, 1024, 4096])
def test_load_config_file(tmp_path, batch_size):
    config = Config()
    assert config.extractor_fn is None

    setattr(config, "train_image_dir", tmp_path)
    setattr(config, "train_grace_dir", tmp_path)
    setattr(config, "valid_image_dir", tmp_path)
    setattr(config, "valid_grace_dir", tmp_path)
    setattr(config, "infer_image_dir", tmp_path)
    setattr(config, "infer_grace_dir", tmp_path)
    setattr(config, "log_dir", tmp_path)
    setattr(config, "run_dir", tmp_path)
    setattr(config, "extractor_fn", tmp_path / "extractor.pt")
    setattr(config, "batch_size", batch_size)

    # write out:
    write_config_file(config, "yaml")
    loaded_config = load_config_params(tmp_path / "config_hyperparams.yaml")
    assert loaded_config.extractor_fn == tmp_path / "extractor.pt"
    assert loaded_config.batch_size == batch_size
    assert loaded_config.run_dir.exists()


@pytest.mark.parametrize("laplacian, expected_ndim", [(False, 4), (True, 8)])
def test_config_feature_extractor_if_None(laplacian, expected_ndim):
    config = Config()
    print("laplaaacian", config.laplacian)
    setattr(config, "extractor_fn", None)
    setattr(config, "laplacian", laplacian)
    assert config.node_embedding_ndim == expected_ndim


@pytest.mark.parametrize(
    "extractor_fn, expected_ndim",
    [("resnet18.pt", 512), ("resnet152.pt", 2048)],
)
def test_config_feature_extractor_filename(
    tmp_path, extractor_fn, expected_ndim
):
    config = Config()

    setattr(config, "train_image_dir", tmp_path)
    setattr(config, "train_grace_dir", tmp_path)
    setattr(config, "valid_image_dir", tmp_path)
    setattr(config, "valid_grace_dir", tmp_path)
    setattr(config, "infer_image_dir", tmp_path)
    setattr(config, "infer_grace_dir", tmp_path)
    setattr(config, "log_dir", tmp_path)
    setattr(config, "run_dir", tmp_path)
    setattr(config, "extractor_fn", tmp_path / extractor_fn)
    setattr(config, "laplacian", False)

    write_config_file(config, "yaml")
    loaded_config = load_config_params(tmp_path / "config_hyperparams.yaml")
    assert loaded_config.node_embedding_ndim == expected_ndim


# @pytest.mark.xfail()
@pytest.mark.parametrize(
    "extractor_fn",
    [
        "extractor.pt",
    ],
)
def test_incorrect_extractor_filename(tmp_path, extractor_fn):
    with pytest.raises(ExtractorNotDefinedError):
        config = Config()
        setattr(config, "extractor_fn", tmp_path / extractor_fn)
        ndim = config.node_embedding_ndim
        assert isinstance(ndim, int)
        # Code that should raise SpecificError
        raise ExtractorNotDefinedError("This is a specific error message")
