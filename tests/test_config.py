import pytest

from grace.training.config import Config, write_config_file, load_config_params


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


@pytest.mark.parametrize("feature_dim", [256, 1024, 4096])
def test_load_config_file(tmp_path, feature_dim):
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
    setattr(config, "feature_dim", feature_dim)

    # write out:
    write_config_file(config, "yaml")
    loaded_config = load_config_params(tmp_path / "config_hyperparams.yaml")
    assert loaded_config.extractor_fn == tmp_path / "extractor.pt"
    assert loaded_config.feature_dim == feature_dim
    assert loaded_config.run_dir.exists()
