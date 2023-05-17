import pytest

from grace.config import Config, write_config_file, load_config_params

def test_write_config_file(tmp_path):

    config = Config()
    setattr(config, "run_dir", tmp_path)

    write_config_file(config, "yaml")
    expected_path = tmp_path / "config_hyperparams.yaml"

    assert expected_path.exists()

def test_load_config_file(tmp_path):

    config = Config()
    setattr(config, "feature_dim", 251)
    setattr(config, "run_dir", tmp_path)
    write_config_file(config)

    loaded_config = load_config_params(
        tmp_path / "config_hyperparams.json"
    )

    assert loaded_config.feature_dim == 251

