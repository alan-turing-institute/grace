import pytest
import torch

from grace.run import run_grace
from grace.training.config import Config, write_config_file


@pytest.mark.xfail(
    reason="sample graph attribute dimensionality doesn't agree"
)
def test_run_grace(mrc_image_and_annotations_dir, simple_extractor):
    tmp_data_dir = mrc_image_and_annotations_dir

    # temp extractor
    extractor_fn = tmp_data_dir / "resnet18.pt"
    torch.save(simple_extractor, extractor_fn)

    config = Config(
        train_image_dir=tmp_data_dir,
        train_grace_dir=tmp_data_dir,
        valid_image_dir=tmp_data_dir,
        valid_grace_dir=tmp_data_dir,
        infer_image_dir=tmp_data_dir,
        infer_grace_dir=tmp_data_dir,
        log_dir=tmp_data_dir,
        run_dir=tmp_data_dir,
        extractor_fn=extractor_fn,
        epochs=1,
        batch_size=1,
        patch_size=(1, 1),
    )
    write_config_file(config, filetype="json")

    # run
    config_fn = tmp_data_dir / "config_hyperparams.json"
    run_grace(config_file=config_fn)
