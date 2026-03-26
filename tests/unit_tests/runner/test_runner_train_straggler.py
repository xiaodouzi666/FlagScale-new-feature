from omegaconf import OmegaConf

from flagscale.runner.runner_train import _get_args_megatron, _update_config_train


def test_update_config_train_sets_straggler_log_dir(tmp_path):
    config = OmegaConf.create(
        {
            "experiment": {
                "exp_dir": str(tmp_path / "exp"),
                "runner": {},
                "task": {"backend": "megatron"},
            },
            "train": {
                "system": {
                    "logging": {},
                    "checkpoint": {},
                },
                "model": {},
                "data": {},
            },
        }
    )

    _update_config_train(config)

    assert config.train.system.straggler_log_dir.endswith("/logs/straggler")


def test_get_args_megatron_passes_straggler_log_dir(tmp_path):
    config = OmegaConf.create(
        {
            "experiment": {
                "exp_dir": str(tmp_path / "exp"),
                "runner": {},
                "task": {"backend": "megatron"},
            },
            "train": {
                "system": {
                    "logging": {},
                    "checkpoint": {},
                    "enable_straggler_detection": True,
                    "straggler_log_dir": str(tmp_path / "exp" / "logs" / "straggler"),
                },
                "model": {},
                "data": {},
            },
        }
    )

    args = _get_args_megatron(config)

    assert "--enable-straggler-detection" in args
    assert "--straggler-log-dir" in args
