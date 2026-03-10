import os
import sys
import types

from omegaconf import OmegaConf

hydra_module = types.ModuleType("hydra")
hydra_core_module = types.ModuleType("hydra.core")
hydra_config_module = types.ModuleType("hydra.core.hydra_config")


class _HydraConfig:
    @staticmethod
    def get():
        raise RuntimeError("HydraConfig.get() should not be called in this test")


hydra_config_module.HydraConfig = _HydraConfig
hydra_core_module.hydra_config = hydra_config_module
hydra_module.core = hydra_core_module
sys.modules.setdefault("hydra", hydra_module)
sys.modules.setdefault("hydra.core", hydra_core_module)
sys.modules.setdefault("hydra.core.hydra_config", hydra_config_module)

from flagscale.runner.runner_train import _get_args_megatron, _update_config_train


def _build_config():
    return OmegaConf.create(
        {
            "experiment": {
                "exp_dir": "./outputs_gpt2",
                "task": {
                    "type": "train",
                    "backend": "megatron",
                    "entrypoint": "./flagscale/train/train_gpt.py",
                },
                "runner": {
                    "backend": "torchrun",
                    "nnodes": 1,
                    "nproc_per_node": 8,
                },
            },
            "train": {
                "system": {
                    "enable_straggler_detection": True,
                    "straggler_profiling_interval": 5,
                    "straggler_report_interval": 10,
                    "straggler_threshold": 1.8,
                    "straggler_log_dir": "./outputs_gpt2/logs/straggler",
                    "logging": {},
                },
                "model": {},
                "data": {},
            },
        }
    )


def test_update_config_train_resolves_straggler_log_dir(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    config = _build_config()

    _update_config_train(config)

    assert config.train.system.straggler_log_dir == os.path.join(
        str(tmp_path), "outputs_gpt2", "logs", "straggler"
    )


def test_get_args_megatron_includes_straggler_cli_args(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    config = _build_config()

    _update_config_train(config)
    args = _get_args_megatron(config)

    assert "--enable-straggler-detection" in args

    profiling_idx = args.index("--straggler-profiling-interval")
    assert args[profiling_idx + 1] == "5"

    report_idx = args.index("--straggler-report-interval")
    assert args[report_idx + 1] == "10"

    threshold_idx = args.index("--straggler-threshold")
    assert args[threshold_idx + 1] == "1.8"

    log_dir_idx = args.index("--straggler-log-dir")
    assert args[log_dir_idx + 1] == os.path.join(
        str(tmp_path), "outputs_gpt2", "logs", "straggler"
    )
