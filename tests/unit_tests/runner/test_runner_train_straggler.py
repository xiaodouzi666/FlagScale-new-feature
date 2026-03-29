from unittest.mock import MagicMock

from omegaconf import OmegaConf

from flagscale.runner.runner_train import (
    _get_args_megatron,
    _resolve_enable_monitoring,
    _resolve_monitor_interval,
    _update_config_train,
    run_node,
)


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


def test_monitoring_aliases_resolve_correctly():
    runner_cfg = OmegaConf.create({"enable_perf_monitor": True, "perf_monitor_interval": 7})

    assert _resolve_enable_monitoring(runner_cfg) is True
    assert _resolve_monitor_interval(runner_cfg) == 7


def test_run_node_forwards_monitoring_settings():
    func = MagicMock()
    resource_info = {"slots": 8, "type": "Metax_C550"}
    user_envs = {"MACA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7"}
    runner_config = OmegaConf.create({})

    run_node(
        func,
        node_rank=0,
        host="localhost",
        resource_info=resource_info,
        user_envs=user_envs,
        runner_config=runner_config,
        nnodes=1,
        available_ip="127.0.0.1",
        available_port=29500,
        with_test=True,
        dryrun=True,
        enable_monitoring=False,
        monitor_interval=9,
    )

    _, kwargs = func.call_args
    assert kwargs["enable_monitoring"] is False
    assert kwargs["monitor_interval"] == 9
