# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CPU tests for verl_omni.utils.config validation."""

import pytest
from omegaconf import OmegaConf

from verl_omni.utils.config import validate_config


def _config(**trainer):
    return OmegaConf.create({"trainer": {"resume_mode": "disable", **trainer}})


def test_validate_config_rejects_unknown_resume_mode():
    with pytest.raises(ValueError, match="Available options"):
        validate_config(_config(resume_mode="resumee"))


def test_validate_config_requires_resume_path():
    with pytest.raises(ValueError, match="resume_from_path"):
        validate_config(_config(resume_mode="resume_path"))


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("sp_size", [1, 2])
@pytest.mark.parametrize("as_dict", [False, True])
def test_validate_config_timestep_staging(enabled, sp_size, as_dict):
    config = _config()
    config.actor_rollout_ref = {
        "actor": {"enable_timestep_staging": enabled, "fsdp_config": {"ulysses_sequence_parallel_size": sp_size}}
    }
    if as_dict:
        config = OmegaConf.to_container(config)
    if enabled and sp_size != 1:
        with pytest.raises(ValueError, match="sequence_parallel_size=1"):
            validate_config(config)
    else:
        validate_config(config)


def _delta_config(**overrides):
    """A minimal omni_delta_sharded config that passes the gates; overrides merge on top."""
    base = {
        "trainer": {"resume_mode": "disable", "use_v1": True, "v1": {"trainer_mode": "separate_async"}},
        "actor_rollout_ref": {
            "model": {"lora": {"rank": 0}},
            "actor": {"fsdp_config": {"qat": {"enable": False}}},
            "rollout": {"checkpoint_engine": {"backend": "omni_delta_sharded"}},
        },
    }
    return OmegaConf.merge(OmegaConf.create(base), OmegaConf.create(overrides))


def test_raw_delta_sharded_backend_is_rejected_with_guidance():
    """verl gates the raw delta_sharded name to sglang; point users at the omni name."""
    config = _delta_config(actor_rollout_ref={"rollout": {"checkpoint_engine": {"backend": "delta_sharded"}}})
    with pytest.raises(ValueError, match="omni_delta_sharded"):
        validate_config(config)


def test_delta_sharded_admitted_for_full_weight_separate_async():
    validate_config(_delta_config())  # diffusion v1 separate_async, full-weight

    # omni_separate_async is admitted without trainer.use_v1 (main_omni forces v1 later)
    config = _delta_config()
    OmegaConf.update(config, "trainer.v1.trainer_mode", "omni_separate_async")
    OmegaConf.update(config, "trainer.use_v1", False)
    validate_config(config)


@pytest.mark.parametrize("mode", ["sync", "omni_sync", "colocate_async"])
def test_delta_sharded_rejects_non_separate_async_modes(mode):
    with pytest.raises(ValueError, match="trainer_mode"):
        validate_config(_delta_config(trainer={"v1": {"trainer_mode": mode}}))


def test_delta_sharded_rejects_legacy_diffusion_trainer():
    # trainer.v1.trainer_mode=separate_async but the v0 runner was selected
    with pytest.raises(ValueError, match="use_v1"):
        validate_config(_delta_config(trainer={"use_v1": False}))


@pytest.mark.parametrize(
    "model_override",
    [
        {"lora": {"rank": 8}},  # adapter sync (merge=false default)
        {"lora": {"rank": 8, "merge": True}},  # merged full-weight export
        {"lora_rank": 16},  # legacy knob
        {"lora_adapter_path": "/tmp/adapter"},
    ],
)
def test_delta_sharded_rejects_lora(model_override):
    with pytest.raises(ValueError, match="full-weight"):
        validate_config(_delta_config(actor_rollout_ref={"model": model_override}))


@pytest.mark.parametrize(
    "actor_override",
    [
        {"fsdp_config": {"qat": {"enable": True}}},
        {"megatron": {"qat": {"enable": True}}},
    ],
)
def test_delta_sharded_rejects_qat(actor_override):
    with pytest.raises(ValueError, match="QAT"):
        validate_config(_delta_config(actor_rollout_ref={"actor": actor_override}))


def test_delta_gates_do_not_fire_for_other_backends():
    config = _delta_config(
        trainer={"use_v1": False, "v1": {"trainer_mode": "sync"}},
        actor_rollout_ref={
            "model": {"lora": {"rank": 8}},
            "rollout": {"checkpoint_engine": {"backend": "nccl"}},
        },
    )
    validate_config(config)


@pytest.mark.parametrize("strategy", ["fsdp", "fsdp2", "veomni", "megatron"])
@pytest.mark.parametrize("enabled", [False, True])
def test_validate_config_no_sync_gradient_accumulation(strategy, enabled):
    config = _config()
    config.actor_rollout_ref = {"actor": {"strategy": strategy, "use_no_sync_for_gradient_accumulation": enabled}}
    if enabled and strategy not in ("fsdp", "fsdp2"):
        with pytest.raises(ValueError, match="fsdp or fsdp2"):
            validate_config(config)
    else:
        validate_config(config)


def test_dynamic_resource_scheduling_default_off_is_admitted():
    validate_config(_config())
    config = _config()
    config.async_training = {"use_dynamic_resource_scheduling": False}
    validate_config(config)


def test_dynamic_resource_scheduling_raises_on_v1_entrypoints():
    config = _config()
    config.async_training = {"use_dynamic_resource_scheduling": True}
    with pytest.raises(ValueError, match="hybrid_rollout.enable_switch"):
        validate_config(config)


def _router_replay_config(
    *, mode="disabled", capture=False, strategy="megatron", use_remove_padding=True, pad_to_length=False
):
    config = _config()
    config.actor_rollout_ref = {
        "actor": {
            "strategy": strategy,
            "megatron": {
                "router_replay": {"mode": mode},
                "use_remove_padding": use_remove_padding,
                "pad_to_length": pad_to_length,
            },
        },
        "rollout": {"enable_rollout_routing_replay": capture},
    }
    return config


@pytest.mark.parametrize(("mode", "capture"), [("disabled", False), ("R2", False), ("R3", True)])
def test_validate_config_admits_consistent_router_replay(mode, capture):
    validate_config(_router_replay_config(mode=mode, capture=capture))


def test_validate_config_r3_requires_rollout_capture():
    with pytest.raises(ValueError, match="enable_rollout_routing_replay=true"):
        validate_config(_router_replay_config(mode="R3", capture=False))


@pytest.mark.parametrize("mode", ["disabled", "R2"])
def test_validate_config_rollout_capture_requires_r3(mode):
    with pytest.raises(ValueError, match="only consumed by"):
        validate_config(_router_replay_config(mode=mode, capture=True))


def test_validate_config_router_replay_requires_megatron():
    with pytest.raises(ValueError, match="strategy='megatron'"):
        validate_config(_router_replay_config(mode="R2", strategy="fsdp"))


def test_validate_config_r3_requires_thd_packing():
    with pytest.raises(ValueError, match="THD packing"):
        validate_config(_router_replay_config(mode="R3", capture=True, use_remove_padding=False))


def test_validate_config_r3_rejects_pad_to_length():
    with pytest.raises(ValueError, match="pad_to_length"):
        validate_config(_router_replay_config(mode="R3", capture=True, pad_to_length=True))
