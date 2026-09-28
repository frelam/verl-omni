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
"""CPU tests for actor-side teacher unembed distribution (Task 5).

Exercises ``set_teacher_unembeds`` (storage + key vocab) and the injection of
``teacher_unembeds`` / ``teacher_key_to_id`` into the actor micro-batch, on a
worker stand-in that does not construct a real TrainingWorker.
"""

from types import SimpleNamespace

import pytest
import torch

from verl_omni.workers.engine_workers import ActorRolloutRefWorker


class _FakeTrainingWorker:
    def __init__(self):
        self.calls = []

    def train_mini_batch(self, data):
        self.calls.append(data)
        return None


def _make_worker(role="actor"):
    w = ActorRolloutRefWorker.__new__(ActorRolloutRefWorker)
    w.role = role
    w.actor = _FakeTrainingWorker()
    w.actor.train_mini_batch = w.actor.train_mini_batch  # noqa
    # _with_routing_replay_flag requires this worker attribute.
    w.enable_routing_replay = True
    # update_actor reads actor.enable_timestep_staging off the worker config.
    w.config = SimpleNamespace(actor={})
    return w


class TestSetTeacherUnembeds:
    def test_stores_bf16_cpu_and_key_vocab(self):
        w = _make_worker()
        w.set_teacher_unembeds({"t0": torch.randn(100, 32, dtype=torch.float32), "t1": torch.randn(100, 32)})
        assert set(w._teacher_key_vocab) == {"t0", "t1"}
        for W in w._teacher_unembeds.values():
            assert W.dtype == torch.bfloat16
            assert W.device.type == "cpu"

    def test_asserts_actor_role(self):
        w = _make_worker(role="rollout")
        with pytest.raises(AssertionError, match="only valid"):
            w.set_teacher_unembeds({"t0": torch.randn(4, 4)})


class TestUpdateActorInjection:
    def test_injects_unembeds_and_key_map(self):
        from tensordict import TensorDict

        w = _make_worker()
        w.set_teacher_unembeds({"b": torch.randn(10, 4), "a": torch.randn(10, 4)})

        data = TensorDict({"input_ids": torch.zeros(2, 3, dtype=torch.long)}, batch_size=[2])
        out = w.update_actor(data)

        # train_mini_batch saw the injected NonTensorData fields (tensordict
        # flattens NonTensorData to the bare value when read back).
        seen = w.actor.calls[-1]
        unembeds = seen["teacher_unembeds"]
        assert set(unembeds.keys()) == {"a", "b"}

        key_map = seen["teacher_key_to_id"]
        assert key_map == {"a": 0, "b": 1}

        # The original reference dict passed to the worker is unmodified.
        assert out is None

    def test_no_injection_without_unembeds(self):
        from tensordict import TensorDict

        w = _make_worker()
        data = TensorDict({"input_ids": torch.zeros(2, 3, dtype=torch.long)}, batch_size=[2])
        w.update_actor(data)
        assert "teacher_unembeds" not in w.actor.calls[-1]
        assert "teacher_key_to_id" not in w.actor.calls[-1]


def _with_routing_column(data, keys: list[str]):
    from tensordict import NonTensorData, NonTensorStack

    data["teacher_key"] = NonTensorStack.from_list([NonTensorData(k) for k in keys])
    return data


class TestTeacherKeyIdsInjection:
    def test_multi_teacher_maps_routing_strings(self):
        from tensordict import TensorDict

        w = _make_worker()
        w.set_teacher_unembeds({"b": torch.randn(10, 4), "a": torch.randn(10, 4)})

        data = TensorDict({"input_ids": torch.zeros(3, 3, dtype=torch.long)}, batch_size=[3])
        _with_routing_column(data, ["b", "a", "b"])
        w.update_actor(data)

        seen = w.actor.calls[-1]
        assert seen["teacher_key_to_id"] == {"a": 0, "b": 1}
        assert seen["teacher_key_ids"].tolist() == [1, 0, 1]
        assert seen["teacher_key_ids"].dtype == torch.long

    def test_multi_teacher_unknown_key_raises(self):
        from tensordict import TensorDict

        w = _make_worker()
        w.set_teacher_unembeds({"a": torch.randn(10, 4), "b": torch.randn(10, 4)})

        data = TensorDict({"input_ids": torch.zeros(1, 3, dtype=torch.long)}, batch_size=[1])
        _with_routing_column(data, ["nope"])
        with pytest.raises(KeyError, match="nope"):
            w.update_actor(data)

    def test_multi_teacher_without_routing_column_skips(self):
        from tensordict import TensorDict

        w = _make_worker()
        w.set_teacher_unembeds({"a": torch.randn(10, 4), "b": torch.randn(10, 4)})

        data = TensorDict({"input_ids": torch.zeros(2, 3, dtype=torch.long)}, batch_size=[2])
        w.update_actor(data)
        assert "teacher_key_ids" not in w.actor.calls[-1]

    def test_single_teacher_ignores_routing_column(self):
        from tensordict import TensorDict

        w = _make_worker()
        w.set_teacher_unembeds({"t0": torch.randn(10, 4)})

        # The nitrobrew agent loop stores the placeholder "default" for the
        # single-teacher case; it must not be mapped (the loss uses id 0).
        data = TensorDict({"input_ids": torch.zeros(2, 3, dtype=torch.long)}, batch_size=[2])
        _with_routing_column(data, ["default", "default"])
        w.update_actor(data)
        assert "teacher_key_ids" not in w.actor.calls[-1]
