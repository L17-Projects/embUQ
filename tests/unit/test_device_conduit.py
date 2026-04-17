"""Unit tests for configure_device_conduit and configure_gpu_batch_sub_experiment."""

import pytest

from meso_uq.workflow_acceleration import (
    configure_device_conduit,
    configure_gpu_batch_sub_experiment,
)

# ---------------------------------------------------------------------------
# configure_device_conduit
# ---------------------------------------------------------------------------


def test_configure_device_conduit_gpu_makes_no_changes() -> None:
    engine = {}
    configure_device_conduit(engine, device="gpu", mpi_ranks=1)
    assert engine == {}


def test_configure_device_conduit_gpu_ignores_mpi_ranks() -> None:
    engine = {}
    configure_device_conduit(engine, device="gpu", mpi_ranks=8)
    assert engine == {}


def test_configure_device_conduit_cpu_single_rank_no_distributed() -> None:
    engine = {}
    configure_device_conduit(engine, device="cpu", mpi_ranks=1)
    # Single rank CPU → no Distributed conduit required
    assert "Conduit" not in engine or engine.get("Conduit", {}).get("Type") != "Distributed"


def test_configure_device_conduit_cpu_multi_rank_sets_distributed() -> None:
    # Korali Engine auto-creates sub-keys; simulate that with pre-initialised dict
    engine = {"Conduit": {}}
    configure_device_conduit(engine, device="cpu", mpi_ranks=4)
    assert engine["Conduit"]["Type"] == "Distributed"
    assert engine["Conduit"]["Ranks Per Worker"] == 1


def test_configure_device_conduit_cpu_default_ranks_is_single() -> None:
    engine = {}
    configure_device_conduit(engine, device="cpu")
    assert "Conduit" not in engine or engine.get("Conduit", {}).get("Type") != "Distributed"


def test_configure_device_conduit_invalid_device_raises() -> None:
    engine = {}
    with pytest.raises(ValueError, match="--device must be"):
        configure_device_conduit(engine, device="tpu", mpi_ranks=1)


def test_configure_device_conduit_invalid_device_message_includes_value() -> None:
    engine = {}
    with pytest.raises(ValueError, match="tpu"):
        configure_device_conduit(engine, device="tpu")


# ---------------------------------------------------------------------------
# configure_gpu_batch_sub_experiment
# ---------------------------------------------------------------------------


def test_configure_gpu_batch_sub_experiment_sets_all_three_keys() -> None:
    sub: dict = {"Problem": {}}
    batch_fn = lambda s: None  # noqa: E731
    single_fn = lambda s: None  # noqa: E731

    configure_gpu_batch_sub_experiment(sub, batch_model_fn=batch_fn, single_model_fn=single_fn)

    assert sub["Problem"]["Use Batch Evaluation"] is True
    assert sub["Problem"]["Batch Computational Model"] is batch_fn
    assert sub["Problem"]["Computational Model"] is single_fn


def test_configure_gpu_batch_sub_experiment_batch_differs_from_single() -> None:
    sub: dict = {"Problem": {}}
    batch_fn = lambda s: "batch"  # noqa: E731
    single_fn = lambda s: "single"  # noqa: E731

    configure_gpu_batch_sub_experiment(sub, batch_model_fn=batch_fn, single_model_fn=single_fn)

    assert sub["Problem"]["Batch Computational Model"] is not sub["Problem"]["Computational Model"]
