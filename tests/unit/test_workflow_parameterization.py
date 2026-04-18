import numpy as np
import pytest

from meso_uq.workflow_acceleration import (
    active_hierarchical_variable_names,
    active_variable_names,
    configure_device_conduit,
    configure_gpu_batch_sub_experiment,
    configure_korali_conduit,
    expand_parameter_vector,
    expand_reduced_parameters,
    get_fixed_parameters,
    phase1_prior_specs,
    phase2_hyperprior_specs,
    require_single_rank,
    to_korali_path,
)


@pytest.fixture
def full_config():
    return {
        "prior_Yt": [1.0, 2.0],
        "prior_kb": [3.0, 4.0],
        "prior_b1": [5.0, 6.0],
        "prior_b2": [7.0, 8.0],
        "prior_a3": [9.0, 10.0],
        "prior_a4": [11.0, 12.0],
        "prior_d0": [13.0, 14.0],
        "prior_sigma": [15.0, 16.0],
        "hyperprior_mu_Yt": [1.0, 2.0],
        "hyperprior_sigma_Yt": [0.1, 0.2],
        "hyperprior_mu_kb": [3.0, 4.0],
        "hyperprior_sigma_kb": [0.3, 0.4],
        "hyperprior_mu_b1": [5.0, 6.0],
        "hyperprior_sigma_b1": [0.5, 0.6],
        "hyperprior_mu_b2": [7.0, 8.0],
        "hyperprior_sigma_b2": [0.7, 0.8],
        "hyperprior_mu_a3": [9.0, 10.0],
        "hyperprior_sigma_a3": [0.9, 1.0],
        "hyperprior_mu_a4": [11.0, 12.0],
        "hyperprior_sigma_a4": [1.1, 1.2],
        "hyperprior_mu_d0": [13.0, 14.0],
        "hyperprior_sigma_d0": [1.3, 1.4],
    }


def test_fixed_parameter_helpers_respect_reduced_config(full_config):
    reduced = dict(full_config)
    reduced["fixed_params"] = {"b1": 0.0, "b2": 0.0, "a3": 0.0, "a4": 0.0}

    assert get_fixed_parameters(reduced) == {"b1": 0.0, "b2": 0.0, "a3": 0.0, "a4": 0.0}
    assert active_variable_names(reduced) == ["Yt", "kb", "d0", "sigma"]
    assert active_hierarchical_variable_names(reduced) == ["Yt", "kb", "d0"]


def test_phase1_prior_specs_drop_fixed_parameters(full_config):
    reduced = dict(full_config)
    reduced["fixed_params"] = {"b1": 0.0, "b2": 0.0, "a3": 0.0, "a4": 0.0}

    prior_specs = phase1_prior_specs(reduced)

    assert [name for name, _ in prior_specs] == ["Yt", "kb", "d0", "sigma"]
    assert prior_specs[0][1] == [1.0, 2.0]
    assert prior_specs[-1][1] == [15.0, 16.0]


def test_phase2_hyperprior_specs_drop_fixed_parameters(full_config):
    reduced = dict(full_config)
    reduced["fixed_params"] = {"b1": 0.0, "b2": 0.0, "a3": 0.0, "a4": 0.0}

    hyperprior_specs = phase2_hyperprior_specs(reduced)

    assert [name for name, _, _ in hyperprior_specs] == ["Yt", "kb", "d0"]
    assert hyperprior_specs[-1][1] == [13.0, 14.0]
    assert hyperprior_specs[-1][2] == [1.3, 1.4]


def test_expand_parameter_vector_handles_reduced_legacy_and_full():
    reduced = expand_parameter_vector(
        [10.0, 20.0, 0.5, 0.1], fixed_params={"b1": 1.0, "b2": 2.0, "a3": 3.0, "a4": 4.0}
    )
    legacy = expand_parameter_vector([10.0, 20.0, 1.0, 2.0, 3.0, 4.0, 0.1])
    full = expand_parameter_vector([10.0, 20.0, 1.0, 2.0, 3.0, 4.0, 0.5, 0.1])

    assert reduced.tolist() == pytest.approx([10.0, 20.0, 1.0, 2.0, 3.0, 4.0, 0.5, 0.1])
    assert legacy.tolist() == pytest.approx([10.0, 20.0, 1.0, 2.0, 3.0, 4.0, 0.0, 0.1])
    assert full.tolist() == pytest.approx([10.0, 20.0, 1.0, 2.0, 3.0, 4.0, 0.5, 0.1])


def test_configure_device_conduit_handles_gpu_cpu_and_invalid_device() -> None:
    gpu_engine = {"Conduit": {}}
    configure_device_conduit(gpu_engine, device="gpu", mpi_ranks=8)
    assert gpu_engine == {"Conduit": {}}

    cpu_engine = {"Conduit": {}}
    configure_device_conduit(cpu_engine, device="cpu", mpi_ranks=2)
    assert cpu_engine["Conduit"] == {"Type": "Distributed", "Ranks Per Worker": 1}

    with pytest.raises(ValueError, match="--device must be 'cpu' or 'gpu'"):
        configure_device_conduit(cpu_engine, device="tpu", mpi_ranks=1)


def test_get_fixed_parameters_rejects_non_mapping() -> None:
    with pytest.raises(ValueError, match="Expected fixed_params to be a mapping"):
        get_fixed_parameters({"fixed_params": ["b1", 0.0]})


def test_configure_gpu_batch_sub_experiment_sets_required_problem_keys() -> None:
    sub_experiment = {"Problem": {}}

    def batch_model(*_args, **_kwargs) -> None:
        return None

    def single_model(*_args, **_kwargs) -> None:
        return None

    configure_gpu_batch_sub_experiment(sub_experiment, batch_model, single_model)

    assert sub_experiment["Problem"]["Use Batch Evaluation"] is True
    assert sub_experiment["Problem"]["Batch Computational Model"] is batch_model
    assert sub_experiment["Problem"]["Computational Model"] is single_model


def test_expand_reduced_parameters_rejects_invalid_shape() -> None:
    with pytest.raises(ValueError, match="Expected reduced parameter array of shape"):
        expand_reduced_parameters(np.ones((2, 3), dtype=np.float32))


def test_require_single_rank_accepts_single_rank_comm() -> None:
    class _Comm:
        @staticmethod
        def Get_size() -> int:
            return 1

    require_single_rank(_Comm(), context="phase")


def test_require_single_rank_rejects_multi_rank_comm() -> None:
    class _Comm:
        @staticmethod
        def Get_size() -> int:
            return 2

    with pytest.raises(ValueError, match="requires a single MPI rank"):
        require_single_rank(_Comm(), context="phase")


def test_configure_korali_conduit_handles_single_and_multi_rank() -> None:
    single = {"Conduit": {}}
    configure_korali_conduit(single, mpi_ranks=1)
    assert "Conduit" not in single

    multi = {}
    configure_korali_conduit(multi, mpi_ranks=4, ranks_per_worker=2)
    assert multi["Conduit"]["Type"] == "Distributed"
    assert multi["Conduit"]["Ranks Per Worker"] == 2


def test_to_korali_path_handles_relative_and_value_error_fallback(monkeypatch) -> None:
    assert to_korali_path("relative/path") == "relative/path"

    monkeypatch.setattr(
        "meso_uq.workflow_acceleration.os.path.relpath",
        lambda *_a, **_k: (_ for _ in ()).throw(ValueError("x")),
    )
    assert to_korali_path("/abs/path", base_dir="/abs") == "/abs/path"
