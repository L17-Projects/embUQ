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
    phase1_variable_names,
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


def test_phase1_variable_names_preserve_emb_defaults_and_allow_opt_outs(full_config) -> None:
    reduced = dict(full_config)
    reduced["fixed_params"] = {"b1": 0.0, "b2": 0.0, "a3": 0.0, "a4": 0.0}

    assert phase1_variable_names(reduced) == ["Yt", "kb", "d0", "sigma"]
    assert phase1_variable_names(reduced, include_sigma=False) == ["Yt", "kb", "d0"]
    assert phase1_variable_names(reduced, include_d0=False) == ["Yt", "kb", "sigma"]


def test_phase2_hyperprior_specs_drop_fixed_parameters(full_config):
    reduced = dict(full_config)
    reduced["fixed_params"] = {"b1": 0.0, "b2": 0.0, "a3": 0.0, "a4": 0.0}

    hyperprior_specs = phase2_hyperprior_specs(reduced)

    assert [name for name, _, _ in hyperprior_specs] == ["Yt", "kb", "d0"]
    assert hyperprior_specs[-1][1] == [13.0, 14.0]
    assert hyperprior_specs[-1][2] == [1.3, 1.4]


def test_direct_emb_ka_kb_phase_contract() -> None:
    config = {
        "structure": "emb",
        "phase1_contract_mode": "emb_direct_ka_kb",
        "prior_ka": [1000.0, 100000.0],
        "prior_kb": [100.0, 100000.0],
        "prior_d0": [0.0, 0.5],
        "prior_sigma": [0.001, 0.5],
        "hyperprior_mu_ka": [1000.0, 100000.0],
        "hyperprior_sigma_ka": [0.0, 50000.0],
        "hyperprior_mu_kb": [100.0, 100000.0],
        "hyperprior_sigma_kb": [0.0, 50000.0],
    }

    assert active_variable_names(config) == ["ka", "kb", "d0", "sigma"]
    assert phase1_variable_names(config, include_d0=False) == ["ka", "kb", "sigma"]
    assert active_hierarchical_variable_names(config) == ["ka", "kb"]
    assert [name for name, _bounds in phase1_prior_specs(config)] == ["ka", "kb", "d0", "sigma"]
    assert [name for name, _mu, _sigma in phase2_hyperprior_specs(config)] == ["ka", "kb"]


def test_phase1_prior_specs_apply_dataset_overrides_without_changing_variable_order() -> None:
    config = {
        "structure": "emb",
        "phase1_contract_mode": "emb_direct_ka_kb",
        "prior_ka": [1000.0, 100000.0],
        "prior_kb": [100.0, 100000.0],
        "prior_d0": [0.0, 0.5],
        "prior_sigma": [0.001, 0.5],
    }

    specs = phase1_prior_specs(
        config,
        prior_overrides={
            "ka": [17951.817681, 19959.913358],
            "kb": [8369.922251, 9702.253910],
        },
    )

    assert specs == [
        ("ka", [17951.817681, 19959.913358]),
        ("kb", [8369.922251, 9702.253910]),
        ("d0", [0.0, 0.5]),
        ("sigma", [0.001, 0.5]),
    ]


def test_gv_parameterization_uses_calibrated_material_order() -> None:
    config = {
        "structure": "gv",
        "prior_ka": [0.1, 1.1],
        "prior_kb": [0.2, 1.2],
        "prior_mu": [0.3, 1.3],
        "prior_b1": [0.4, 1.4],
        "prior_b2": [0.5, 1.5],
        "prior_a3": [0.6, 1.6],
        "prior_a4": [0.7, 1.7],
        "prior_mu_l": [0.8, 1.8],
        "prior_c": [0.9, 1.9],
        "prior_sigma": [0.01, 0.10],
        "hyperprior_mu_ka": [0.1, 1.1],
        "hyperprior_sigma_ka": [0.01, 0.2],
        "hyperprior_mu_kb": [0.2, 1.2],
        "hyperprior_sigma_kb": [0.01, 0.2],
        "hyperprior_mu_mu": [0.3, 1.3],
        "hyperprior_sigma_mu": [0.01, 0.2],
        "hyperprior_mu_b1": [0.4, 1.4],
        "hyperprior_sigma_b1": [0.01, 0.2],
        "hyperprior_mu_b2": [0.5, 1.5],
        "hyperprior_sigma_b2": [0.01, 0.2],
        "hyperprior_mu_a3": [0.6, 1.6],
        "hyperprior_sigma_a3": [0.01, 0.2],
        "hyperprior_mu_a4": [0.7, 1.7],
        "hyperprior_sigma_a4": [0.01, 0.2],
        "hyperprior_mu_mu_l": [0.8, 1.8],
        "hyperprior_sigma_mu_l": [0.01, 0.2],
        "hyperprior_mu_c": [0.9, 1.9],
        "hyperprior_sigma_c": [0.01, 0.2],
    }

    assert active_variable_names(config) == ["ka", "kb", "mu", "b1", "b2", "a3", "a4", "mu_l", "c", "sigma"]
    assert active_hierarchical_variable_names(config) == ["ka", "kb", "mu", "b1", "b2", "a3", "a4", "mu_l", "c"]
    assert [name for name, _bounds in phase1_prior_specs(config)] == [
        "ka",
        "kb",
        "mu",
        "b1",
        "b2",
        "a3",
        "a4",
        "mu_l",
        "c",
        "sigma",
    ]
    assert [name for name, _mu, _sigma in phase2_hyperprior_specs(config)] == [
        "ka",
        "kb",
        "mu",
        "b1",
        "b2",
        "a3",
        "a4",
        "mu_l",
        "c",
    ]


def test_gv_phase1_variable_names_allow_optional_d0_without_controls() -> None:
    config = {
        "structure": "gv",
        "prior_ka": [0.1, 1.1],
        "prior_kb": [0.2, 1.2],
        "prior_mu": [0.3, 1.3],
        "prior_b1": [0.4, 1.4],
        "prior_b2": [0.5, 1.5],
        "prior_a3": [0.6, 1.6],
        "prior_a4": [0.7, 1.7],
        "prior_mu_l": [0.8, 1.8],
        "prior_c": [0.9, 1.9],
        "prior_d0": [1.0, 2.0],
        "prior_sigma": [0.01, 0.10],
        "prior_temperature": [300.0, 600.0],
        "prior_pressure": [1.0, 5.0],
    }

    assert phase1_variable_names(config) == [
        "ka",
        "kb",
        "mu",
        "b1",
        "b2",
        "a3",
        "a4",
        "mu_l",
        "c",
        "sigma",
    ]
    assert phase1_variable_names(config, include_sigma=False) == [
        "ka",
        "kb",
        "mu",
        "b1",
        "b2",
        "a3",
        "a4",
        "mu_l",
        "c",
    ]
    assert phase1_variable_names(config, include_d0=True) == [
        "ka",
        "kb",
        "mu",
        "b1",
        "b2",
        "a3",
        "a4",
        "mu_l",
        "c",
        "d0",
        "sigma",
    ]

    names = phase1_variable_names(config, include_d0=True)
    assert "temperature" not in names
    assert "pressure" not in names


def test_gv_phase1_prior_specs_match_requested_contract() -> None:
    config = {
        "structure": "gv",
        "prior_ka": [0.1, 1.1],
        "prior_kb": [0.2, 1.2],
        "prior_mu": [0.3, 1.3],
        "prior_b1": [0.4, 1.4],
        "prior_b2": [0.5, 1.5],
        "prior_a3": [0.6, 1.6],
        "prior_a4": [0.7, 1.7],
        "prior_mu_l": [0.8, 1.8],
        "prior_c": [0.9, 1.9],
        "prior_d0": [1.0, 2.0],
        "prior_sigma": [0.01, 0.10],
        "prior_temperature": [300.0, 600.0],
    }

    default_specs = phase1_prior_specs(config)
    assert [name for name, _bounds in default_specs] == [
        "ka",
        "kb",
        "mu",
        "b1",
        "b2",
        "a3",
        "a4",
        "mu_l",
        "c",
        "sigma",
    ]
    assert default_specs[-1] == ("sigma", [0.01, 0.10])

    with_d0_specs = phase1_prior_specs(config, include_d0=True)
    assert [name for name, _bounds in with_d0_specs] == [
        "ka",
        "kb",
        "mu",
        "b1",
        "b2",
        "a3",
        "a4",
        "mu_l",
        "c",
        "d0",
        "sigma",
    ]
    assert ("d0", [1.0, 2.0]) in with_d0_specs
    assert all(name != "temperature" for name, _bounds in with_d0_specs)


def test_mixed_structure_parameterization_is_rejected() -> None:
    with pytest.raises(ValueError, match="Mixed-structure inference parameterization"):
        active_variable_names({"structures": ["emb", "gv"]})
    with pytest.raises(ValueError, match="Unsupported inference structure"):
        phase1_variable_names({"structure": "vesicle"})


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


def test_expand_reduced_parameters_empty_fixed_params_defaults_to_zero() -> None:
    """Regression: empty fixed_params dict must not raise KeyError on b1/b2/a3/a4."""
    result = expand_reduced_parameters(
        np.array([[10.0, 20.0, 0.5, 0.1]], dtype=np.float32),
        fixed_params={},
    )
    assert result[0].tolist() == pytest.approx([10.0, 20.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.1])


def test_expand_parameter_vector_with_empty_fixed_params_does_not_raise() -> None:
    """Regression: get_fixed_parameters returns {} when no fixed_params in config;
    expand_parameter_vector must still work for the reduced (4-param) case."""
    result = expand_parameter_vector([10.0, 20.0, 0.5, 0.1], fixed_params={})
    assert result.tolist() == pytest.approx([10.0, 20.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.1])


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
