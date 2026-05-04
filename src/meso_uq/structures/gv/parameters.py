from __future__ import annotations

from ..registry import NoiseModelSpec, ParameterContract, ParameterSpec


GV_MATERIAL_PARAMETER_NAMES = ("ka", "kb", "mu", "b1", "b2", "a3", "a4", "mu_l", "c")

GV_KA = ParameterSpec("ka", r"$k_a$", "In-plane area modulus.", "DPD energy / area")
GV_KB = ParameterSpec("kb", r"$k_b$", "Bending modulus.", "DPD energy")
GV_MU = ParameterSpec("mu", r"$\mu$", "In-plane shear modulus.", "DPD energy / area")
GV_B1 = ParameterSpec("b1", r"$b_1$", "First strain-limiting coefficient.", "dimensionless")
GV_B2 = ParameterSpec("b2", r"$b_2$", "Second strain-limiting coefficient.", "dimensionless")
GV_A3 = ParameterSpec("a3", r"$a_3$", "Third-order area strain coefficient.", "dimensionless")
GV_A4 = ParameterSpec("a4", r"$a_4$", "Fourth-order area strain coefficient.", "dimensionless")
GV_MU_L = ParameterSpec("mu_l", r"$\mu_l$", "Longitudinal shear modulus term from orthotropic fit.", "DPD energy / area")
GV_C = ParameterSpec("c", r"$c$", "Orthotropic coupling modulus.", "DPD energy / area")

GV_D0 = ParameterSpec(
    "d0",
    r"$d_0$",
    "Optional response offset used only when observable alignment requires it.",
    "observable units",
    is_calibrated=False,
    is_nuisance=True,
    optional=True,
)
GV_SIGMA = ParameterSpec(
    "sigma",
    r"$\sigma$",
    "Observation noise scale for multiplicative uncertainty.",
    "dimensionless",
    is_calibrated=False,
    is_nuisance=True,
)

GV_NOISE_MODEL = NoiseModelSpec(
    kind="multiplicative",
    parameter="sigma",
    description="Apply multiplicative observation noise to the predicted observable.",
)

GV_PARAMETER_CONTRACT = ParameterContract(
    calibrated=(GV_KA, GV_KB, GV_MU, GV_B1, GV_B2, GV_A3, GV_A4, GV_MU_L, GV_C),
    nuisance=(GV_D0, GV_SIGMA),
    noise_model=GV_NOISE_MODEL,
)
