from typing import List, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


EMB_EXPERIMENTS = {"compression", "indentation", "resonance"}
STRUCTURE_EXPERIMENTS = {
    "emb": EMB_EXPERIMENTS,
}
SUPPORTED_STRUCTURES = {"emb"}
EMB_PHASE1_PRIOR_FIELDS = (
    "prior_Yt",
    "prior_kb",
    "prior_b1",
    "prior_b2",
    "prior_a3",
    "prior_a4",
    "prior_d0",
    "prior_sigma",
)
EMB_GENERIC_DIRECT_PHASE1_CONTRACT_MODE = "emb_direct_ka_kb"
EMB_DIRECT_PHASE1_PRIOR_FIELDS = (
    "prior_ka",
    "prior_kb",
    "prior_d0",
    "prior_sigma",
)
PHASE1_PRIOR_FIELDS = EMB_PHASE1_PRIOR_FIELDS


def format_emb_diameter(diameter_um: float) -> str:
    text = f"{float(diameter_um):.6f}".rstrip("0").rstrip(".")
    if "." not in text:
        text = f"{text}.0"
    return text


def emb_geometry_id(diameter_um: float) -> str:
    return f"diameter_{format_emb_diameter(diameter_um)}um"


def infer_structure(experiment_name: Optional[str], explicit_structure: Optional[str] = None) -> Optional[str]:
    if explicit_structure is not None:
        if explicit_structure not in SUPPORTED_STRUCTURES:
            raise ValueError(
                f"Unsupported structure '{explicit_structure}'. Expected one of {sorted(SUPPORTED_STRUCTURES)}"
            )
        _validate_experiment_structure_pair(experiment_name, explicit_structure)
        return explicit_structure
    if experiment_name in EMB_EXPERIMENTS:
        return "emb"
    return None


def _validate_experiment_structure_pair(
    experiment_name: Optional[str],
    structure: str,
) -> None:
    if experiment_name is None:
        return
    for expected_structure, experiment_names in STRUCTURE_EXPERIMENTS.items():
        if experiment_name in experiment_names and structure != expected_structure:
            raise ValueError(
                f"Experiment '{experiment_name}' does not belong to structure '{structure}'. "
                f"It belongs to structure '{expected_structure}'."
            )


class ExperimentSelection(BaseModel):
    structure: Optional[str] = None
    name: str
    lane: Optional[str] = None
    geometries: Optional[List[str]] = None
    controls: Optional[List[str]] = None
    diameters: Optional[List[float]] = None
    enabled: bool = True
    grouped_reference_data: bool = False
    reference_diameters: Optional[List[float]] = None
    prior_ka: Optional[List[float]] = Field(default=None, min_length=2, max_length=2)
    prior_kb: Optional[List[float]] = Field(default=None, min_length=2, max_length=2)
    prior_d0: Optional[List[float]] = Field(default=None, min_length=2, max_length=2)
    prior_sigma: Optional[List[float]] = Field(default=None, min_length=2, max_length=2)
    prior_ka_by_diameter_um: Optional[dict] = None
    prior_kb_by_diameter_um: Optional[dict] = None

    @field_validator("structure")
    @classmethod
    def validate_structure(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        if value not in SUPPORTED_STRUCTURES:
            raise ValueError(f"Unsupported structure '{value}'")
        return value

    @field_validator("diameters")
    @classmethod
    def validate_diameters(cls, value: Optional[List[float]]) -> Optional[List[float]]:
        if value is None:
            return value
        if any(diameter <= 0 for diameter in value):
            raise ValueError("All diameters must be positive")
        return sorted(value)

    @field_validator("reference_diameters")
    @classmethod
    def validate_reference_diameters(cls, value: Optional[List[float]]) -> Optional[List[float]]:
        if value is None:
            return value
        if any(diameter <= 0 for diameter in value):
            raise ValueError("All reference diameters must be positive")
        return value

    @model_validator(mode="after")
    def normalize_legacy_geometry_fields(self) -> "ExperimentSelection":
        self.structure = infer_structure(self.name, self.structure)
        if self.structure is None:
            raise ValueError(
                f"Experiment '{self.name}' requires an explicit structure to avoid ambiguous references"
            )
        if self.structure == "emb" and self.diameters:
            normalized = [emb_geometry_id(diameter) for diameter in self.diameters]
            self.geometries = sorted(set((self.geometries or []) + normalized))
        if self.geometries is not None:
            self.geometries = sorted(set(self.geometries))
        if self.controls is not None:
            self.controls = sorted(set(self.controls))
        return self


class PriorBounds(BaseModel):
    min_val: float
    max_val: float

    @model_validator(mode="after")
    def check_bounds_order(self):
        if self.min_val >= self.max_val:
            raise ValueError(f"min_val ({self.min_val}) must be less than max_val ({self.max_val})")
        return self

    def as_tuple(self) -> Tuple[float, float]:
        return (self.min_val, self.max_val)

    def as_list(self) -> List[float]:
        return [self.min_val, self.max_val]

    def contains(self, value: float) -> bool:
        return self.min_val <= value <= self.max_val

    @classmethod
    def from_list(cls, bounds: List[float]) -> "PriorBounds":
        if len(bounds) != 2:
            raise ValueError(f"Bounds must have exactly 2 elements, got {len(bounds)}")
        return cls(min_val=bounds[0], max_val=bounds[1])


class HyperpriorBounds(BaseModel):
    mu_min: float
    mu_max: float
    sigma_min: float = 0.0
    sigma_max: float

    @model_validator(mode="after")
    def check_bounds_order(self):
        if self.mu_min >= self.mu_max:
            raise ValueError(f"mu bounds: {self.mu_min} must be < {self.mu_max}")
        if self.sigma_min >= self.sigma_max:
            raise ValueError(f"sigma bounds: {self.sigma_min} must be < {self.sigma_max}")
        if self.sigma_min < 0:
            raise ValueError(f"sigma_min must be >= 0, got {self.sigma_min}")
        return self


class TMCMCParams(BaseModel):
    pop_size: int = Field(ge=100, le=500000)
    max_gen: int = Field(default=-1)
    target_cov: float = Field(ge=0.1, le=1.0)
    covariance_scaling: float = Field(ge=0.001, le=1.0)


class ResonanceEvaluatorConfig(BaseModel):
    """Typed selection of an analytical or artifact-backed EMB resonance map."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal[
        "analytical_dpd_formula",
        "analytical_vacuum_shell",
        "artifact_surface",
        "artifact_emulator_bank",
        "artifact_polynomial_bank",
    ] = "analytical_dpd_formula"
    artifact_path: Optional[str] = None
    artifact_sha256: Optional[str] = None
    expected_fixed_kb_dpd: Optional[float] = Field(default=None, gt=0.0)
    provenance_path_overrides: dict[str, str] = Field(default_factory=dict)
    bank_build_report_path: Optional[str] = None
    bank_build_report_sha256: Optional[str] = None
    independent_go_path: Optional[str] = None
    independent_go_sha256: Optional[str] = None
    promotion_contract_path: Optional[str] = None
    promotion_contract_sha256: Optional[str] = None
    allow_benchmark_artifact: bool = False

    @field_validator(
        "artifact_sha256",
        "bank_build_report_sha256",
        "independent_go_sha256",
        "promotion_contract_sha256",
    )
    @classmethod
    def validate_artifact_sha256(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        normalized = value.strip().lower()
        if len(normalized) != 64 or any(
            character not in "0123456789abcdef" for character in normalized
        ):
            raise ValueError(
                "resonance.evaluator.artifact_sha256 must be a 64-character "
                "hexadecimal SHA-256 digest."
            )
        return normalized

    @model_validator(mode="after")
    def validate_artifact_mode(self) -> "ResonanceEvaluatorConfig":
        artifact_modes = {
            "artifact_surface",
            "artifact_emulator_bank",
            "artifact_polynomial_bank",
        }
        if self.mode in artifact_modes and not self.artifact_path:
            raise ValueError(
                "resonance.evaluator.artifact_path is required for artifact-backed modes."
            )
        analytical_modes = {"analytical_dpd_formula", "analytical_vacuum_shell"}
        if self.mode in analytical_modes and self.artifact_path is not None:
            raise ValueError(
                "resonance.evaluator.artifact_path is only valid for artifact-backed modes."
            )
        if self.mode in analytical_modes and self.artifact_sha256 is not None:
            raise ValueError(
                "resonance.evaluator.artifact_sha256 is only valid for artifact-backed modes."
            )
        if self.mode in analytical_modes and self.expected_fixed_kb_dpd is not None:
            raise ValueError(
                "resonance.evaluator.expected_fixed_kb_dpd is only valid for artifact-backed modes."
            )
        if self.mode == "artifact_polynomial_bank":
            required = {
                "artifact_sha256": self.artifact_sha256,
                "expected_fixed_kb_dpd": self.expected_fixed_kb_dpd,
            }
            if not self.allow_benchmark_artifact:
                required.update(
                    {
                        "bank_build_report_path": self.bank_build_report_path,
                        "bank_build_report_sha256": self.bank_build_report_sha256,
                        "independent_go_path": self.independent_go_path,
                        "independent_go_sha256": self.independent_go_sha256,
                        "promotion_contract_path": self.promotion_contract_path,
                        "promotion_contract_sha256": self.promotion_contract_sha256,
                    }
                )
            missing = sorted(name for name, value in required.items() if value is None)
            if missing:
                raise ValueError(
                    "artifact_polynomial_bank requires pinned release fields: "
                    + ", ".join(missing)
                )
        return self


class ResonanceConfig(BaseModel):
    """Resonance schema while preserving established analytical YAML fields."""

    model_config = ConfigDict(extra="allow")

    agent: Optional[str] = None
    evaluator: Optional[ResonanceEvaluatorConfig] = None
    excluded_diameters_um: Optional[List[float]] = None
    excluded_diameters_reason: Optional[str] = None

    @field_validator("excluded_diameters_um")
    @classmethod
    def validate_excluded_diameters(cls, value: Optional[List[float]]) -> Optional[List[float]]:
        if value is None:
            return value
        if any(diameter <= 0.0 for diameter in value):
            raise ValueError("resonance.excluded_diameters_um must contain only positive values.")
        return sorted(set(float(diameter) for diameter in value))

    @model_validator(mode="after")
    def validate_exclusion_reason(self) -> "ResonanceConfig":
        if self.excluded_diameters_um and not self.excluded_diameters_reason:
            raise ValueError(
                "resonance.excluded_diameters_reason is required when diameters are excluded."
            )
        return self


class InferenceConfig(BaseModel):
    emb_diameters: Optional[List[float]] = Field(default=None, min_length=1)
    frac_diam: float = Field(ge=0.1, le=0.5, default=0.2)
    use_surrogate: bool = Field(default=True)
    pop_size: int = Field(ge=100, le=500000, default=50000)
    max_gen: int = Field(default=-1)
    target_cov: float = Field(ge=0.1, le=1.0, default=0.8)
    covariance_scaling: float = Field(ge=0.001, le=1.0, default=0.04)
    phase1_burn_in: Optional[int] = Field(ge=0, default=None)
    prior_Yt: Optional[List[float]] = Field(default=None, min_length=2, max_length=2)
    prior_ka: Optional[List[float]] = Field(default=None, min_length=2, max_length=2)
    prior_kb: Optional[List[float]] = Field(default=None, min_length=2, max_length=2)
    prior_mu: Optional[List[float]] = Field(default=None, min_length=2, max_length=2)
    prior_b1: Optional[List[float]] = Field(default=None, min_length=2, max_length=2)
    prior_b2: Optional[List[float]] = Field(default=None, min_length=2, max_length=2)
    prior_a3: Optional[List[float]] = Field(default=None, min_length=2, max_length=2)
    prior_a4: Optional[List[float]] = Field(default=None, min_length=2, max_length=2)
    prior_mu_l: Optional[List[float]] = Field(default=None, min_length=2, max_length=2)
    prior_c: Optional[List[float]] = Field(default=None, min_length=2, max_length=2)
    prior_d0: Optional[List[float]] = Field(default=None, min_length=2, max_length=2)
    prior_sigma: Optional[List[float]] = Field(default=None, min_length=2, max_length=2)
    hbi_pop_size: int = Field(ge=100, le=500000, default=50000)
    hbi_burn_in: int = Field(ge=0, default=1)
    hbi_target_cov: float = Field(ge=0.1, le=1.0, default=0.6)
    hbi_covariance_scaling: float = Field(ge=0.001, le=1.0, default=0.02)
    phase3b_pop_size: int = Field(ge=100, le=500000, default=50000)
    phase3b_max_gen: int = Field(default=-1)
    phase3b_target_cov: float = Field(ge=0.1, le=1.0, default=0.6)
    phase3b_covariance_scaling: float = Field(ge=0.001, le=1.0, default=0.02)
    hyperprior_mu_Yt: Optional[List[float]] = None
    hyperprior_sigma_Yt: Optional[List[float]] = None
    hyperprior_mu_ka: Optional[List[float]] = None
    hyperprior_sigma_ka: Optional[List[float]] = None
    hyperprior_mu_kb: Optional[List[float]] = None
    hyperprior_sigma_kb: Optional[List[float]] = None
    hyperprior_mu_mu: Optional[List[float]] = None
    hyperprior_sigma_mu: Optional[List[float]] = None
    hyperprior_mu_b1: Optional[List[float]] = None
    hyperprior_sigma_b1: Optional[List[float]] = None
    hyperprior_mu_b2: Optional[List[float]] = None
    hyperprior_sigma_b2: Optional[List[float]] = None
    hyperprior_mu_a3: Optional[List[float]] = None
    hyperprior_sigma_a3: Optional[List[float]] = None
    hyperprior_mu_a4: Optional[List[float]] = None
    hyperprior_sigma_a4: Optional[List[float]] = None
    hyperprior_mu_mu_l: Optional[List[float]] = None
    hyperprior_sigma_mu_l: Optional[List[float]] = None
    hyperprior_mu_c: Optional[List[float]] = None
    hyperprior_sigma_c: Optional[List[float]] = None
    hyperprior_mu_d0: Optional[List[float]] = None
    hyperprior_sigma_d0: Optional[List[float]] = None
    out: str = Field(default="out_hierarchical")
    debug: int = Field(ge=0, le=2, default=0)
    dump: bool = Field(default=False)
    description: str = Field(default="")
    structure: Optional[str] = Field(default=None)
    structures: Optional[List[str]] = Field(default=None)
    experiment: Optional[str] = Field(default=None)
    data_dir: Optional[str] = Field(default=None)
    data_prefix: Optional[str] = Field(default=None)
    data_files: Optional[dict] = Field(default=None)
    surrogate_dir: Optional[str] = Field(default=None)
    surrogate: Optional[dict] = Field(default=None)
    phase1_contract_mode: Optional[str] = Field(default=None)
    experimental: Optional[dict] = Field(default=None)
    resonance: Optional[ResonanceConfig] = Field(default=None)
    experimental_gv_hbi: bool = Field(default=False)
    geometries: Optional[List[str]] = Field(default=None)
    experiments: Optional[List[ExperimentSelection]] = Field(default=None)
    calibrated_parameters: Optional[List[str]] = Field(default=None)
    direct_compression_prior_extrapolation: Optional[dict] = Field(default=None)

    @field_validator("emb_diameters")
    @classmethod
    def validate_diameters(cls, v):
        if v is None:
            return v
        if any(d <= 0 for d in v):
            raise ValueError("All diameters must be positive")
        return sorted(v)

    @field_validator("structure")
    @classmethod
    def validate_structure(cls, value: Optional[str]) -> Optional[str]:
        return infer_structure(None, value)

    @field_validator("structures")
    @classmethod
    def validate_structures(cls, value: Optional[List[str]]) -> Optional[List[str]]:
        if value is None:
            return value
        normalized = sorted(set(value))
        invalid = [structure for structure in normalized if structure not in SUPPORTED_STRUCTURES]
        if invalid:
            raise ValueError(f"Unsupported structures: {invalid}")
        return normalized

    @field_validator(*PHASE1_PRIOR_FIELDS)
    @classmethod
    def validate_prior_bounds(cls, v):
        if v is None:
            return v
        if len(v) != 2:
            raise ValueError("Prior bounds must have exactly 2 elements [min, max]")
        if v[0] >= v[1]:
            raise ValueError(f"Prior min ({v[0]}) must be less than max ({v[1]})")
        return v

    @model_validator(mode="after")
    def resolve_phase1_burn_in(self):
        if self.phase1_burn_in is None:
            self.phase1_burn_in = self.hbi_burn_in
        self.structure = infer_structure(self.experiment, self.structure)
        if self.structure is None and self.experiment is not None:
            raise ValueError(
                f"Experiment '{self.experiment}' requires an explicit structure to avoid ambiguous references"
            )
        if self.structures is None and self.structure is not None:
            self.structures = [self.structure]
        elif self.structures is not None and self.structure is not None and self.structure not in self.structures:
            self.structures = sorted(set(self.structures + [self.structure]))
        if self.geometries is None and self.emb_diameters:
            self.geometries = [emb_geometry_id(diameter) for diameter in self.emb_diameters]
        if self.experiments is None and self.experiment is not None:
            self.experiments = [
                ExperimentSelection(
                    structure=self.structure,
                    name=self.experiment,
                    geometries=self.geometries,
                    diameters=self.emb_diameters,
                )
            ]
        if self.experiments:
            seen = set()
            for selection in self.experiments:
                if selection.structure is None:
                    selection.structure = infer_structure(selection.name, self.structure)
                if selection.structure is None:
                    raise ValueError(
                        f"Experiment '{selection.name}' requires an explicit structure to avoid ambiguous references"
                    )
                if selection.structure == "emb" and selection.geometries is None and self.emb_diameters:
                    selection.geometries = [emb_geometry_id(diameter) for diameter in self.emb_diameters]
                key = (selection.structure, selection.name, selection.lane)
                if key in seen:
                    raise ValueError(
                        f"Duplicate experiment selection for structure '{selection.structure}' "
                        f"and experiment '{selection.name}' lane '{selection.lane}'"
                    )
                seen.add(key)
        active_structures = set(self.structures or [])
        if self.structure is not None:
            active_structures.add(self.structure)
        if self.experiments:
            active_structures.update(selection.structure for selection in self.experiments if selection.structure)
        if not active_structures:
            active_structures = {"emb"}
        if self.phase1_contract_mode is not None:
            valid_contract_modes = {EMB_GENERIC_DIRECT_PHASE1_CONTRACT_MODE}
            if self.phase1_contract_mode not in valid_contract_modes:
                raise ValueError(
                    f"Unsupported Phase 1 contract mode '{self.phase1_contract_mode}'. "
                    f"Expected one of {sorted(valid_contract_modes)}."
                )
        if "emb" in active_structures:
            if self.phase1_contract_mode == EMB_GENERIC_DIRECT_PHASE1_CONTRACT_MODE:
                self._require_prior_fields(EMB_DIRECT_PHASE1_PRIOR_FIELDS, structure="direct EMB")
            else:
                self._require_prior_fields(EMB_PHASE1_PRIOR_FIELDS, structure="EMB")
        return self

    def _require_prior_fields(self, fields: Tuple[str, ...], *, structure: str) -> None:
        missing = [field for field in fields if getattr(self, field) is None]
        if missing:
            raise ValueError(f"Missing required {structure} Phase 1 prior bounds: {', '.join(missing)}")

    def get_prior_bounds(self, param_name: str) -> PriorBounds:
        bounds_list = getattr(self, f"prior_{param_name}", None)
        if bounds_list is None:
            raise ValueError(f"Unknown parameter: {param_name}")
        return PriorBounds.from_list(bounds_list)


class SamplingConfig(BaseModel):
    n_samples: int = Field(ge=10, le=100000)
    seed: int = Field(ge=0)
    diameter_um: float = Field(gt=0)
    output_dir: str
    checkpoint_frequency: int = Field(ge=1, default=100)
    parameter_bounds: dict
    nodes: int = Field(ge=1, default=1)
    ranks_per_node: int = Field(ge=1, default=4)
    gpus_per_rank: int = Field(ge=0, default=1)


class PropagationConfig(BaseModel):
    method: str = Field(pattern="^(mc|ut|both)$")
    mc_n_samples: Optional[int] = Field(ge=100, le=100000, default=1000)
    mc_seed: Optional[int] = Field(ge=0, default=42)
    ut_alpha: Optional[float] = Field(ge=0.0, le=1.0, default=1.0)
    ut_beta: Optional[float] = Field(default=2.0)
    ut_kappa: Optional[float] = Field(ge=-10.0, le=10.0, default=0.0)
    diameter_um: float = Field(gt=0)
    posterior_samples_file: str
    output_dir: str
    timeout_hours: float = Field(ge=0.5, le=48.0, default=8.0)
    max_retries: int = Field(ge=0, le=5, default=3)


def create_default_inference_config() -> InferenceConfig:
    return InferenceConfig(
        emb_diameters=[2.1, 2.9, 3.0],
        frac_diam=0.2,
        use_surrogate=True,
        pop_size=50000,
        max_gen=-1,
        target_cov=0.8,
        covariance_scaling=0.04,
        phase1_burn_in=1,
        prior_Yt=[10000000.0, 50000000.0],
        prior_kb=[100.0, 1000.0],
        prior_b1=[0.0, 3.0],
        prior_b2=[0.0, 10.0],
        prior_a3=[-2.5, 3.0],
        prior_a4=[0.0, 4.0],
        prior_d0=[0.0, 0.5],
        prior_sigma=[0.0, 1.0],
        hbi_pop_size=50000,
        hbi_burn_in=1,
        hbi_target_cov=0.6,
        hbi_covariance_scaling=0.02,
        phase3b_pop_size=50000,
        phase3b_max_gen=-1,
        phase3b_target_cov=0.6,
        phase3b_covariance_scaling=0.02,
        out="out_hierarchical",
        debug=0,
        dump=False,
        description="Default inference configuration",
    )
