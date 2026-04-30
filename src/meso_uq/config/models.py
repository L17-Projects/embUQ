from typing import List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator, model_validator


EMB_EXPERIMENTS = {"compression", "indentation"}
SUPPORTED_STRUCTURES = {"emb", "gv"}


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
        return explicit_structure
    if experiment_name in EMB_EXPERIMENTS:
        return "emb"
    return None


class ExperimentSelection(BaseModel):
    structure: Optional[str] = None
    name: str
    geometries: Optional[List[str]] = None
    controls: Optional[List[str]] = None
    diameters: Optional[List[float]] = None
    enabled: bool = True

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


class InferenceConfig(BaseModel):
    emb_diameters: Optional[List[float]] = Field(default=None, min_length=1)
    frac_diam: float = Field(ge=0.1, le=0.5, default=0.2)
    use_surrogate: bool = Field(default=True)
    pop_size: int = Field(ge=100, le=500000, default=50000)
    max_gen: int = Field(default=-1)
    target_cov: float = Field(ge=0.1, le=1.0, default=0.8)
    covariance_scaling: float = Field(ge=0.001, le=1.0, default=0.04)
    phase1_burn_in: Optional[int] = Field(ge=0, default=None)
    prior_Yt: List[float] = Field(min_length=2, max_length=2)
    prior_kb: List[float] = Field(min_length=2, max_length=2)
    prior_b1: List[float] = Field(min_length=2, max_length=2)
    prior_b2: List[float] = Field(min_length=2, max_length=2)
    prior_a3: List[float] = Field(min_length=2, max_length=2)
    prior_a4: List[float] = Field(min_length=2, max_length=2)
    prior_d0: List[float] = Field(min_length=2, max_length=2)
    prior_sigma: List[float] = Field(min_length=2, max_length=2)
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
    hyperprior_mu_kb: Optional[List[float]] = None
    hyperprior_sigma_kb: Optional[List[float]] = None
    hyperprior_mu_b1: Optional[List[float]] = None
    hyperprior_sigma_b1: Optional[List[float]] = None
    hyperprior_mu_b2: Optional[List[float]] = None
    hyperprior_sigma_b2: Optional[List[float]] = None
    hyperprior_mu_a3: Optional[List[float]] = None
    hyperprior_sigma_a3: Optional[List[float]] = None
    hyperprior_mu_a4: Optional[List[float]] = None
    hyperprior_sigma_a4: Optional[List[float]] = None
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
    geometries: Optional[List[str]] = Field(default=None)
    experiments: Optional[List[ExperimentSelection]] = Field(default=None)

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

    @field_validator("prior_Yt", "prior_kb", "prior_b1", "prior_b2", "prior_a3", "prior_a4", "prior_d0", "prior_sigma")
    @classmethod
    def validate_prior_bounds(cls, v):
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
                key = (selection.structure, selection.name)
                if key in seen:
                    raise ValueError(
                        f"Duplicate experiment selection for structure '{selection.structure}' and experiment '{selection.name}'"
                    )
                seen.add(key)
        return self

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
