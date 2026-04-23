from typing import List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator, model_validator


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
    emb_diameters: List[float] = Field(min_length=1)
    frac_diam: float = Field(ge=0.1, le=0.5, default=0.2)
    use_surrogate: bool = Field(default=True)
    pop_size: int = Field(ge=100, le=500000, default=50000)
    max_gen: int = Field(default=-1)
    target_cov: float = Field(ge=0.1, le=1.0, default=0.8)
    covariance_scaling: float = Field(ge=0.001, le=1.0, default=0.04)
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
    experiment: Optional[str] = Field(default=None)
    data_dir: Optional[str] = Field(default=None)
    data_prefix: Optional[str] = Field(default=None)
    data_files: Optional[dict] = Field(default=None)
    surrogate_dir: Optional[str] = Field(default=None)
    surrogate: Optional[dict] = Field(default=None)
    experiments: Optional[List[dict]] = Field(default=None)

    @field_validator("emb_diameters")
    @classmethod
    def validate_diameters(cls, v):
        if any(d <= 0 for d in v):
            raise ValueError("All diameters must be positive")
        return sorted(v)

    @field_validator("prior_Yt", "prior_kb", "prior_b1", "prior_b2", "prior_a3", "prior_a4", "prior_d0", "prior_sigma")
    @classmethod
    def validate_prior_bounds(cls, v):
        if len(v) != 2:
            raise ValueError("Prior bounds must have exactly 2 elements [min, max]")
        if v[0] >= v[1]:
            raise ValueError(f"Prior min ({v[0]}) must be less than max ({v[1]})")
        return v

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
