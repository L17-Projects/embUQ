"""Simulation-generation helpers used by legacy workflow wrappers."""

from .emb_generation import (
    EmbGenerationResult,
    ParameterSweep,
    build_emb_hpc_sbatch_text,
    generate_emb_simulation,
    generate_parameter_payloads,
    parse_parameter_sweeps,
    write_emb_generation_commands,
)

__all__ = [
    "EmbGenerationResult",
    "ParameterSweep",
    "build_emb_hpc_sbatch_text",
    "generate_emb_simulation",
    "generate_parameter_payloads",
    "parse_parameter_sweeps",
    "write_emb_generation_commands",
]
