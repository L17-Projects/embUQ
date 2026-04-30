from __future__ import annotations

from ..registry import ExperimentSpec
from .controls import EXPERIMENT_CONTROLS
from .observables import (
    BUCKLING_RESPONSE,
    EIGENMODE_SPECTRUM,
    EXTENSION_CURVE,
    SHEAR_FLOW_RESPONSE,
    TORSION_RESPONSE,
)


GV_EXPERIMENTS = (
    ExperimentSpec(
        structure="gv",
        name="stretching",
        description="Axial stretching sweep imported from the Mirheo staging scripts.",
        controls=EXPERIMENT_CONTROLS["stretching"],
        observables=(EXTENSION_CURVE,),
        notes=("Staging sweep uses tot_force with a fixed bpress lane.",),
    ),
    ExperimentSpec(
        structure="gv",
        name="buckling",
        description="Buckling lane driven by buckling control and background pressure.",
        controls=EXPERIMENT_CONTROLS["buckling"],
        observables=(BUCKLING_RESPONSE,),
        notes=("The imported run script sweeps buck with a fixed negative bpress reference.",),
    ),
    ExperimentSpec(
        structure="gv",
        name="torsion",
        description="Torsional deformation lane using end-cap twist control.",
        controls=EXPERIMENT_CONTROLS["torsion"],
        observables=(TORSION_RESPONSE,),
        notes=("The imported equilibration script hard-codes a negative pressure lane.",),
    ),
    ExperimentSpec(
        structure="gv",
        name="eigenmodes",
        description="Eigenmode or resonance characterization under a pressure-conditioned geometry.",
        controls=EXPERIMENT_CONTROLS["eigenmodes"],
        observables=(EIGENMODE_SPECTRUM,),
    ),
    ExperimentSpec(
        structure="gv",
        name="shear_flow",
        description="Experimental shear-flow lane imported from the OBMD staging path.",
        controls=EXPERIMENT_CONTROLS["shear_flow"],
        observables=(SHEAR_FLOW_RESPONSE,),
        experimental=True,
        requires_opt_in=True,
        opt_in_flag="include_experimental=True",
        notes=(
            "Current staging path uses mirheoOBMD rather than the standard Mirheo runtime.",
            "Known bouncer instability keeps this lane behind explicit opt-in.",
        ),
    ),
)
