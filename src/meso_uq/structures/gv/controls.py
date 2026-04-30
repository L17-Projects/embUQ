from __future__ import annotations

from ..registry import ControlSpec


BPress = ControlSpec("bpress", "Background pressure applied to the vesicle.", "pressure")
PTan = ControlSpec("ptan", "Tangential forcing coefficient used by OBMD coupling.", "dimensionless")
Buck = ControlSpec("buck", "Buckling sweep control used in staged compression setup.", "dimensionless")
Theta = ControlSpec("theta", "End-cap twist angle applied during torsion.", "radians")
TotForce = ControlSpec("tot_force", "Total stretching load distributed over tip particles.", "force")
AFSI = ControlSpec("afsi", "Fluid-structure interaction conservative-force parameter.", "force")


EXPERIMENT_CONTROLS = {
    "stretching": (TotForce, BPress),
    "buckling": (Buck, BPress),
    "torsion": (Theta,),
    "eigenmodes": (BPress,),
    "shear_flow": (PTan, AFSI, BPress),
}


ALL_GV_CONTROLS = (BPress, PTan, Buck, Theta, TotForce, AFSI)
