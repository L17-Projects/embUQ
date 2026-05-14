from meso_uq.sensitivity import UniformPrior, VarConfig

comp_variables = [
    VarConfig(name="Yt", prior=UniformPrior(1.0e7, 1.0e8)),
    VarConfig(name="kb", prior=UniformPrior(100.0, 1000.0)),
    VarConfig(name="b1", prior=UniformPrior(0.0, 3.0)),
    VarConfig(name="b2", prior=UniformPrior(0.0, 5.0)),
    VarConfig(name="a3", prior=UniformPrior(-2.5, 1.0)),
    VarConfig(name="a4", prior=UniformPrior(0.0, 2.0)),
]
