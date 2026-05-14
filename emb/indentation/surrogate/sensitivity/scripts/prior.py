from meso_uq.sensitivity import UniformPrior, VarConfig

ind_variables = [
    VarConfig(name="Yt", prior=UniformPrior(1.0e5, 1.0e9)),
    VarConfig(name="kb", prior=UniformPrior(400.0, 7.0e4)),
    VarConfig(name="b1", prior=UniformPrior(0.0, 3.0)),
    VarConfig(name="b2", prior=UniformPrior(0.0, 10.0)),
    VarConfig(name="a3", prior=UniformPrior(-2.5, 3.0)),
    VarConfig(name="a4", prior=UniformPrior(0.0, 4.0)),
]
