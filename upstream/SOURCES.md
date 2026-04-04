# Source Pins

Primary code import source:

- repository: `BrieucB/UQ_DPD`
- branch: `vega/gpu-batching`
- pinned SHA at bootstrap planning time: `37f5a7ea58c27d835b236b7f3b7bbe55e37a3d86`

Additional reconciliation source:

- repository: `BrieucB/UQ_DPD`
- branch: `master`
- pinned SHA at bootstrap planning time: `e1f9c557b19db3c68d23d84b768f41fca065dbca`

Korali bootstrap source policy:

- use the `korali/` directory inside `UQ_DPD` during bootstrap 1
- if a standalone Korali repository is later exposed to this connector, it can become the dedicated upstream source
