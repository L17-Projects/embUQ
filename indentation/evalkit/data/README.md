# Indentation Reference Data

Place raw CSVs here. The pipeline auto-converts to DPD `.dat` on demand.

Expected DPD output:
- `indentation_data_<diam>um.dat` with `Force  Displacement` (DPD units).

Manual conversion (run via SLURM):
`indentation/evalkit/convert_reference_data.py --input ... --diameter 3.2`
