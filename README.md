# LIPOTHERM — GDGT-SST calibration

LIPOTHERM calibration framework helps reconstruct sea-surface temperature from isoprenoid GDGT distributions in marine sediments using supervised machine learning. 

Code accompanying:

> Kumar, V., Tiwari, M., Roy, B., 2026. LIPOTHERM: A calibration framework for
> GDGT paleothermometry based on machine learning. *Geochemistry, Geophysics,
> Geosystems*, in revision.

## Run

```bash
python lipotherm.py data/example_downcore.csv
```

The script trains the calibration model (Extra Trees) on `data/training_data.csv` and applies it to any down-core series given on the command line. 

## Input format

A CSV with one row per sample, with the six isoprenoid GDGT fractional abundances. Any
other columns present in the data sheet are carried through to the output file unchanged.

Three naming conventions are accepted:

| Component | Accepted column names |
|---|---|
| GDGT-0 | `fGDGT_0`, `GDGT-0`, `GDGT.0` |
| GDGT-1 | `fGDGT_1`, `GDGT-1`, `GDGT.1` |
| GDGT-2 | `fGDGT_2`, `GDGT-2`, `GDGT.2` |
| GDGT-3 | `fGDGT_3`, `GDGT-3`, `GDGT.3` |
| Crenarchaeol | `fGDGT_cren`, `Cren`, `Crenarchaeol` |
| Crenarchaeol isomer | `fGDGT_cren_prime`, `Cren'`, `Cren.` |


`data/example_downcore.csv` is an example of a valid input file.
