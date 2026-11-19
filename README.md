# Borehole interface diagnostics
Author-approved public-data research snapshot, 19 November 2026. It supports two manuscripts in preparation; neither revised manuscript has been submitted.

This package contains the pinned CC BY 4.0 Netherlands dataset, explicit source policies, public-data targets/splits, saved predictions, and the numerical source snapshots used by the two related manuscripts. It contains no Zhengzhou workbook, coordinates, row-level observations or predictions; no private report or manuscript is included.

## Verify without fitting
Install the versions in requirements.txt into an isolated environment, then:
```bash
python run_public.py verify
python -m unittest discover -s GeoBench/research_20261118/code -p 'test_*.py'
```
The verification recomputes Netherlands summaries from saved predictions.

## Refit a documented public-data fold
```bash
python run_public.py rerun-fold --fold 0 --output outputs/baseline-fold0
python run_public.py constraints --fold 0 --output outputs/constraints-fold0
```
The output directory must be new. The driver checks new predictions against the archived reference, including support. Use folds 0 through 4 for all regions of the public-data study. The private Zhengzhou case is not reproducible from this package.

The archived module main functions retain original workspace assumptions and should not be used as portable commands. Use run_public.py. Data adapters and pure numerical functions can also be inspected directly. The original audit_source module is included as an import dependency; its private-data entry point has no source workbook in this package and is not invoked by the public driver.

## Scientific scope
Raw archive: 1440 boreholes. Grouped eligible task: 3401 internal contacts from 1325 holes, with 3345 outer supported targets. These are interpreted labels, not new field truth. Conditional interface predictions do not establish geological presence.
Saved exploratory versions B–F retain adverse findings. C's deployed-batch interpretation uses decision_metrics_within_fold.csv, not its historical pooled ranking. D's intervals are empirical cross-fitted-residual intervals with distinct marginal/simultaneous targets, without arbitrary-spatial-dependence guarantees.
The Technical Note uses standard weighted projection and reports its applicability boundary; no new optimization algorithm is claimed.

## Licensing and release status
Data: CC BY 4.0, original authors and DOI in DATA_CITATION.md. This grant does not cover the authors' new code.
Author software and associated documentation: MIT, see LICENSE. The input data licence is separate; MIT does not relicense third-party data.
The author approved this release scope and MIT code licence on 19 November 2026. MANIFEST.json lists the snapshot files; SCOPE_AUDIT.json records the content check.


## Full curves
Run `python GeoBench/research_20261118/code/full_curves.py` to recompute the Netherlands full error ECDF, 50%–99% nominal calibration grid and 1%–100% within-fold retention curves from saved outputs. No models are refitted. Any infinite interval is a lack of finite calibration support, not a successful guarantee.
