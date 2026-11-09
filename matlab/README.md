# MATLAB and Octave scripts

Build and demo scripts for the MEX gateways in [`../src/matlab/`](../src/matlab)
(full usage is in [docs/MATLAB.md](../docs/MATLAB.md)):

- `make.m`: compiles `matsdca_fit`, `matsdca_loss` and `matsdca_prox` in place against
  MATLAB's `libmwblas`.
- `demo.m`: projects a random matrix onto the simplex with `matsdca_prox` and trains a
  multiclass classifier with `matsdca_fit` on train/validation/test splits.
- `getdata.m`: generates the synthetic 3-class circle dataset from Lapin, Hein and
  Schiele (*CVPR*, 2016) used by `demo.m`.

```matlab
run matlab/make.m
addpath matlab
run matlab/demo.m
```

With GNU Octave (compiles with `mkoctfile` and runs `test_numerics.m` and `demo.m`):

```bash
OCTAVE_HOME=<prefix> tests/matlab/run_octave_tests.sh
```
