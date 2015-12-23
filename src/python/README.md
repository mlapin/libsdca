# Python wrappers for libsdca

## Installation

Naviage to `/your/project/dir/libsdca/src/python/`.  From the command line, run:

```bash
python setup.py build_ext --inplace
```

The package should install and you should be able to `import libsdca`.

## Overview

This package exposes two wrappers for libsdca and a python class that's used for returning a eval data from the solvers:

- `py_prox` wraps proximal operators.  It is similar to `libsdca_prox` from the matlab interface.
- `py_solve` wraps solvers for multiclass classification.  It is similar to `libsdca_solve` from the matlab interface.

See `libsdca.pyx` for details and usage.

## Caution

This codebase is under development and has not been thoroughly tested.  Please use at your own risk.  Suggestions, bug reports, and improvements are welcome.
