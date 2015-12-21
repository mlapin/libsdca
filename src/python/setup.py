from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension

import numpy as np

# Modifiy this if BLAS and LAPACK libraries are not in /usr/lib.
BLAS_LIB_DIR = '/usr/lib'
BLAS_LIB = ['blas']
LAPACK_LIB = ['lapack']

BLAS_COMPILE_ARGS = ['-DBLAS_DEFAULT_LOCAL_HEADER']
BASE_COMPILE_ARGS = ['-std=c++11', '-lstdc++']
BASE_LINK_ARGS = ['-std=c++11', '-lstdc++']

INCLUDE_DIRS = [np.get_include(), '../']

extension = Extension(name = 'libsdca',
                 language = 'c++',
                 libraries = LAPACK_LIB + BLAS_LIB,
                 include_dirs = INCLUDE_DIRS,
                 library_dirs = [ BLAS_LIB_DIR ],
                 extra_compile_args = BASE_COMPILE_ARGS + BLAS_COMPILE_ARGS,
                 extra_link_args = BASE_LINK_ARGS,
                 sources = ['libsdca.pyx'])

setup(name = 'libsdca',
      description = 'Python wrappers for libsdca.',
      version = '0.0.1',
      long_description = '''
      Python wrappers for libsdca.
      libsdca is a library for multiclass classification
      based on stochastic dual coordinate ascent (SDCA).
      ''',
      author = '121onto',
      author_email = '121onto@gmail.com',
      ext_package = 'libsdca',
      ext_modules = cythonize(extension),
      packages = 'libsdca',
      classifiers = [
          'Development Status :: 2 - Pre-Alpha',
          'Intended Audience :: Science/Research',
          'Operating System :: OS Independent',
          'Programming Language :: C++',
          'Programming Language :: Python',
          'Programming Language :: Python :: 3',
          'Topic :: Scientific/Engineering',
      ],
)
