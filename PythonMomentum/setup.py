from distutils.core import setup
from Cython.Build import cythonize

setup(
  name = 'Test Cython',
  ext_modules = cythonize("PythonMomentum.py"),
)
