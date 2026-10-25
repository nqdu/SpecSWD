"""Tag wheels containing prebuilt CMake modules for the current platform."""

from pathlib import Path
from setuptools import Distribution, setup


class SpecDistribution(Distribution):
    def has_ext_modules(self):
        return any(path.suffix in {".so", ".pyd", ".dylib"}
                   for path in Path("specd/lib").glob("*"))


setup(distclass=SpecDistribution)
