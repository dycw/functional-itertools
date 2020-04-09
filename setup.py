from __future__ import annotations

from setuptools import find_packages
from setuptools import setup

from versioneer import get_cmdclass
from versioneer import get_version


setup(
    # metadata
    name="functional_itertools",
    version=get_version(),
    author="Bao Wei",
    author_email="baowei.ur521@gmail.com",
    # options
    zip_safe=False,
    install_requires=["attrs >= 19.3", "more-itertools >= 8.2"],
    python_requires=">=3.7",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    options={"bdist_wheel": {"universal": "1"}},
    # versioneer
    cmdclass=get_cmdclass(),
)
