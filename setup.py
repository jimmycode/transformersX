import shutil
from pathlib import Path

from setuptools import find_packages, setup

# Remove stale transformersX.egg-info directory to avoid https://github.com/pypa/pip/issues/5466
stale_egg_info = Path(__file__).parent / "transformersX.egg-info"
if stale_egg_info.exists():
    print(
        (
            "Warning: {} exists.\n\n"
            "This directory is automatically generated by Python's packaging tools.\n"
            "I will remove it now so that this package can be installed in editable mode.\n\n"
            "See https://github.com/pypa/pip/issues/5466 for details.\n"
        ).format(stale_egg_info)
    )
    shutil.rmtree(stale_egg_info)

extras = {}

setup(
    name="transformersX",
    version="0.0.1",
    author="Yuxiang Wu",
    author_email="topcoderjimmy@gmail.com",
    description="Transformers made perfect!",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="transformers patch",
    license="Apache",
    url="https://github.com/jimmycode/transformersX",
    package_dir={"": "src"},
    packages=find_packages("src"),
    install_requires=[
        "transformers",
    ],
    extras_require=extras,
    python_requires=">=3.5.0",
)
