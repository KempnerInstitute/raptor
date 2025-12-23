from pathlib import Path

from setuptools import find_packages, setup


def read_version() -> str:
    version_file = Path(__file__).parent / "raptor" / "__init__.py"
    for line in version_file.read_text().splitlines():
        if line.startswith("__version__"):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError("Unable to find __version__ in raptor/__init__.py")


setup(
    name="raptor",
    version=read_version(),
    description="Block-Recurrent Dynamics in ViTs (Raptor)",
    packages=find_packages(),
    python_requires=">=3.9",
)
