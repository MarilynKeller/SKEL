from setuptools import find_packages, setup
from aitviewer import __version__

requirements = ["wheel",
                "-e git+https://github.com/mattloper/chumpy.git#egg=chumpy",
                "smplx"]

setup(
    name="skel",
    description="SKEL model Loader.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    version=1.0,
    author="Marilyn Keller",
    packages=find_packages(),
    include_package_data=True,
    keywords=[
        "motion",
        "machine learning",
        "sequences",
        "smpl",
        "computer graphics",
        "computer vision",
        "3D",
        "meshes",
        "skel",
        "smpl"
    ],
    platforms=["any"],
    python_requires=">=3.7,<3.11",
    install_requires=requirements,
)
