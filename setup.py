from setuptools import find_packages
from setuptools import setup

setup(
    name="tgm",
    version="0.0.0",
    description="Library implementing a temporal Gaussian mixture.",
    author="Theophile Champion",
    author_email="txc314@student.bham.ac.uk",
    url="https://github.com/ChampiB/temporal-gaussian-mixture/",
    license="Apache 2.0",
    packages=find_packages(),
    scripts=["scripts/run"],
    include_package_data=True,
    install_requires=[
        "gym>=0.21.0",
        "numpy==1.23.4",
        "torchvision==0.15.1",
        "torch==2.0.0",
        "matplotlib==3.6.2",
        "pydot",
        "pytest==7.3.1",
        "customtkinter==0.3"
        "screeninfo"
    ],
    python_requires="~=3.9",
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
    keywords="pytorch, machine learning, active inference"
)
