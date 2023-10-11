from setuptools import setup
from setuptools import find_packages

long_description= """
Text analysis tools for the Computers and the Humanities Course
"""

required = [
    "pandas",
    "statsmodels",
    "seaborn",
    "matplotlib",
    "sklearn"
]

setup(
    name="tp",
    version="0.0.1",
    description="Text tools",
    long_description=long_description,
    author="Miguel Escobar Varela",
    install_requires=required,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    packages=find_packages(),
)
