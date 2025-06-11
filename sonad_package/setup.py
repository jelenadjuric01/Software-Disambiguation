from setuptools import setup, find_packages


# Read README for long description (optional)
try:
    with open("README.md", "r") as fh:
        long_description = fh.read()
except FileNotFoundError:
    long_description = "SOftware NAme Disambiguation: A tool for disambiguating software names and their URLs in scientific literature."

setup(
    name="sonad",  # Replace with your actual package name
    version="0.1.2",
    author="Jelena Duric",
    author_email="djuricjelena611@gmail.com",
    description="SOftware NAme Disambiguation: A tool for disambiguating software names and their URLs in scientific literature.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jelenadjuric01/Software-Disambiguation",  # Optional
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules"
    ],
    license="MIT",
    python_requires="==3.9",
    install_requires=[
    "pandas>=1.3,<2.3",
    "numpy>=1.21,<1.27",
    "cloudpickle>=2.0",
    "scikit-learn>=1.0,<1.4",
    "xgboost>=1.5,<2.2",
    "lightgbm>=3.3,<4.1",
    "sentence-transformers>=2.2,<3.0",
    "textdistance>=4.2",
    "beautifulsoup4>=4.9,<5.0",
    "requests>=2.25,<2.33",
    "SPARQLWrapper>=1.8,<2.1",
    "lxml>=4.9,<6.0",
    "elementpath==4.0.0",
    "somef>=0.9.11"
],

    include_package_data=True,
    package_data={
        "sonad": [
            'model.pkl',
            'CZI/*',
            'json/*'
        ],  # Include your data files
    },
    entry_points={
        'console_scripts': [
            'sonad=sonad.cli:cli',
        ],
    },
)