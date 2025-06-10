from setuptools import setup, find_packages

# Read requirements from requirements.txt
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

# Read README for long description (optional)
try:
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()
except FileNotFoundError:
    long_description = "SOftware NAme Disambiguation: A tool for disambiguating software names and their URLs in scientific literature."

setup(
    name="sonad",  # Replace with your actual package name
    version="0.1.0",
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
    python_requires="==3.9",
    install_requires=requirements,
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