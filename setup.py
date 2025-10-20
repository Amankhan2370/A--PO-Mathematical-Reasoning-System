import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="math_equivalence",
    version="1.0.0",
    author="Dan Hendrycks",
    description="A utility for determining whether 2 answers for a problem in the MATH dataset are equivalent.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hendrycks/math",
    license="MIT",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    package_dir={"": "modeling"},
    py_modules=["math_equivalence"],
    python_requires=">=3.7",
    install_requires=[
        # Core math_equivalence module has no external dependencies
        # Add dependencies here if needed
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov",
        ],
        "full": [
            "torch",
            "transformers",
            "numpy",
            "tqdm",
            "datasets",
        ],
    },
)
