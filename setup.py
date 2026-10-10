"""
Setup script for SmallLLM
"""

from setuptools import setup, find_packages
import pathlib

# Read the long description from README
here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="smallllm",
    version="0.1.0",
    description="A tiny AI that runs on 2011-era PCs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ciao22king/SmallLLM",
    author="ciao22king",
    author_email="",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
        "Operating System :: OS Independent",
    ],
    keywords="ai, nlp, transformer, language-model, tiny, lightweight",
    package_dir={"": "."},
    packages=find_packages(),
    python_requires=">=3.6",
    install_requires=[
        "numpy>=1.16.0",
        "Flask>=2.0.0",
    ],
    entry_points={
        "console_scripts": [
            "smallllm=smallllm.cli:main",
            "smallllm-api=smallllm.api:run_api",
            "smallllm-gui=gui.main_window:main",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/ciao22king/SmallLLM/issues",
        "Source": "https://github.com/ciao22king/SmallLLM",
    },
)
