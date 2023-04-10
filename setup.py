"""Builds the package.
@author: Younggue Bae
"""
from __future__ import print_function
import os
import sys

import pkg_resources
from setuptools import setup, find_packages


if sys.version_info < (2, 7):
    print("Python versions prior to 2.7 are not supported.",
          file=sys.stderr)
    exit(-1)

setup_requires = [
]

dependency_links = [
    # "https://download.pytorch.org/whl/nightly/cu116" # For PyTorch2.0
]

setup(
    name="nextgpt",
    version="0.0.1",
    description="nextGPT",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Younggue Bae",
    # author_email="louiezzang@gmail.com",
    package_dir={"": "src"},
    url="https://github.com/louiezzang/next-gpt",
    keywors=[
        "GPT",
        "ChatGPT",
        "LLaMA",
    ],
    packages=find_packages(where="src"),
    install_requires=[
        str(r)
        for r in pkg_resources.parse_requirements(
            open(os.path.join(os.path.dirname(__file__), "requirements.txt"))
        )
    ],
    include_package_data=True,
    # extras_require={'dev': ['pytest']},
    setup_requires=setup_requires,
    dependency_links=dependency_links,
)
