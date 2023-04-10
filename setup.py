"""Builds the package.
@author: Younggue Bae
"""
import os
import sys
from setuptools import setup, find_packages

here = os.path.dirname(os.path.realpath(__file__))

def read_file(filename: str):
    try:
        lines = []
        with open(filename) as file:
            lines = file.readlines()
            lines = [line.rstrip() for line in lines if not line.startswith('#')]
        return lines
    except:
        return []

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
    install_requires=read_file(f"{here}/requirements.txt"),
    python_requires=">=3.6",
    include_package_data=True,
)
