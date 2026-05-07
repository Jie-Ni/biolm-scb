from setuptools import setup

setup(
    name="scb-diagnostic",
    version="1.0.0",
    description="Predict mature ICL amplification from any HuggingFace Transformer",
    author="Ni Jie",
    author_email="jie.ni@uibk.ac.at",
    url="https://github.com/Jie-Ni/biolm-scb",
    py_modules=["scb_diagnostic"],
    install_requires=[
        "torch>=2.0",
        "transformers>=4.30",
        "numpy>=1.20",
    ],
    entry_points={
        "console_scripts": ["scb-diagnostic=scb_diagnostic:main"],
    },
    license="MIT",
)
