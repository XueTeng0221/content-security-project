from setuptools import setup, find_packages

setup(
    name="av-deepfake-detector",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "torchaudio>=2.0.0",
    ],
)
