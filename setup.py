from setuptools import setup, find_packages

with open("README.md", "r") as f:
    readme = f.read()

setup(
    name="vsgan",
    version="1.0.7",
    author="PRAGMA",
    author_email="pragma.exe@gmail.com",
    description="VapourSynth GAN Implementation using RRDBNet, based on ESRGAN's implementation",
    license="MIT",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://gitlab.com/imPRAGMA/VSGAN",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "torch",
        "vapoursynth"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
