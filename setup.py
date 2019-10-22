import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="vsgan",
    version="1.0.0.post2",
    author="PRAGMA",
    author_email="pragma.exe@gmail.com",
    description="VapourSynth GAN Implementation using RRDBNet, based on ESRGAN's implementation",
    license='MIT',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.com/imPRAGMA/VSGAN",
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy',
        'torch'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)