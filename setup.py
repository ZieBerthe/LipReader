from setuptools import setup, find_packages

setup(
    name="lipreader",
    version="0.1.0",
    description="A lip reading pipeline for reconstructing words from video",
    author="LipReader Team",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "mediapipe>=0.10.0",
        "scipy>=1.10.0",
        "pillow>=10.0.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ]
    },
)
