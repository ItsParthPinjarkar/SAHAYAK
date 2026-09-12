from setuptools import setup, find_packages

setup(
    name="sahayak-silent-zone-detection",
    version="1.0.0",
    author="Rishikesh Singh, Hiya Shaikh",
    author_email="singhrishikesh1@users.noreply.github.com, hiyashaikh16@users.noreply.github.com",
    description="Autonomous Multi-Engine AI Platform for Silent Zone Detection & Disaster Intelligence in India",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
)
