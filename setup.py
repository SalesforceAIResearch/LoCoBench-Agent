#!/usr/bin/env python3

from setuptools import setup, find_packages

setup(
    name="locobench",
    version="0.1.0",
    description="LoCoBench: A Novel Benchmark for Evaluating Long-Context Language Models in Software Development Tasks",
    author="LoCoBench Team",
    author_email="locobench@example.com",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        # Core data processing
        "datasets>=2.14.0",
        "huggingface_hub>=0.17.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        
        # Code analysis tools
        "tree-sitter>=0.20.0",
        "tree-sitter-python>=0.20.0",
        "tree-sitter-javascript>=0.20.0",
        "tree-sitter-java>=0.20.0",
        "tree-sitter-cpp>=0.20.0",
        "tree-sitter-go>=0.20.0",
        "tree-sitter-typescript>=0.20.0",
        
        # Static analysis and metrics
        "radon>=6.0.0",  # Complexity analysis
        "lizard>=1.17.0",  # Code complexity metrics
        "gitpython>=3.1.0",  # Git repository analysis
        
        # LLM APIs and evaluation
        "openai>=1.0.0",
        "anthropic>=0.7.0",
        "google-generativeai>=0.3.0",
        "lighteval>=0.3.0",
        
        # Utilities
        "tqdm>=4.65.0",
        "click>=8.1.0",
        "pyyaml>=6.0",
        "jsonlines>=3.1.0",
        "rich>=13.0.0",
        "pathspec>=0.11.0",
        
        # Testing and validation
        "pytest>=7.4.0",
        "pytest-asyncio>=0.21.0",
        "black>=23.0.0",
        "flake8>=6.0.0",
        
        # Async processing
        "aiohttp>=3.8.0",
        "asyncio-throttle>=1.0.0",
    ],
    extras_require={
        "dev": [
            "jupyter>=1.0.0",
            "ipython>=8.0.0",
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
            "plotly>=5.15.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "locobench=locobench.cli:main",
            "lce-generate=locobench.generation.cli:main",
            "lce-evaluate=locobench.evaluation.cli:main",
        ]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Quality Assurance",
        "Topic :: Software Development :: Testing",
    ],
) 