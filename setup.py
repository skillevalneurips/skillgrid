from setuptools import setup, find_packages

setup(
    name="skilleval-bench",
    version="0.1.0",
    description="Benchmarking Skill Composition of Agent Tools",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "pyyaml>=6.0",
        "numpy>=1.24",
        "packaging>=24.0",
    ],
    extras_require={
        "openai": ["openai>=1.0"],
        "anthropic": ["anthropic>=0.25"],
        "google": ["google-generativeai>=0.5"],
        "datasets": ["datasets>=2.14"],
        "viz": ["matplotlib>=3.7", "pandas>=2.0", "streamlit>=1.30"],
        "hf": [
            "transformers>=4.46,<5",
            "trl>=0.10,<1",
            "accelerate>=0.34",
            "peft>=0.12",
        ],
        "all": [
            "openai>=1.0",
            "anthropic>=0.25",
            "google-generativeai>=0.5",
            "datasets>=2.14",
            "matplotlib>=3.7",
            "pandas>=2.0",
            "streamlit>=1.30",
            "transformers>=4.46,<5",
            "trl>=0.10,<1",
            "accelerate>=0.34",
            "peft>=0.12",
        ],
    },
)
