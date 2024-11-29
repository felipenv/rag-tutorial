from setuptools import setup, find_packages

setup(
    name="rag tutorial",  # Replace with your project name
    version="0.1.0",
    description="A simple rag implementation",
    author="Name",
    author_email="email@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=[
        # List your dependencies here
    ],
    python_requires=">=3.10",
)
