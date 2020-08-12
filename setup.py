import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="uniker",
    version="0.0.1",
    author="Vivian Cheng",
    author_email="viviancheng1993@gmail.com",
    description="UniKER - combine logical rule and KG embedding",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/CREATE-knowledge-planning/UniKER",
    packages=setuptools.find_packages(),
    python_requires='>=3.7',
)