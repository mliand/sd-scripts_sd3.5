from setuptools import find_packages, setup


setup(
    name="library",
    packages=find_packages(),
    install_requires=[
        "pydantic>=2,<3",
        "pydantic-settings>=2,<3",
        "unfoldNd",
        "openai-whisper",
    ],
)
