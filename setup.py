from setuptools import setup, find_namespace_packages

setup(
    name="ecojax",
    url="https://github.com/tboulet/EcoJAX", 
    author="Timothé Boulet",
    author_email="timothe.boulet0@gmail.com",
    
    packages=find_namespace_packages(exclude=["tests*", "venv*"]),

    version="1.0",
    license="GNU",
    description="A framework for simulating ecosystems in JAX.",
    long_description=open('README.md').read(),      
    long_description_content_type="text/markdown",  
)