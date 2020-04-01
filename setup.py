import setuptools

#read in readme
with open("README.md", "r") as readme:
    detailed_description = readme.read()

#prepare setup
setuptools.setup(
    name="MicroPy",
    version="0.1.3",
    author="Rene Lachmann",
    author_email="herr.rene.richter@gmail.de",
    description="A Toolbox for image analysis in Python that was necessary to be build within and for my PhD.",
    long_description=detailed_description,
    long_description_content_type="text/markdown",
    
    #install_requires=['NanoImagingPack'],
    #dependency_links=['https://github.com/blink1073/tifffile','https://test.pypi.org/simple/NanoImagingPack'],
    #url="https://test.pypi.org/simple/NanoImagingPack", 
    #include_package_data=True,
    #package_dir = {'':'NanoImagingPack'},
    #package_data={'MicroPy'}, #'MicroPy':['IMAGES/*']
    #packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Operating System :: OS Independent",
    ],
)