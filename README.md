# Introduction
This repository contains python code used for testing the Haar dictionary learning method proposed in my PhD thesis. 

# Dependencies
The following python packages are used:

	numpy
	scipy
	pywt
	skimage


# External Files and Optional Dependencies
To compute the KSVD a working installation of `octave` and `oct2py` are needed, as well as the KSVD-Box [1] package which must be compiled. To compute the HaarPSI values of the figures the `haarpsi.py` file is needed [2]. To use spectral clustering with the EMD (Earth Mover's Distance) the python `pyemd` package is needed.

# Example Usage

For image reconstruction tasks the user will mostly be interested in the Test class. See file `test.py` for example usage.


[1] http://www.cs.technion.ac.il/~ronrubin/software.html

[2] http://www.haarpsi.org/
