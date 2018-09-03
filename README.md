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

Import the module

	import haardict as hd

Use the wrap-up functions `learn_dict` and `reconstruct`:
	
	oc_dict,computation_time = hd.learn_dict(['pathtolearnimg1.png','pathtolearnimg2.png'],patch_size=(8,8),method='2ddict',dictisize=85)
	rec_img,coefs,reconstruction_time,noisy_img = hd.reconstruct(oc_dict,'pathtorecimg.png',patch_size=(8,8),sparsity = 2)
	
and compute the HaarPSI of the reconstruction:

	from haarpsi import haar_psi
	hd.haar_psi(hd.np_or_img_to_array('pathtorecimg.png'),rec_img)[0]
	
	
Or do everything manually: load an image and extract 8x8 patches from it
	
	npimg = hd.np_or_img_to_array('pathtoimg.png')
	patches = hd.extract_patches(npimg)
	
Cluster the patches hiearchically using 2-means:

	twomclust = hd.twomeans_clustering(patches,1e1)
	twomclust.comput()
	
Learn the haar-dictionary using 2-means clustering:

	hdict = hd.hiearchical_dict(patches)
	hdict.compute('twomeans',nbranchings=85)
		
or use K-SVD to learn the dictionary:

	ksvddict = hd.ksvd_dict(patches,dictsize=85,sparsity=2)
	ksvddict.compute()
	
Finally reconstruct the image with the learnt dictionary:

	hd_means,hd_coefs = hdict.encode_patches(patches,sparsity=2)
	hd_rec_patches = hdict.reconstruct_patches(hd_coefs,hd_means)
	hd_rec = hd.assemble_patches(hd_rec_patches,npimg.shape)
	
	ksvd_means,ksvd_coefs = ksvdict.encode_patches(patches,sparsity=2)
	ksvd_rec_patches = ksvdict.reconstruct_patches(ksvd_coefs,ksvd_means)
	ksvd_rec = hd.assemble_patches(ksvd_rec_patches,npimg.shape)
	
and compute the HaarPSI values:

	hd.haar_psi(hd.np_or_img_to_array('pathtorecimg.png'),hd_rec)[0]
	hd.haar_psi(hd.np_or_img_to_array('pathtorecimg.png'),ksvd_rec)[0]
		
[1] http://www.cs.technion.ac.il/~ronrubin/software.html

[2] http://www.haarpsi.org/
