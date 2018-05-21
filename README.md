# Introduction
This repository contains python code used for testing tree-based dictionary learning methods

# Dependencies
The following python packages are used:

	numpy
	scipy
	pywt
	skimage


# External Files
ksvdapprox, Haarpsi

# Optional Dependencies
pyemd, 

# Example Usage

Import the module

	import twoDdict as td

You could then use the wrap-up functions `learn_dict` and `reconstruct`:
	
	oc_dict = td.learn_dict(['pathtoimg.png'],method='2ddict',cluster_epsilon=1.5e1)
	rec_img,coefs = td.reconstruct(oc_dict,'pathtoimg.png',sparsity = 2)
	
Or do everything manually: load an image and extract 8x8 patches from it
	
	npimg = td.np_or_img_to_array('pathtoimg.png')
	patches = td.extract_patches(npimg)
	
Cluster the patches hiearchically using 2-means:

	twomclust = td.twomeans_clustering(patches,1e1)
	twomclust.comput()
	
Use the hiearchical clustering to learn a haar-based dictionary:

	hdict = td.hierarchical_dict(twomclust,patches,'haar')
		
or use K-SVD to learn the dictionary:

	ksvddict = td.ksvd_dict(patches,dictsize=165,sparsity=2)
	
Finally reconstruct the image with the learnt dictionary:

	hd_means,hd_coefs = td.hdict.encode_patches(patches)
	hd_rec_patches = td.hdict.reconstruct_patches(hd_means,hd_coefs)
	hd_rec = td.assemble_patches(hd_rec_patches,npimg.shape)
	
	ksvd_means,ksvd_coefs = td.ksvdict.encode_patches(patches)
	ksvd_rec_patches = td.ksvdict.reconstruct_patches(ksvd_means,ksvd_coefs)
	ksvd_rec = td.assemble_patches(ksvd_rec_patches,npimg.shape)
	
		
