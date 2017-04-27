import gc
#from twoDdict import *
import matplotlib.pyplot as plt
import numpy as np
import twoDdict
import os
import skimage.io

def test_learn_dict(l=2,r=2):
    images_paths = []
    #for i in range(1):
    #    images.append(np.random.uniform(size=(200,300)))
    #starting_path = './bonin_et_al_pics_2009'
    #starting_path = './360_Colour_Items_Moreno-Martinez_Montoro/Nature'
    #for dirpath,dirnames,filenames in os.walk(starting_path):
    #    for f in filenames:
    #        if f[-3:].upper() in  ['JPG','GIF','PNG']:
    #            images_paths.append(dirpath+'/'+f)
    #paths = images_paths[:2]
    #paths = ['./360_Colour_Items_Moreno-Martinez_Montoro/Nature/cliff.jpg','./360_Colour_Items_Moreno-Martinez_Montoro/Nature/island.jpg']
    #paths = ['./360_Colour_Items_Moreno-Martinez_Montoro/Nature/cliff.jpg',
    #         './360_Colour_Items_Moreno-Martinez_Montoro/Nature/cloud.jpg',
    #         './360_Colour_Items_Moreno-Martinez_Montoro/Nature/gold.jpg']
    paths = ['seis0_orig.eps','seis2_orig.eps']
    return(twoDdict.learn_dict(paths,l,r))


def fast_test_patch_reconstruction(sparsity=40):
    ocdict = twoDdict.ocdict()
    ocdict.load_pickle('./ocd-ccg-nonnormalized')
    #ocdict.load_pickle('./ocdict-MMM-Nature-2')
    #ocdict.load_pickle('./ocdict-MMM-Nature-2-nonnormalized')
    ocdict.shape = ocdict.patches[0].matrix.shape
    ocdict.height,ocdict.width = ocdict.shape
    mountain = skimage.io.imread('./360_Colour_Items_Moreno-Martinez_Montoro/Nature/mountain.jpg',as_grey=True)
    mountain_patches = twoDdict.extract_patches(mountain)
    p1 = twoDdict.Patch(mountain_patches[3455])
    return(test_patch_reconstruction(ocdict,p1,sparsity=sparsity))


def test_patch_reconstruction(ocdict,patch=None,sparsity = 40):
    if patch is None:
        patch = twoDdict.Patch(np.random.uniform(size=ocdict.shape))
    coeffs = ocdict.sparse_code(patch,sparsity)
    outpatch = ocdict.decode(coeffs)
    fig,(ax1,ax2) = plt.subplots(1,2)
    ax1.imshow(patch.matrix, cmap=plt.cm.gray,interpolation='none')
    ax2.imshow(outpatch, cmap=plt.cm.gray,interpolation='none')
    ax1.set_title = 'original'
    ax2.set_title = 'reconstructed'
    fig.show()
    return(coeffs)

def test_reconstruction_patches(ocdict,patches=None,sparsity = 10):
    height,width = ocdict.shape
    if patches is None:
        patches = [Patch(np.random.uniform(size=ocdict.shape)),Patch(np.random.uniform(size=ocdict.shape))]
    height *= len(patches)
    img = np.vstack([x.array for x in patches])
    #patch = Patch(np.hstack([x.array for x in patches]))
    outpatches = []
    for patch in patches:
        outpatch = ocdict.decode(ocdict.sparse_code(patch,sparsity))
        outpatches.append(outpatch)
    result = assemble_patches(outpatches,(height,width))
    #fig,(ax1,ax2) = plt.subplots(1,2)
    fig,(ax1,ax2,ax3) = plt.subplots(1,3)
    ax1.imshow(img, cmap=plt.cm.gray,interpolation='none')
    ax2.imshow(result, cmap=plt.cm.gray,interpolation='none')
    #ax2.imshow(outpatches[0], cmap=plt.cm.gray,interpolation='none')
    ax3.imshow(outpatches[1], cmap=plt.cm.gray,interpolation='none')
    ax1.set_title = 'original'
    ax2.set_title = 'reconstructed'
    fig.show()

def fast_test_reconstruction(sparsity=20,clip=True,plot=False):
    ocdict = twoDdict.ocdict()
    #ocdict.load_pickle('./ocd-ccg-nonnormalized')
    ocdict.load_pickle('./ocd-seis0+2-nonnormalized')
    twodpca = twoDdict.twodpca()
    twodpca.load_pickle('./twodpca-ccg')
    #ocdict.load_pickle('./ocdict-MMM-Nature-2')
    #ocdict.load_pickle('./ocdict-MMM-Nature-2-nonnormalized')
    #ocdict.shape = ocdict.patches[0].shape
    ocdict.shape = ocdict.patches[0].matrix.shape
    ocdict.height,ocdict.width = ocdict.shape
    #ocdict.matrix_is_normalized = False
    #imgpath =  './360_Colour_Items_Moreno-Martinez_Montoro/Living_Things/Marine_creatures/lobster.jpg'
    #imgpath =  './360_Colour_Items_Moreno-Martinez_Montoro/Nature/mountain.jpg'
    imgpath =  './code/epwt/img/cameraman256.png'
    #imgpath =  './code/epwt/img/peppers256.png'
    return(test_reconstruction(ocdict,imgpath,sparsity,clip,plot))
    
def test_reconstruction(ocdict,imgpath,sparsity=20,clip=True,plot=True):
    psize = (ocdict.height,ocdict.width)
    spars= sparsity

    if imgpath[-3:].upper() in  ['JPG','GIF','PNG','EPS']:
        img = skimage.io.imread(imgpath,as_grey=True)
    elif imgpath[-3:].upper() in  ['NPY']:
        img = np.load(imgpath)
    patches = twoDdict.extract_patches(img,size=psize)
    outpatches = []
    for p in patches:
        outpatches.append(ocdict.decode(ocdict.sparse_code(twoDdict.Patch(p),spars)))
    out = twoDdict.assemble_patches(outpatches,img.shape)
    if clip:
        out = twoDdict.clip(out)
    hpi = twoDdict.HaarPSI(img,out)
    print('HaarPSI = %f ' % hpi)
    psnrval = twoDdict.psnr(img,out)
    print('PSNR = %f  ' % psnrval)
    if plot:
        fig, (ax1, ax2) = plt.subplots(1, 2)#, sharey=True)
        #ax1.imshow(img[34:80,95:137], cmap=plt.cm.gray,interpolation='none')
        #ax2.imshow(out[34:80,95:137], cmap=plt.cm.gray,interpolation='none')
        ax1.imshow(img, cmap=plt.cm.gray,interpolation='none')
        ax2.imshow(out, cmap=plt.cm.gray,interpolation='none')
        #ax3.imshow(outclip, cmap=plt.cm.gray,interpolation='none')
        fig.show()
    return(img,out)
    
def test_denoise(sigma=12):
    sigma_noise = sigma
    sparsity = 10
    ocdict = twoDdict.ocdict()
    ocdict.load_pickle('./ocd-ccg-nonnormalized')
    ocdict.shape = ocdict.patches[0].matrix.shape
    ocdict.height,ocdict.width = ocdict.shape
    psize = (ocdict.height,ocdict.width)
    #imgpath =  './360_Colour_Items_Moreno-Martinez_Montoro/Living_Things/Marine_creatures/lobster.jpg
    #imgpath =  './360_Colour_Items_Moreno-Martinez_Montoro/Nature/mountain.jpg'
    imgpath =  './code/epwt/img/cameraman256.png'
    img = skimage.io.imread(imgpath,as_grey=True)
    imgn = twoDdict.clip(img + np.random.normal(0,sigma_noise,size=img.shape))
    hpi1 = twoDdict.HaarPSI(img,imgn)
    print('hpi1 = %f' % hpi1)
    patches = twoDdict.extract_patches(imgn,size=psize)
    outpatches = []
    for p in patches:
        outpatches.append(ocdict.decode(ocdict.sparse_code(twoDdict.Patch(p),sparsity)))
    out = twoDdict.clip(twoDdict.assemble_patches(outpatches,img.shape))
    hpi2 = twoDdict.HaarPSI(img,out)
    print('hpi2 = %f' % hpi2)
    fig, (ax1, ax2,ax3) = plt.subplots(1, 3)#, sharey=True)
    #ax1.imshow(img[34:80,95:137], cmap=plt.cm.gray,interpolation='none')
    #ax2.imshow(imgn[34:80,95:137], cmap=plt.cm.gray,interpolation='none')
    #ax3.imshow(out[34:80,95:137], cmap=plt.cm.gray,interpolation='none')
    ax1.imshow(img, cmap=plt.cm.gray,interpolation='none')
    ax2.imshow(imgn, cmap=plt.cm.gray,interpolation='none')
    ax3.imshow(out, cmap=plt.cm.gray,interpolation='none')
    ax1.set_title = 'original'
    ax2.set_title = 'noisy'
    ax3.set_title = 'denoised'
    fig.show()
    return(img,imgn,out)

def test_assembling():
    imgpath =  './360_Colour_Items_Moreno-Martinez_Montoro/Living_Things/Marine_creatures/lobster.jpg'
    img = skimage.io.imread(imgpath,as_grey=True)
    patches = extract_patches(img,size=(8,8))
    out = assemble_patches(patches,img.shape)
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.imshow(img, cmap=plt.cm.gray,interpolation='none')
    ax2.imshow(out, cmap=plt.cm.gray,interpolation='none')
    ax1.set_title = 'original'
    ax2.set_title = 're assembled'
    fig.show()
