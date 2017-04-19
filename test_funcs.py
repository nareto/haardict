import gc
#from twoDdict import *
import twoDdict
import os
import skimage.io

def learn_dict():
    images_paths = []
    #for i in range(1):
    #    images.append(np.random.uniform(size=(200,300)))
    #starting_path = '/Users/renato/ownCloud/phd/bonin_et_al_pics_2009'
    starting_path = '/Users/renato/ownCloud/phd/360_Colour_Items_Moreno-Martinez_Montoro/Nature'
    for dirpath,dirnames,filenames in os.walk(starting_path):
        for f in filenames:
            if f[-3:].upper() in  ['JPG','GIF','PNG']:
                images_paths.append(dirpath+'/'+f)
    #paths = images_paths[:2]
    #paths = ['/Users/renato/ownCloud/phd/360_Colour_Items_Moreno-Martinez_Montoro/Nature/cliff.jpg','/Users/renato/ownCloud/phd/360_Colour_Items_Moreno-Martinez_Montoro/Nature/island.jpg']
    paths = ['/Users/renato/ownCloud/phd/360_Colour_Items_Moreno-Martinez_Montoro/Nature/cliff.jpg']
    images = []
    for f in paths:
        images.append(skimage.io.imread(f,as_grey=True))
    print('Learning from images: %s' % paths)

    patches = []
    for i in images:
        patches += [twoDdict.Patch(p) for p in twoDdict.extract_patches(i)]

    twodpca = twoDdict.twodpca(patches,l=1,r=1)
    #ipdb.set_trace()
    twodpca.compute_simple_bilateral_2dpca()

    for p in patches:
        p.compute_feature_matrix(twodpca.U,twodpca.V)

    ocd = twoDdict.ocdict(patches)
    ocd.twomeans_cluster()

    return(ocd)

def fast_test_patch_reconstruction(sparsity=40):
    ocdict = ocDict()
    ocdict.load_pickle('/Users/renato/ownCloud/phd/code/2ddict/ocdict-MMM-Nature-2')
    #ocdict.load_pickle('/Users/renato/ownCloud/phd/code/2ddict/ocdict-MMM-Nature-2-nonnormalized')
    ocdict.shape = ocdict.patches[0].shape
    mountain = skimage.io.imread('/Users/renato/ownCloud/phd/360_Colour_Items_Moreno-Martinez_Montoro/Nature/mountain.jpg',as_grey=True)
    mountain_patches = extract_patches(mountain)
    p1 = Patch(mountain_patches[3455])
    return(test_patch_reconstruction(ocdict,p1,sparsity=sparsity))


def test_patch_reconstruction(ocdict,patch=None,sparsity = 40):
    if patch is None:
        patch = Patch(np.random.uniform(size=ocdict.shape))
    coeffs = ocdict.sparse_code(patch,sparsity)
    outpatch = ocdict.decode(coeffs)
    fig,(ax1,ax2) = plt.subplots(1,2)
    ax1.imshow(patch.array, cmap=plt.cm.gray,interpolation='none')
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

def fast_test_reconstruction(sparsity=20,plot=False):
    ocdict = ocDict()
    ocdict.load_pickle('/Users/renato/ownCloud/phd/code/2ddict/ocdict-MMM-Nature-2')
    #ocdict.load_pickle('/Users/renato/ownCloud/phd/code/2ddict/ocdict-MMM-Nature-2-nonnormalized')
    ocdict.shape = ocdict.patches[0].shape
    #imgpath =  '/Users/renato/ownCloud/phd/360_Colour_Items_Moreno-Martinez_Montoro/Living_Things/Marine_creatures/lobster.jpg'
    #imgpath =  '/Users/renato/ownCloud/phd/360_Colour_Items_Moreno-Martinez_Montoro/Nature/mountain.jpg'
    imgpath =  '/Users/renato/ownCloud/phd/code/epwt/img/cameraman256.png'
    #imgpath =  '/Users/renato/ownCloud/phd/code/epwt/img/peppers256.png'
    return(test_reconstruction(ocdict,imgpath,sparsity,plot))
    
def test_reconstruction(ocdict,imgpath,sparsity=20,plot=True):
    psize = (ocdict.height,ocdict.width)
    spars= sparsity

    img = skimage.io.imread(imgpath,as_grey=True)
    patches = extract_patches(img,size=psize)
    outpatches = []
    for p in patches:
        outpatches.append(ocdict.decode(ocdict.sparse_code(Patch(p),spars)))
    out = assemble_patches(outpatches,img.shape)
    outclip = clip(out)
    hpi = HaarPSI(img,out)
    hpi_clip = HaarPSI(img,outclip)
    print('HaarPSI = %f  %f' % (hpi,hpi_clip))
    psnrval = psnr(img,out)
    psnrval_clip = psnr(img,outclip)
    print('PSNR = %f   %f' % (psnrval,psnrval_clip))
    if plot:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)#, sharey=True)
        #ax1.imshow(img[34:80,95:137], cmap=plt.cm.gray,interpolation='none')
        #ax2.imshow(out[34:80,95:137], cmap=plt.cm.gray,interpolation='none')
        ax1.imshow(img, cmap=plt.cm.gray,interpolation='none')
        ax2.imshow(out, cmap=plt.cm.gray,interpolation='none')
        ax3.imshow(outclip, cmap=plt.cm.gray,interpolation='none')
        fig.show()
    return(img,out)
    
def test_denoise(sigma=12):
    sigma_noise = sigma
    sparsity = 20
    ocdict = ocDict()
    ocdict.load_pickle('/Users/renato/ownCloud/phd/code/2ddict/ocdict-MMM-Nature-2')
    ocdict.shape = ocdict.patches[0].shape
    psize = (ocdict.height,ocdict.width)
    #imgpath =  '/Users/renato/ownCloud/phd/360_Colour_Items_Moreno-Martinez_Montoro/Living_Things/Marine_creatures/lobster.jpg
    #imgpath =  '/Users/renato/ownCloud/phd/360_Colour_Items_Moreno-Martinez_Montoro/Nature/mountain.jpg'
    imgpath =  '/Users/renato/ownCloud/phd/code/epwt/img/cameraman256.png'
    img = skimage.io.imread(imgpath,as_grey=True)
    imgn = img + np.random.normal(0,sigma_noise,size=img.shape)
    hpi1 = HaarPSI(255*img,255*imgn)
    print('hpi1 = %f' % hpi1)
    patches = extract_patches(imgn,size=psize)
    outpatches = []
    for p in patches:
        outpatches.append(ocdict.decode(ocdict.sparse_code(Patch(p),sparsity)))
    out = assemble_patches(outpatches,img.shape)
    hpi2 = HaarPSI(255*img,255*out)
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
    imgpath =  '/Users/renato/ownCloud/phd/360_Colour_Items_Moreno-Martinez_Montoro/Living_Things/Marine_creatures/lobster.jpg'
    img = skimage.io.imread(imgpath,as_grey=True)
    patches = extract_patches(img,size=(8,8))
    out = assemble_patches(patches,img.shape)
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.imshow(img, cmap=plt.cm.gray,interpolation='none')
    ax2.imshow(out, cmap=plt.cm.gray,interpolation='none')
    ax1.set_title = 'original'
    ax2.set_title = 're assembled'
    fig.show()
