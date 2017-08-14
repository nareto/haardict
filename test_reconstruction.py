import numpy as np
import matplotlib.pyplot as plt
import twoDdict
import sys

def main(ocdict,imgpath,sparsity=5,plot=False):

    clip = False
    twodpca = None

    psize = (ocdict.height,ocdict.width)
    spars= sparsity

    if imgpath[-3:].upper() in  ['JPG','GIF','PNG','EPS']:
        img = skimage.io.imread(imgpath,as_grey=True)
    elif imgpath[-3:].upper() in  ['NPY']:
        img = np.load(imgpath)
    patches = twoDdict.extract_patches(img,size=psize)
    outpatches = []
    coefsnonzeros = []
    for p in patches:
        patch = twoDdict.Patch(p)
        coefs,mean = ocdict.sparse_code(patch,spars)
        outpatches.append(ocdict.decode(coefs,mean))
        coefsnonzeros.append(len(coefs.nonzero()[0]))

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

if __name__ == '__main__':
    main(twoDdict.ocdict(filepath=sys.argv[1]),sys.argv[2])

    
