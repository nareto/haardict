import twoDdict
import sys

sigma_noise = 12
sparsity = 10
ocdict = twoDdict.ocdict()
ocdict.load_pickle('./ocd-s0s2-826')
ocdict.shape = ocdict.patches[0].matrix.shape
ocdict.height,ocdict.width = ocdict.shape
psize = (ocdict.height,ocdict.width)
imgpath =  sys.argv[1]
if imgpath[-4:] == '.npy':
    img = np.load(imgpath)
else:
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
