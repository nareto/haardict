import twoDdict
import sys

ocdict = twoDdict.ocdict(filepath=sys.argv[1])
npatches = 1
sparsity = 10
plot = True

np.random.seed(234234)
height,width = ocdict.shape
patches = [twoDdict.Patch(np.random.uniform(size=ocdict.shape)) for x in range(npatches)]
height *= len(patches)
img = np.vstack([x.matrix for x in patches])
#patch = Patch(np.hstack([x.array for x in patches]))
outpatches = []
coeficients = []
for patch in patches:
    coefs,mean = ocdict.sparse_code(patch,sparsity)
    outpatch = ocdict.decode(coefs,mean)
    outpatches.append(outpatch)
    coeficients.append(coefs)
result = twoDdict.assemble_patches(outpatches,(height,width))
#fig,(ax1,ax2) = plt.subplots(1,2)
#print('Coeffs1: %d\nCoeffs2: %d\nError1: %f\nError2: %f\nGlobal Error: %f' %
#      (len(coeficients[0].nonzero()[0]),len(coeficients[1].nonzero()[0]),np.linalg.norm(patches[0].matrix - outpatches[0]), np.linalg.norm(patches[1].matrix - outpatches[1]), np.linalg.norm(img-result)))
print('Coeffs1: %d\nError1: %f' %
      (len(coeficients[0].nonzero()[0]),np.linalg.norm(img-result)))
if plot:
    #fig,(ax1,ax2,ax3) = plt.subplots(1,3)
    fig,(ax1,ax2) = plt.subplots(1,2)
    ax1.imshow(img, cmap=plt.cm.gray,interpolation='none')
    ax2.imshow(result, cmap=plt.cm.gray,interpolation='none')
    #ax2.imshow(outpatches[0], cmap=plt.cm.gray,interpolation='none')
    #ax3.imshow(outpatches[1], cmap=plt.cm.gray,interpolation='none')
    ax1.set_title = 'original'
    ax2.set_title = 'reconstructed'
    fig.show()

