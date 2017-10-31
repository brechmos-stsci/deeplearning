from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from skimage import data, exposure
import numpy as np
import scipy.misc
from PIL import Image
import requests
from io import BytesIO
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from collections import Counter
import matplotlib.pyplot as plt

model = ResNet50(weights='imagenet')

# ---------------------------------------------------------------------
# Load and pre-processing
# ---------------------------------------------------------------------

## Load in the data
carina_orig = np.load('/Users/crjones/Documents/Science/HargisDDRF/carina_224.npy')
carina = carina_orig[:,:,:,:500]
print(carina.shape)

# Display the data
plt.imshow(carina[:,:,:,300])
plt.colorbar()
plt.show()

# Fiddle with the data
#for ii in range(carina.shape[3]):
#    carina[:,:,:,ii] = histeq(carina_orig[:,:,:,ii])


# Display the data again
plt.imshow(carina[:,:,:,20])
plt.colorbar()
plt.show()

# ---------------------------------------------------------------------
# Now processing
# ---------------------------------------------------------------------

# Quick test on one dataset
x = carina[0:224,0:224,:,1]
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

print(x.shape)
preds = model.predict(x)
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
predictions = decode_predictions(preds, top=50)[0]
print('\nPredicted:')
for x in predictions:
    print(x)


# Calculate the predicitons for all data
labels = []
allpredictions = {}
N = carina.shape[3]

for ii in range(N):
    x = carina[0:224,0:224,:,ii]
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    # decode the results into a list of tuples (class, description, probability)
    # (one such list for each sample in the batch)
    predictions = decode_predictions(preds, top=10)[0]
    
    allpredictions[ii] = []
    for tt in predictions:         
        allpredictions[ii].append([tt[1], tt[2]])

# Add in the transpose version
for ii in range(N):
    x = carina[0:224,0:224,:,ii]
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    y = x
    y[0,:,:,0] = x[0,:,:,0].T
    y[0,:,:,1] = x[0,:,:,1].T
    y[0,:,:,2] = x[0,:,:,2].T
    preds = model.predict(y)
    predictions = decode_predictions(preds, top=50)[0]    
    allpredictions[N+ii] = []
    for tt in predictions:         
        allpredictions[N+ii].append([tt[1], tt[2]])

# Add in the reversed versions
for ii in range(N):
    x = carina[0:224,0:224,:,ii]
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    y = x

    y[0,:,:,0] = x[0,::-1,:,0].T
    y[0,:,:,1] = x[0,::-1,:,1].T
    y[0,:,:,2] = x[0,::-1,:,2].T
    preds = model.predict(y)
    predictions = decode_predictions(preds, top=50)[0]    
    allpredictions[2*N+ii] = []
    for tt in predictions:         
        allpredictions[2*N+ii].append([tt[1], tt[2]])

# Add in the reversed versions
for ii in range(N):
    y = x
    y[0,:,:,0] = x[0,:,::-1,0].T
    y[0,:,:,1] = x[0,:,::-1,1].T
    y[0,:,:,2] = x[0,:,::-1,2].T
    preds = model.predict(y)
    predictions = decode_predictions(preds, top=50)[0]    
    allpredictions[3*N+ii] = []
    for tt in predictions:         
        allpredictions[3*N+ii].append([tt[1], tt[2]])


# In[12]:


# Reformat the all information
labels = list(set([x[0] for k, y in allpredictions.items() for x in y ]))
X = np.zeros((len(allpredictions), len(labels)))
for ii in range(len(allpredictions)):
    for jj in allpredictions[ii]:
        X[ii,labels.index(jj[0])] = jj[1]
plt.imshow(X)
print('Number of unqiue labels {}'.format(len(labels)))


# In[13]:



#Y = TSNE(n_components=2).fit_transform(X)
Y = PCA(n_components=2).fit_transform(X)

# In[14]:


plt.figure(1)
plt.clf()
plt.plot(Y[:,0], Y[:,1], '.')


# set min point to 0 and scale
Y_min = np.min(Y, axis=0)
Y_max = np.max(Y, axis=0)

# create embedding image
S = 2000  # size of full embedding image
G = np.zeros((S, S, 3), dtype=np.uint8)
s = 50  # size of single image


for ii in range(Y.shape[0]):

    # set location
    x,y = Y[ii]
    
    a = (x - Y_min[0]) / (Y_max[0] - Y_min[0]) * S
    b = (y - Y_min[1]) / (Y_max[1] - Y_min[1]) * S
    a,b = int(a), int(b)
    
    I = scipy.misc.imresize(carina_orig[:,:,:,ii%N], (s, s, 3))
    
    if a>0 and b >0 and a+s < 2000 and b+s < 2000:
        G[b:b+s, a:a+s,:] = I

plt.figure(2)
plt.clf()
plt.imshow(G)
plt.show()


# In[ ]:




