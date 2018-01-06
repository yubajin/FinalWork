
import numpy as np

def batch_generator(X,y,batch_size=64,shuffle=False,random_seed=None):
    idx = np.arange(y.shape[0])
    if shuffle:
        rng = np.random.RandomState(random_seed)
        rng.shuffle(idx)
        X = X[idx]
        y = y[idx]

    for i in range(0,X.shape[0],batch_size):
        yield (X[i:i+batch_size,:],y[i:i+batch_size])