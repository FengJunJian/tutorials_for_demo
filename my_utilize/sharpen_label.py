import numpy as np

def softmax(z):
    t = np.exp(z)
    a = np.exp(z) / np.sum(t, axis=-1)
    return a

def sharpen(x,T):
    return np.power(x,1.0/T)/np.power(x,1.0/T).sum()

a0=np.random.random(5)
#a1=softmax(a0)
a2=sharpen(a0,0.5)
print(a2)
