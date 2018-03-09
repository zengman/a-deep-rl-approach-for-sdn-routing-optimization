import numpy as np
t = np.full(5,10,dtype=float)
v = np.asarray([3,6,11,15,7])
for i in range(len(t)):
    x,y = t[i],v[i]
    if y>x:
        d = 1-(y-x)/x
        v = v*d
        print(v)


