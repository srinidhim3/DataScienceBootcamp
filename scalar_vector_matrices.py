import numpy as np

# scalars 
s = 5

# vectors
v = np.array([5,-2, 4])
print(v)

# matrix
m = np.array([[5,12,6],[-3,0,14]])
print(m)

print(m.shape)
print(v.reshape(1,3))
print(v.reshape(3,1))