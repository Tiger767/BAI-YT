import numpy as np

# list to numpy ndarray
a = [1, 2, 3, 4, 5, 6, 7, 8, 9]
print(a)
x = np.array(a)
print(x)
y = np.array([9, 8, 7, 6, 5, 4, 3, 2, 1])
print(y)

# Math
z = x * y + y + x**2 - y / 2
print(z)
z += 5.500000003
print(z)
print(np.round(z, 3))
mean = z.mean()
std = z.std()
print(mean, std)
z_norm = (z - mean) / std
print(z_norm)

# Index
x = np.array([[1,2,3], [3,2,1]])
print(x.shape, len(x))
x[0] += 1
print(x)
x[:, 1] *= 5 

# Conditions
z = x * 1.0
print(np.equal(x, z))
print(np.greater(x, z))
print(np.less(x, z))

# Random
x = np.random.randint(0, 100, size=(5, 5))
print(x)
x = np.clip(x, 10, 90)
print(x)
x = np.random.uniform(size=5)
print(x)
x = np.random.normal(size=50)
print(x)