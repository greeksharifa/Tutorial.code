import numpy as np

data = np.random.randn(2, 3)
data
data * 10
data + data
data.shape
data.dtype

data1 = [6, 7.5, 8, 0, 1]
arr1 = np.array(data1)
arr1

data2 = [[1,2,3,4], [5,6,7,8]]
arr2 = np.array(data2)
arr2
arr2.ndim
arr2.shape
arr2.dtype

np.zeros(10)
np.zeros((3, 6))
np.empty((2,3,2))
np.arange(15)
np.arange(5,15)
np.array([1,3,5], 'float64').dtype
# np.array / asarray / arange / ones, ones_like / zeros, zeros_like / empty ,
# empty_like / eye, identity
arr1 = np.array([1,2,3], dtype=np.float64)
np.linspace(-3,3,100)
np.digitize()

arr = np.array([1,2,3,4,5])
arr.dtype
float_arr = arr.astype(np.float64)
float_arr.dtype
numeric_strings = np.array(['1.23','-2.5','45242.1'], dtype=np.string_)
numeric_strings.astype(float)
empty_uint32 = np.empty(8, dtype='u4')
empty_uint32

arr = np.array([[1.,2.,3.],[4.,5.,6.]])
arr
arr * arr
arr - arr
1 / arr
arr ** 0.5
arr = np.array([[1.,2.,3.],[4.,5.,6.,7.]])
arr
arr = np.arange(10)
arr[5:8] = 12
arr
arr_slice = arr[5:8]
arr_slice[1] = 12345
arr
arr_slice[:] = 64
arr # view
arr2d = np.array([[1,2,3],[4,5,6],[7,8,9]])
arr2d[2]
arr2d[0][2]
arr2d[0,2]
 
arr3d = np.array([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]])
arr3d
arr3d[0]
old_values = arr3d[0].copy()
arr3d[0] = 43
arr3d
arr3d[0] = old_values
arr3d
arr3d[1,0]

arr[1:6]
arr2d
arr2d[:2]
arr2d[:2, 1:]
arr2d[1, :2]
arr2d[2, :1]
arr2d[:, :1]
arr2d[:2, 1:] = 0
arr2d

names = np.array(['Bob','Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
data = np.random.randn(7,4)
names
data
names == 'Bob'
data[names == 'Bob']
data[names == 'Bob', 2:]
data[names == 'Bob', 3]
names != 'Bob'
data[~(names == 'Bob')]

mask = (names == 'Bob') | (names == 'Will')
mask
data[mask]

# copy : astype, boolean indexing, pancy
# view : slice, asarray(if same type), T, swapaxes, transpose
data[data < 0] = 0
data
data[names != 'Joe'] = 7
data

import numpy as np
arr = np.empty((8, 4))
for i in range(8):
    arr[i] = i
arr
arr[[4,3,0,6]]
arr[4,3]
arr[[-3,-5,-7]]
arr = np.arange(32).reshape((8,4))
arr = np.arange(32).reshape([8,4])
arr = np.arange(32).reshape(8,4)
arr
arr[[1,5,7,2], [0,3,1,2]]
arr[[1,5,7,2]][:, [0,3,1,2]]
arr[np.ix_([1,5,7,2], [0,3,1,2])]
arr[[1,5,7,2]]

arr = np.arange(15).reshape((3,5))
arr
arr.T
arr = np.random.randn(6,3)
np.dot(arr.T, arr)
arr = np.arange(16).reshape((2,2,4))
arr
arr.transpose((1,0,2))
arr.swapaxes(1,2)
arr = np.arange(10)
np.sqrt(arr)
np.exp(arr)
x = np.random.randn(8)
y = np.random.randn(8)
x
y
np.maximum(x,y)
np.max(x)
arr = np.random.randn(7) * 5
quotient, remainder = np.modf(arr)
quotient
remainder
# abs, fabs / sqrt / square / Exp / Log, log10, log2, log1p(1+x) / sign / ceil, floor / rint / modf
# isnan, isfinite, isinf / cos, cosh, sin, sinh, tan, tanh / arccos, ... / logical_not (~arr)

# add, subtract, multiply / divide, floor_divide / power / maximum, fmax, minimum, fmin
# mod / copysign / greater, greater_equal, less, less_equal, not_equal / logical_and, logical_or, logical_xor

points = np.arange(-5,5,0.01)
xs, ys = np.meshgrid(points, points)
ys
xs
import matplotlib.pyplot as plt
import Tkinter
import numpy as np
z = np.sqrt(xs ** 2 + ys ** 2)
z
plt.imshow(z, cmap=plt.cm.gray); plt.colorbar()
