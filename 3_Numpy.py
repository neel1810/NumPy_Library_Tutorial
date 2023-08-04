# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 19:01:41 2020

@author: DELL
"""

import numpy as np
help(np.dtype)
help(np.array([1,2]))
np.array([1,2,3],dtype=np.int64).dtype
# create a numpy array
myar1 = np.array([[1,2,3,4], [5,6,7,8]], dtype=np.int64)
print(myar1)
myar1.dtype
myar1.shape
myar1.strides
myar1.size
# Create an array of ones
np.ones((3,4))
np.conj(3+1j)
# Create an array of zeros
np.zeros((2,3,4),dtype=np.int16)

# Create an array with random values
np.random.random((2,2))
np.random.rand() ## Values between 0,1 uniform
np.random.rand(3,2)
np.random.randn(2, 4) # normal
#low, high, size
np.random.randint(2)
np.random.randint(2,5, size=10)
np.random.randint(5, size=(2, 4))

# Create an empty array
A=np.empty((3,2))

# Create a full array
np.full((2,2),99)

# Create an array of evenly-spaced values
#(start,stop excl.the last,hop by)
ar1=np.arange(10,25,5)
ar1
np.random.shuffle(ar1)
ar1
np.arange(0,271,27)
#np.arange(0,50,19)
# Create an array of evenly-spaced values
#(start, stop, no of values)
np.linspace(0,10,20)
'''
from numpy import pi
x=np.linspace(0,2*pi,100)
f = np.sin(x)
import matplotlib.pyplot as plt
plt.plot(f)
'''
# np.eye
np.eye(3)
np.identity(3)
## Saving
table_27 = np.arange(0,271,27)
np.savetxt(r"D:\t27.csv",table_27, delimiter=' ')
#my_array2 = np.genfromtxt('test_out.txt',
#                      skip_header=1,
     #                 filling_values=-99)
al=[20,30]
bl=[1,2]
cl=al+bl
cl
 a = np.array( [20,30,40,50] )
 b = np.arange( 4 )
 b
 c = a+b
 c

 b**2
 #import matplotlib.pyplot as plt
 10*np.sin(a)
 a<35


# Matrix multiplication
A = np.array( [[2,1],[0,3]] )
A
B = np.array( [[2,0],[3,4]] )
B
A * B                       # elementwise product
#array([[2, 0],[0, 4]])
A @ B                       # matrix product
#array([[5, 4], [3, 4]])
A.dot(B)                    # another matrix product
#array([[5, 4],[3, 4]])

## some += operations
a = np.ones((2,3), dtype=int)
b = np.random.random((2,3))
a *= 3
 a
#array([[3, 3, 3],[3, 3, 3]])
 b + a
 b
#array([[3.51182162, 3.9504637 , 3.14415961],
#       [3.94864945, 3.31183145, 3.42332645]])

np.random.random((3,4))

=======================================================================
