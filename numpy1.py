#NumPy is numerical python library
#Import Numpy 
import numpy as np

list1=[1,2,3] # not using numpy
arr1 = np.array([10,11,12,13]) # defining 1-D array 
arr1_ = np.array(list1)
arr2 = np.array([[11,12,13,14,15],[21,22,23,24,25,25],[31,32,33,34,35]]) # 2D Array declation

#Some array functions
arr2.ndim
arr2.shape
arr2.dtype.name
arr2.itemsize
arr2.size
type(arr2)

a_float = np.array([1.01,1.03,1.05]) # dtype is float
a_mix = np.array([1.,2,T])

# define a complex number array
a_complex = np.array([[10,20],[1,2]],dtype = complex)

#Several inbuilt functions
np.arange(10)
np.arange(3,30,3) #change the last 3 to 0.3 and observe the changes # start,stop,hop-by
a_ar=np.arange(10).reshape(2,5)

np.linspace(0,3,10) # returns 10 equidistant values between 0 t0 3 

np.zeros((3,3))
np.empty((2,4))
np.ones((3,2,5))
np.full((2,2),55)
np.eye(3)
np.identity(5)

# Create an array with random values
np.random.random((2,2))
np.random.rand() ## Values between 0,1 uniform
np.random.rand(3,2)
np.random.randn(2, 4) # normal
#low, high, size
np.random.randint(2)
np.random.randint(2,5, size=10)
np.random.randint(5, size=(2, 4))