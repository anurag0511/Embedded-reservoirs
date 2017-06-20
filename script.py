import os
from numpy import *
import numpy as np
from matplotlib.pyplot import *
import matplotlib.pylab as plt
from pylab import *
import thread
import scipy.linalg

with open('sum_file.txt','w') as f:
	f.write(" ")
 
for i in range(50):
	os.system ("sudo python network_2.py")


with open('sum_file.txt','r') as f:
	data = f.readline().split()
data1 = loadtxt('sum_file.txt')
data2 = loadtxt('data_file.txt')
print "the len is ", data1.shape
Y = zeros((1,100))
x = 0 
for i in range(99):
	for j in range(50):
		x=x+data1[j][i]
	Y[0][i]=x/50
	x=0	
new2 = zeros((1,100))
for i in range(99):
	new2[0][i] = Y[0][i]-data2[i]
	print "the deviation is ", new2[0][i] 
plt.ion()  
plt.plot(Y.T,"r")	 
plt.plot(data2,"b")
plt.plot(new2.T,"g")	
plt.show
plt.savefig('plot3.png',format='png')


