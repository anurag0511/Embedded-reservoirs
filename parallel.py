
##import numpy
from numpy import *
import numpy as np
from matplotlib.pyplot import *
import matplotlib.pylab as plt
from pylab import *
import thread
import scipy.linalg
import threading
import Queue
from Queue import *
import time
import RPi.GPIO as GPIO
ion()

GPIO.setmode(GPIO.BCM)
GPIO.setup(18,GPIO.OUT)
pwm = GPIO.PWM(18,50)

with open('data_file.txt','r') as f:
	data = f.read().split()
data1 =loadtxt('data_file.txt')
var = 0
#data_x=[]  
#data_y=[]
for i in data:
#	
#	if var%2 == 0:
#		data_x.append(data1[var])
#	else :
#		data_y.append(data1[var])
#		
	print "this is the angle value", data1[var]
	var=var+1
 
x = len(data1)
#print "the size is ", x
#y = len(data_y)

plt.ion()   
plt.plot(data1[9:x])
#plt.plot(data_y[10:y-1])
plt.show
plt.savefig('plot1.png',format='png')
plt.close

inputsize = 1
outsize =1

#resSize = 100

trainLen = 99
initLen = 10
testLen = 99

q = Queue()
start_time = time.time()

def reservoir_1(resSize):

	W = np.random.rand(resSize,resSize)-0.5
	 
	Win = (np.random.rand(resSize,1+inputsize)-0.5)*1
	#print "this is done"
	#print "this is done2"
	rhoW = max(abs(linalg.eig(W)[0]))
	#print "this is done3", rhoW
	k=0;
	W *= 1.25/rhoW   
	X = zeros((1+inputsize+resSize,trainLen-initLen))
	for j in range(len(data1)):
	#	if j%2== 0:
		Yt = data1[None,initLen+1:trainLen+1]
	a = 0.2  
	#print Yt.shape
	x= zeros((resSize,1))
	for t in range(trainLen):
		u = data1[t]
		x = (1-a)*x + a*tanh( dot (Win , vstack((1,u)) ) + dot( W, x))
		if (t>= initLen): 
			X[:,t-initLen] = vstack((1,u,x))[:,0]  

	reg = .5e-4
	X_t = X.T
	Wout = dot(dot(Yt,X_t), linalg.inv( dot(X,X_t) + reg*eye(1+inputsize+resSize)))
	#print "the Output weights are ",Wout.max()
	 
	Y = zeros((outsize,testLen))     
	u = data1[trainLen]

	for t in range (testLen):
		x = (1-a)*x + a*tanh(dot ( Win,vstack((1,u)) ) + dot( W, x))
		y = dot( Wout, vstack((1,u,x)) )
		Y[:,t]= y
		u=y	
			
	
	map(q.put, Y)
	return Y
t=[]
resSize = 50	
count = 2
for i in range(count):
	t.insert(i,threading.Thread(name = "thread new",target = reservoir_1, args = (resSize,)))
	
for i in range(count):
	t[i].start()


q.join()

for i in range(count):
	t[i].join()

temp = zeros((outsize,testLen))
temp2 = zeros((outsize,testLen))
out = zeros((outsize,testLen))

temp = q.get()
temp2 = q.get()

#print "the value in temp is", temp[1]
for i in range (testLen):
	out[0][i] = (temp[i]+temp2[i])/2
mse =0
for j in range(35):
	mse = mse + square( (data1[10+j]-out[0][10+j])/35)
   
mse = mse/35	 
#mse = sum( square( data1[(trainLen+1):(trainLen+36)]-out[0,0:35]))/35
print "the program ended with mse ", mse,"and time in seconds", (time.time()-start_time)

new2 = zeros((1,100))
for i in range(89):
	new2[0][i] = out[0][10+i]-data1[10+i]
	print "the deviation is ", new2[0][i] 

maxd = max(new2[0])
mind = min(new2[0])	
print "the kmax and mindeviation are ",maxd, mind ,(maxd-mind)
pwm.start(2)
time.sleep(2)
new_time = time.time()

for i in range (89):
	time.sleep(0.6*((new2[0][i] - mind)/(maxd-mind)))
	new_duty = 2 + 8*((new2[0][i] - mind)/(maxd-mind))
	pwm.ChangeDutyCycle(new_duty)

print "the new loop servo time is ", time.time()-new_time
time.sleep(2)
pwm.ChangeDutyCycle(2)	
time.sleep(2)
GPIO.cleanup()

#print "the length is ", q.qsize()
#f = open('sum_file.txt','a+')
#for i in range(testLen):
#	f.write(str(Y[0].item(i)))
#	f.write(" ")

#f.write("\n")

plt.ion()  
plt.plot(out.T,"r")	 	
plt.show
plt.savefig('plot2.png',format='png')
plt.close
