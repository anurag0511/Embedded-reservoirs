import smbus
import math
import datetime
import sys
from numpy import *
import numpy as np
from matplotlib.pyplot import *
import matplotlib.pylab as plt
from pylab import *
import thread
import scipy.linalg
import time
import random
import RPi.GPIO as GPIO

#Kalman filter variables
pi = 3.57
Q_angle = 0.02
Q_gyro = 0.0015
R_angle = 0.005
y_bias = 0.0
x_bias = 0.0

XP_00 = 0.0
XP_01 = 0.0
XP_10 = 0.0
XP_11 = 0.0
YP_00 = 0.0
YP_01 = 0.0
YP_10 = 0.0
YP_11 = 0.0
kalman_x_list = []
Kalman_xangle = 0.0
Kalman_yangle = 0.0

#i2c intialisation
bus=smbus.SMBus(1)
i2c_adr = 0x68
M_PI = 3.14159265358979323846
Rad_To_Deg =  57.29578
bus.write_byte_data(i2c_adr,0x6b,0)

kalmanY = 0.0
kalmanX = 0.0
time1 = datetime.datetime.now() 
var=0

#set the training params
inputsize = 1
resSize = 100
outsize =1 
trainLen = 99
initLen = 0
testLen = 100
maxsamples = 100

#set the weight matrices 
random.seed()

W = (np.random.rand(resSize,resSize)*2)-1
Wprob = np.random.rand(resSize,resSize)
W[Wprob<0.7]= 0

Wfb = (np.random.rand(resSize,outsize)-0.5)*2
Wfbprob = np.random.rand(resSize,outsize)
Wfb[Wfbprob<0.7] = 0

Wout = (np.random.rand(outsize,resSize)-0.5)*2

Win = (np.random.rand(resSize,inputsize)-0.5)*2
Winprob =np.random.rand(resSize,inputsize)
Win[Winprob<0.7] = 0

Wdel = zeros((outsize,resSize))
#spectral radius 
rhoW = max(abs(linalg.eig(W)[0]))
W *= 1.25/rhoW
#leaking/learning rate
a = 0.5
#set the input updating dataset
x= zeros((resSize,1))

#GPIO initialisation
GPIO.setmode(GPIO.BCM)
GPIO.setup(18,GPIO.OUT)
pwm = GPIO.PWM(18,50)

def kalmanFilterY( accelangle, gyrorate, DT):
	y=0.0
	s=0.0
	global Kalman_yangle
	global Q_angle
	global Q_gyro
	global y_bias
	global YP_00
	global YP_01
	global YP_10
	global YP_11
	
	Kalman_yangle += DT*(gyrorate - y_bias)
	YP_00 += (-DT*(YP_10+YP_01)+Q_angle*DT)
	YP_01 += -DT*YP_11
	YP_10 += -DT*YP_11 
	YP_11 += Q_gyro*DT
	
	y = accelangle - Kalman_yangle 
	s = YP_00 + R_angle
	K_0 = YP_00/s
	K_1 = YP_10/s
	
	Kalman_yangle += (K_0*y)
	y_bias += K_1*y
	YP_00 -= K_0*YP_00
	YP_01 -= K_0*YP_01
	YP_10 -= K_1*YP_00
	YP_11 -= K_1*YP_01
	return Kalman_yangle
	

def kalmanFilterX( accelangle, gyrorate, DT):
	x=0.0
	s=0.0

	global Kalman_xangle
	global Q_angle
	global Q_gyro
	global x_bias
	global XP_00
	global XP_01
	global XP_10
	global XP_11
	
	Kalman_xangle += DT*(gyrorate - x_bias)
	XP_00 += (-DT*(XP_10+XP_01)+Q_angle*DT)
	XP_01 += -DT*XP_11
	XP_10 += -DT*XP_11 
	XP_11 += Q_gyro*DT
	
	x= accelangle - Kalman_xangle
	s = XP_00 + R_angle
	K_0 = XP_00/s
	K_1 = XP_10/s
	
	Kalman_xangle += (K_0*x)
	x_bias += K_1*x
	XP_00 -= K_0*XP_00
	XP_01 -= K_0*XP_01
	XP_10 -= K_1*XP_00
	XP_11 -= K_1*XP_01
	return Kalman_xangle	
	
def read_byte(address):
	return bus.read_byte_data(i2c_adr,address)

def read_word( address,flag):
	if flag==1:
		msb = bus.read_byte_data(i2c_adr,address)
		lsb = bus.read_byte_data(i2c_adr,address+1)
	else:
		msb = bus.read_byte_data(i2c_adr,address+1)
		lsb = bus.read_byte_data(i2c_adr,address)
	
	value = (msb<<8)+lsb
	return value

def read_word_comp(address,flag):
	value=read_word(address,flag)
	if(value>=0x8000):
		return -((65535 - value)+1)
	else:
		return value

def get_final_value():
	s=None
	gyro_x_angle = 0.0
	gyro_y_angle = 0.0
	gyro_z_angle = 0.0
	gyro_x = read_word_comp(0x43,1)
	gyro_y = read_word_comp(0x45,1)
	gyro_z = read_word_comp(0x47,1)
	
	accel_x = read_word_comp(0x3b,1)
	accel_y = read_word_comp(0x3d,1)
	accel_z = read_word_comp(0x3f,1)
	
	mag_x = read_word_comp(0x03,0)
	mag_y = read_word_comp(0x05,0)
	mag_z = read_word_comp(0x07,0)
	
	time2=datetime.datetime.now() - time1
	time3 = datetime.datetime.now()
	TP = time2.microseconds/(1000000*1.0)
	
	gyro_x_rate = gyro_x *(0.07)
	gyro_y_rate = gyro_y*(0.07)
	gyro_z_rate = gyro_z*(0.07)
	
	gyro_x_angle += gyro_x_rate*TP 
	gyro_y_angle += gyro_y_rate*TP
	gyro_z_angle += gyro_z_rate*TP
	
	accel_x_angle = (math.atan2(accel_y,accel_z) + M_PI)*Rad_To_Deg
	accel_y_angle = (math.atan2(accel_z,accel_x) + M_PI)*Rad_To_Deg
		
	kalmanY = kalmanFilterY(accel_y_angle,gyro_y_rate,TP) 
	kalmanX = kalmanFilterX(accel_x_angle,gyro_x_rate,TP)
	kalman_x_list.append(kalmanX)
	size = len(kalman_x_list)
	print "the angles are"
	print "y angle filtered ", kalmanY,"y_angle  ",accel_y_angle
	print "the angles are"
	print "x angle filtered ", kalmanX,"x_angle  ",accel_x_angle
	print " size is    ", size
	s = None
	s = str(kalmanX)+"-"+str(kalmanY)
	return s
x_value=np.array([])
x_value = np.append(x_value,[0])
temp =0
print "the dataset recording starts"
data1= np.array([])   
while temp < 100:
	xval,yval = get_final_value().split("-")
	print "rthe xval is ", xval
	data1=np.append(data1,[float(xval)])
	temp+=1
norm =max(abs(data1))
print "the normaliser is ", norm	
	


#norm = 1
start_time = time.time()
for j in range(1,100):
	var = 0	
	while var<100:
		if j == 1:
			xval1,yval1 = get_final_value().split("-")
			x_value = np.append(x_value,[float(xval1)])
			
		#time.sleep(0.03)
		#x_value[:] = [i / 200 for i in x_value]
		#training the data
		if var > 1:
			u =x_value[var-1]/norm
			#print " the u is ", u
			x = (1-a)*x + a*tanh( dot (Win , u ) + dot( W, x) + dot(Wfb, (x_value[var-2]/norm)) )
			ypre = dot(Wout,x)    
			Wdel = 0.01*((x_value[var]/norm) - ypre)*(x.T)
			Wout = Wout + Wdel
		var+=1	
print "the training is done", time.time()-start_time	

Y = zeros((outsize,testLen))
for t in range(1,testLen):
	u = x_value[t-1]/norm
	if t<21:
		  x = (1-a)*x + a*tanh( dot (Win , u ) + dot( W, x) + dot(Wfb, x_value[t]/norm) )
	else:
		x = (1-a)*x + a*tanh( dot (Win , u ) + dot( W, x) + dot(Wfb,y) ) 
	y = dot(Wout,x)
	Y[:,t]= y

mse = sum(square((x_value[50:95]/norm)-Y[0,50:95])/45)
print "the mse is ", mse
new2 = np.array([])
for i in range (1,testLen):
	new2 = np.append(new2, [abs(x_value[i]/norm - Y[0][i])])

pwm.start(2)
time.sleep(2)
new_time= time.time()
for i in range (1,testLen-1):
	time.sleep(0.6*(new2[i]))
	new_duty = 2 + 8*((new2[i]))
	pwm.ChangeDutyCycle(new_duty)


print "the new loop servo time is ", time.time()-new_time
time.sleep(2)
pwm.ChangeDutyCycle(2)	
time.sleep(2)
GPIO.cleanup()
     
							
plt.ion()  
plt.plot(Y.T,"r")
plt.plot(x_value/norm,"g")	 	
plt.show
plt.savefig('plot2.png',format='png')
