import RPi.GPIO as GPIO

import time

GPIO.setmode(GPIO.BCM)
GPIO.setup(18,GPIO.OUT)
pwm = GPIO.PWM(18,50)
time.sleep(2)
pwm.start(2)
time.sleep(2)
#pwm.ChangeDutyCycle(5)
time.sleep(0.6)
pwm.ChangeDutyCycle(10)
#time.sleep(0.6)
#pwm.ChangeDutyCycle(2)
time.sleep(0.6)
pwm.stop()
GPIO.cleanup()
 

 
