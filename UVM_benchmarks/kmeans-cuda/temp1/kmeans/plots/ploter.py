import numpy as np
import matplotlib.pyplot as plt

with open("x.txt") as fx:
	data = fx.readlines()
x = [float(row.rstrip('\n')) for row in data]


with open("y.txt") as fy:
	data = fy.readlines()
y = [float(row.rstrip('\n')) for row in data]

fig = plt.figure()

ax1 = fig.add_subplot(111)

ax1.set_title("Plot title")    
ax1.set_xlabel('x label')
ax1.set_ylabel('y label')

ax1.plot(x,y,'ro')

leg = ax1.legend()

plt.show()
