#add initcoord values by auto

import numpy as np
import matplotlib.pyplot as plt
from random import random
import math


val=[500,1000,5000,10000,50000,100000,500000,1000000]

points_x = []
points_y = []
def rand_cluster(n,r):
	x,y = 0,0
	for i in range(n):
		theta = 2*math.pi*random()
		s = r*random()
		points_x.append(x+s*math.cos(theta))
		points_y.append(y+s*math.sin(theta))

with open("input/initCoord.txt") as mu:
	data = mu.readlines()
mu_x = [float(row.rstrip('\n').split()[0]) for row in data]
mu_y = [float(row.rstrip('\n').split()[1]) for row in data]


def write_into_file():
	with open('input/'+str(len(points_x))+'_points.txt', 'w') as the_file:
		the_file.truncate()
		for i in range(len(points_x)):
			the_file.write(str(points_x[i])+" "+str(points_y[i])+'\n')

for i in val:
	del points_x[:]
	del points_y[:]
	rand_cluster(i,i)
	write_into_file()
	print("generated plot for %d points" % i)

	plt.scatter(points_x,points_y,c='b',s=70)
	plt.scatter(mu_x,mu_y,s=180,marker='^',c="yellow")
	plt.savefig('plots/input/'+ str(i) +'_plot')
