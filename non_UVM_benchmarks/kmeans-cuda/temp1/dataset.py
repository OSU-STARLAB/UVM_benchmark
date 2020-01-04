import numpy as np
from random import random
import math

points_x = []
points_y = []
def rand_cluster(n,r):
	x,y = 0,0
	for i in range(n):
		theta = 2*math.pi*random()
		s = r*random()
		points_x.append(x+s*math.cos(theta))
		points_y.append(y+s*math.sin(theta))


def write_into_file():
	with open('input/1000000points.txt', 'r+') as the_file:
		the_file.truncate()
		for i in range(len(points_x)):
			the_file.write(str(points_x[i])+" "+str(points_y[i])+'\n')

rand_cluster(100000,100000)
write_into_file()

