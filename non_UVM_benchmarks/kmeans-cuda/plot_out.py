import numpy as np
import matplotlib.pyplot as plt
from random import random
import math

val=[500,1000,5000,10000,50000,100000,500000,1000000]

for k in val:
	with open('input/'+ str(k) +'_points.txt') as the_file:
		data = the_file.readlines()
	x = [float(row.rstrip('\n').split()[0]) for row in data]
	y = [float(row.rstrip('\n').split()[1]) for row in data]


	with open('output/cuda/'+ str(k) +'_group_members.txt') as group:
		data = group.readlines()
	gr = [int(row.rstrip('\n')) for row in data]


	with open("output/cuda/"+  str(k)  +"_centroids.txt") as mu:
		data = mu.readlines()
	mu_x_out = [float(row.rstrip('\n').split()[0]) for row in data]
	mu_y_out = [float(row.rstrip('\n').split()[1]) for row in data]

	plt.scatter(x,y,c=gr,s=70)
	plt.scatter(mu_x_out,mu_y_out,s=180,marker='^',c="red")
	plt.savefig('plots/output/'+ str(k) +'_plot')
	print("generated plot for %d points" % k)

