with open("CUDAtimes.txt") as cuda:
	cudalist=cuda.read().splitlines()
	length=len(cudalist)

cudalist1=[]
for i in range(length):
	cudalist1.append(float(cudalist[i]))
#print (cudalist1)

with open("Sequentialtimes.txt") as seq:
	seqlist=seq.read().splitlines()
	#print (seqlist)

seqlist1=[]
for i in range(length):
	seqlist1.append(float(seqlist[i]))
#print (seqlist1)

stdlist = []
with open("Standardtimes.txt") as std:
	stdlist=std.read().splitlines()
	#print (seqlist)

stdlist1 = []
for i in range(length):
	stdlist1.append(float(stdlist[i]))



val=[500,1000,5000,10000,50000,100000,500000,1000000]

for i in range(length):
	# print ("Dataset size = "+str(val[i])+"    Sequential-Speed-Up = "+str(seqlist1[i]/cudalist1[i])+"    Standard-Speed-Up = "+str(stdlist1[i]/cudalist1[i]))
	print seqlist1[i]/cudalist1[i]
