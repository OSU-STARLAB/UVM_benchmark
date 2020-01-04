import random

val=[500,1000,5000,10000,50000,100000,500000]
					
for k in val:
    list=[random.randint(0,255) for x in range(k)]
    output=open("input/x_coordinates_"+str(k)+".txt", "w")

    for i in list:
        output.write(str(i)+'\n')

    list=[random.randint(0,255) for x in range(k)]
    output=open("input/y_coordinates_"+str(k)+".txt", "w")

    for i in list:
        output.write(str(i)+'\n')

