import os
filename = "data125.cu"
file_write = "data45.cu"

f_w = open(file_write,"w")
with open(filename) as f:
    content = f.readlines()

    for i, line in enumerate(content):
        if i < 8:
            f_w.write(line)
        elif i >= 8:
            f_w.write(line[0:90])
            f_w.write("\n")
f_w.close()