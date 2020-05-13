import os
from metric_list import metrics
# print(metrics[0])


# folders = ["bfs", "BN", "CNN", "kmeans", "knn", "logistic-regression", "rodinia","SVM"]
# files = ["", "", "", "","", "","", "","", "","", "","", "","", ""]
folders = ["BN"]
files = ["./ordergraph"]
UVM_flag = True
print_options = " --print-gpu-trace "
enable_metric = " --metrics "

base_path = "UVM_benchmarks/"
if not UVM_flag:
    base_path = "non_UVM_benchmarks/"

for i, folder in enumerate(folders):
    path = base_path + folder + "/"
    command = "cd " + path + "; nvprof " + print_options + \
        enable_metric + metrics[11] + " " + files[i]
    print(command)
    os.system(command)
