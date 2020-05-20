import os
from metric_list import metrics
# print(metrics[0])


# folders = ["bfs", "BN", "CNN", "kmeans", "knn", "logistic-regression", "rodinia","SVM"]
# files = ["", "", "", "","", "","", "","", "","", "","", "","", ""]
folders = ["BN"]
files = ["./ordergraph"]
UVM_flag = False
print_options = " --csv --log-file test.csv  --track-memory-allocations off"


for UVM_flag in [True, False]:
    base_path = "UVM_benchmarks/"
    if not UVM_flag:
        base_path = "non_UVM_benchmarks/"

    for i, folder in enumerate(folders):
        path = base_path + folder + "/"
        command = "cd " + path + "; nvprof " + print_options + metrics[9]  + metrics[75] + " " + files[i]
        print(command)
        os.system(command)
