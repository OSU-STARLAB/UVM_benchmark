import os
from metric_list import metrics
# print(metrics[0])


# folders = ["bfs", "BN", "CNN", "kmeans", "knn", "logistic-regression", "rodinia","SVM"]
# files = ["", "", "", "","", "","", "","", "","", "","", "","", ""]
folders = ["BN", "bfs", "CNN", "kmeans", "knn", "logistic-regression", "SVM", "rodinia"]
subfolders = ["backprop", "cfd", "dwt2d", "gaussian", "hotspot", "hotspot3D", "huffman", "nn", "nw", "particlefilter", "pathfinder", "srad", "streamcluster"]
UVM_flag = False
print_options = "--print-gpu-trace --csv  --track-memory-allocations off --profile-child-processes"
print_file = " --log-file "


for UVM_flag in [True, False]:
    base_path = "UVM_benchmarks/"
    if not UVM_flag:
        base_path = "non_UVM_benchmarks/"
    for i, folder in enumerate(folders):
        if folder != "rodinia":
            path = base_path + folder + "/"
            command = "cd " + path
            command = "cd " + path + "; nvprof " + print_options + print_file + folder + "_%p_metric.csv"  + " " + metrics[24]  + metrics[8] + metrics[12] + " " + " ./run"
            print(command)
            os.system(command)
            command = "cd " + path + "; nvprof " + print_options + print_file + folder + "_%p_summary.csv"   + " " + " ./run"
            print(command)
            os.system(command)
        else:
            for j, subfolder in enumerate(subfolders):
                path = base_path + folder + "/" + subfolder
                command = "cd " + path
                command = "cd " + path + "; nvprof " + print_options + print_file + subfolder + "_%p_metric.csv" + " "  + metrics[24]  + metrics[8] + metrics[12] + " " + " ./run"
                print(command)
                os.system(command)
                command = "cd " + path + "; nvprof " + print_options + print_file + subfolder + "_%p_summary.csv" + " " + " ./run"
                print(command)
                os.system(command)
