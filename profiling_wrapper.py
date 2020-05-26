import os
from metric_list import metrics
# print(metrics[0])
import time
from time import localtime, strftime


# folders = ["bfs", "BN", "CNN", "kmeans", "knn", "logistic-regression", "rodinia","SVM"]
# files = ["", "", "", "","", "","", "","", "","", "","", "","", ""]
folders = ["BN", "bfs", "CNN", "kmeans", "knn", "logistic-regression", "SVM", "rodinia", "polybench"]
subfolders = ["backprop", "cfd", "dwt2d", "gaussian", "hotspot", "hotspot3D", "huffman", "nn", "nw", "particlefilter", "pathfinder", "srad", "streamcluster"]
subfolders2 = ["2DCONV", "2MM", "3DCONV", "3MM", "ATAX", "BICG", "CORR", "COVAR", "FDTD-2D", "GEMM", "GESUMMV", "GRAMSCHM", "MVT", "SYR2K", "SYRK"]
UVM_flag = False
print_options = "--print-gpu-trace --csv  --track-memory-allocations off --profile-child-processes"
print_file = " --log-file "
time_str = strftime("%Y-%m-%d-%H-%M-%S", localtime())
if not os.path.exists('results/'+time_str+"/"):
    os.mkdir('results/'+time_str+"/")
output_path = " ../../results/"
output_sub_path = " ../../../results/"

print_general_options = "--csv  --track-memory-allocations off --profile-child-processes "

def profile_details():
    for UVM_flag in [True, False]:
        base_path = "UVM_benchmarks/"
        if not UVM_flag:
            base_path = "non_UVM_benchmarks/"
        for i, folder in enumerate(folders):
            if folder != "rodinia" and folder != "polybench":
                path = base_path + folder + "/"
                command = "cd " + path
                command = "cd " + path + "; nvprof " + print_options + print_file + folder + "_%p_metric.csv"  + " " + metrics[24]  + metrics[8] + metrics[12] + " " + " ./run"
                print(command)
                os.system(command)
                command = "cd " + path + "; nvprof " + print_options + print_file + folder + "_%p_summary.csv"   + " " + " ./run"
                print(command)
                os.system(command)
            elif folder == "polybench":
                for j, subfolder in enumerate(subfolders2):
                    path = base_path + folder + "/" + subfolder
                    command = "cd " + path
                    command = "cd " + path + "; nvprof " + print_options + print_file + subfolder + "_%p_metric.csv" + " "  + metrics[24]  + metrics[8] + metrics[12] + " " + " ./run"
                    print(command)
                    os.system(command)
                    command = "cd " + path + "; nvprof " + print_options + print_file + subfolder + "_%p_summary.csv" + " " + " ./run"
                    print(command)
                    os.system(command)
            else:
                for j, subfolder in enumerate(subfolders):
                    path = base_path + folder + "/" + subfolder
                    command = "cd " + path
                    command = "cd " + path + "; nvprof " + print_general_options + print_file + subfolder + "_%p_metric.csv" + " "  + metrics[24]  + metrics[8] + metrics[12] + " " + " ./run"
                    print(command)
                    os.system(command)
                    command = "cd " + path + "; nvprof " + print_general_options + print_file + subfolder + "_%p_summary.csv" + " " + " ./run"
                    print(command)
                    os.system(command)

def profile_general():
    for UVM_flag in [True, False]:
        base_path = "UVM_benchmarks/"
        if not UVM_flag:
            base_path = "non_UVM_benchmarks/"
        for i, folder in enumerate(folders):
            if folder != "rodinia" and folder != "polybench":
                path = base_path + folder + "/"
                if UVM_flag:
                    command = "cd " + path + "; nvprof " + print_general_options + print_file + output_path + time_str + "/"  + folder + "_%p_summary_UVM.csv"   + " " + " ./run"
                else:
                    command = "cd " + path + "; nvprof " + print_general_options + print_file + output_path + time_str + "/"  + folder + "_%p_summary_non_UVM.csv"   + " " + " ./run"
                print(command)
                os.system(command)
            elif folder == "polybench":
                for j, subfolder in enumerate(subfolders2):
                    path = base_path + folder + "/" + subfolder
                    if UVM_flag:
                        command = "cd " + path + "; nvprof " + print_general_options + print_file+ output_sub_path + time_str + "/"  + subfolder + "_%p_summary_UVM.csv" + " " + " ./run"
                    else:
                        command = "cd " + path + "; nvprof " + print_general_options + print_file+ output_sub_path + time_str + "/"  + subfolder + "_%p_summary_non_UVM.csv" + " " + " ./run"
                    print(command)
                    os.system(command)
            else:
                for j, subfolder in enumerate(subfolders):
                    path = base_path + folder + "/" + subfolder
                    if UVM_flag:
                        command = "cd " + path + "; nvprof " + print_general_options + print_file + output_sub_path + time_str + "/"  + subfolder + "_%p_summary_UVM.csv" + " " + " ./run"
                    else:
                        command = "cd " + path + "; nvprof " + print_general_options + print_file+ output_sub_path + time_str + "/"  + subfolder + "_%p_summary_non_UVM.csv" + " " + " ./run"
                    print(command)
                    os.system(command)


if __name__ == "__main__":
    profile_general()