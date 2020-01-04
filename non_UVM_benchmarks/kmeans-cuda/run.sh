python plot_in.py

g++ -std=c++11 kmeansSequential.cpp -o kmeansSequential 
nvcc -std=c++11 kmeans_cuda.cu -o kmeans_cuda -Wno-deprecated-gpu-targets
nvcc -std=c++11 standard_kmeans.cu -o standard_kmeans -Wno-deprecated-gpu-targets



./kmeansSequential 2 input/500_points.txt 500
./kmeansSequential 2 input/1000_points.txt 1000 
./kmeansSequential 2 input/5000_points.txt 5000
./kmeansSequential 2 input/10000_points.txt 10000
./kmeansSequential 2 input/50000_points.txt 50000
./kmeansSequential 2 input/100000_points.txt 100000
./kmeansSequential 2 input/500000_points.txt 500000
./kmeansSequential 2 input/1000000_points.txt 1000000


./standard_kmeans 3 input/500_points.txt 3 100
./standard_kmeans 3 input/1000_points.txt 3 100 
./standard_kmeans 3 input/5000_points.txt 3 100
./standard_kmeans 3 input/10000_points.txt 3 100
./standard_kmeans 3 input/50000_points.txt 3 100
./standard_kmeans 3 input/100000_points.txt 3 100
./standard_kmeans 3 input/500000_points.txt 3 100
./standard_kmeans 3 input/1000000_points.txt 3 100


./kmeans_cuda 2 input/500_points.txt 500
./kmeans_cuda 2 input/1000_points.txt 1000 
./kmeans_cuda 2 input/5000_points.txt 5000
./kmeans_cuda 2 input/10000_points.txt 10000
./kmeans_cuda 2 input/50000_points.txt 50000
./kmeans_cuda 2 input/100000_points.txt 100000
./kmeans_cuda 2 input/500000_points.txt 500000
./kmeans_cuda 2 input/1000000_points.txt 1000000



python printSpeedUp.py
rm *.txt

python plot_out.py


