./kmeans_standard 3 input/500_points.txt 3 100
./kmeans_standard 3 input/1000_points.txt 3 100 
./kmeans_standard 3 input/5000_points.txt 3 100
./kmeans_standard 3 input/10000_points.txt 3 100
./kmeans_standard 3 input/50000_points.txt 3 100
./kmeans_standard 3 input/100000_points.txt 3 100
./kmeans_standard 3 input/500000_points.txt 3 100
./kmeans_standard 3 input/1000000_points.txt 3 100


./kmeans_cuda 2 input/500_points.txt 500
./kmeans_cuda 2 input/1000_points.txt 1000 
./kmeans_cuda 2 input/5000_points.txt 5000
./kmeans_cuda 2 input/10000_points.txt 10000
./kmeans_cuda 2 input/50000_points.txt 50000
./kmeans_cuda 2 input/100000_points.txt 100000
./kmeans_cuda 2 input/500000_points.txt 500000
./kmeans_cuda 2 input/1000000_points.txt 1000000

rm *.txt