./kmeans_standard 3 ../../data/kmeans/500_points.txt 3 100
./kmeans_standard 3 ../../data/kmeans/1000_points.txt 3 100 
./kmeans_standard 3 ../../data/kmeans/5000_points.txt 3 100
./kmeans_standard 3 ../../data/kmeans/10000_points.txt 3 100
./kmeans_standard 3 ../../data/kmeans/50000_points.txt 3 100
./kmeans_standard 3 ../../data/kmeans/100000_points.txt 3 100
./kmeans_standard 3 ../../data/kmeans/500000_points.txt 3 100
./kmeans_standard 3 ../../data/kmeans/1000000_points.txt 3 100


./kmeans_cuda 2 ../../data/kmeans/500_points.txt 500
./kmeans_cuda 2 ../../data/kmeans/1000_points.txt 1000 
./kmeans_cuda 2 ../../data/kmeans/5000_points.txt 5000
./kmeans_cuda 2 ../../data/kmeans/10000_points.txt 10000
./kmeans_cuda 2 ../../data/kmeans/50000_points.txt 50000
./kmeans_cuda 2 ../../data/kmeans/100000_points.txt 100000
./kmeans_cuda 2 ../../data/kmeans/500000_points.txt 500000
./kmeans_cuda 2 ../../data/kmeans/1000000_points.txt 1000000

rm *.txt