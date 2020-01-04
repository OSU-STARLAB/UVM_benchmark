# kmeans-cuda

k-means clustering is a method of clustering which aims to partition n data points into k clusters (n >> k) in which each observation belongs to the cluster with the nearest mean.  

The nearness is calculated by distance function which is mostly Euclidian  distance or Manhattan distance.  
Parallel kmeans cuda and sequential Kmeans algorithm is compared.  

<p align="center">
  <img width="480" height="400" src="plots/input/500_plot.png"><br>
  <a align="center"> Datapoints to be clustered</a>
</p>

<p align="center">
  <img width="480" height="400" src="plots/output/500_plot.png"><br>
  <a align="center"> Clustered Datapoints (k = 3)</a>
</p>

Kmeans is powerful algorithm used on images. Its used to cluster the image pixel values into different bins (k value) to reduce the image information required to store the image. Widely used in image compression.

<p align="center">
  <img width="480" height="400" src="image-clustering/colosseo.jpg"><br>
  <a align="center"> Colloseum Image</a>
</p>

<p align="center">
  <img width="480" height="400" src="image-clustering/c_8.jpg"><br>
  <a align="center"> Clustered Colloseum Image (k = 8)</a>
</p>
