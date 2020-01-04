echo "parallel image compression using cuda"
echo "compiling..."
nvcc kmeans.cu
echo "enter the name of the file"
read filename
# echo "enter dimensions"
# read width
# read height
width=$(identify -format "%w" $filename)
height=$(identify -format "%h" $filename)
python2 imageConverter.py $filename
name=${filename%.*}
./a.out $name".raw" $name"out.raw" $width $height 64 500
python2 imageConverter.py $name"out.raw" $width $height


