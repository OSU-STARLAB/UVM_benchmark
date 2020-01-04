all:
	nvcc -lcuda -lcublas *.cu -o CNN  -arch=compute_70 -Wno-deprecated-gpu-targets

run:
	./CNN
clean:
	rm CNN
