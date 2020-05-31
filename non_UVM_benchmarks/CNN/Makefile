all:
	nvcc -lcuda -lcublas *.cu -o CNN   -Wno-deprecated-gpu-targets

run:
	./CNN
clean:
	rm -f CNN
