EXECUTABLE := memory_occ
CUFILES := memory_occ.cu

default:
	nvcc -g -Wno-deprecated-gpu-targets ${CUFILES} -lcudart -o ${EXECUTABLE}
clean:
	rm -rf ${EXECUTABLE} 