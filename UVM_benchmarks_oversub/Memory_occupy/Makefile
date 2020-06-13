EXECUTABLE := memory_occ
EXECUTABLE2 := manual_occ
CUFILES := memory_occ.cu
CUFILES2 := manual_occ.cu
default:
	nvcc -g -Wno-deprecated-gpu-targets ${CUFILES} -lcudart -o ${EXECUTABLE}
	nvcc -g -Wno-deprecated-gpu-targets ${CUFILES2} -lcudart -o ${EXECUTABLE2}
clean:
	rm -rf ${EXECUTABLE} 
	rm -rf ${EXECUTABLE2} 