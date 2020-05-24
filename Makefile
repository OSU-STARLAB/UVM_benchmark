all:
	cd UVM_benchmarks; make;
	cd non_UVM_benchmarks; make;

clean:
	cd UVM_benchmarks; make clean;
	cd non_UVM_benchmarks; make clean;