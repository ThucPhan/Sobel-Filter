#run makefile and run convolution

NVCC        = nvcc
EXE	        = convolution
OBJ	        = main.o support.o

default: $(EXE)

main.o: main.cu kernel.cu support.h
	$(NVCC) -c -o $@ main.cu 

support.o: support.cu support.h
	$(NVCC) -c -o $@ support.cu

$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE)

clean:
	rm -rf *.o $(EXE)
