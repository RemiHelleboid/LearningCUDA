# Learning CUDA 

This repository contains examples of CUDA programming.
It aims to be a collection of examples that can be used as a starting point for learning CUDA.
The examples are written in C++ and CUDA C.  
We try to keep as many C++ features as possible, for example working with std::vector and not only with C-style arrays.

## Compilation

To compile the examples follow those steps:

```Shell
mkdir build && cd build
cmake ..
make
```

## Run example
Exampled are stored in the _build/exampleCUDA/_ folder.

From the _build/_ folder, you can run the example _my_example_ by running:
```Bash
./exampleCUDA/my_example
```

You can profile the execution of the example by running:
```Bash
nvprof ./exampleCUDA/my_example
```

## Examples

### Hello World (hello_world.cu)

This example is a simple hello world program that prints "Hello World" on the screen.
The hello world kerne is run on 2 blocks of 2 threads each, so you should see 4 times "Hello World" on the screen.

### C-style array addition (add_arrays.cu)

This example shows how to add two C-style arrays on the GPU.
It uses the cudaMallocManaged which allocates the memory on the GPU and the CPU with a single call. It is convenient for simple examples, but less flexible than cudaMalloc and cudaMemcpy.

### Vector addition with std::vector (add_stdvectors.cu)

This example shows how to add two std::vector on the GPU.
Obviously, it is not possible to use cudaMallocManaged with std::vector, so we use cudaMalloc and cudaMemcpy to allocate and copy the data to the GPU. On the GPU, the data are represented by a pointer to the first element of the vector plus the size of the vector.
Actually we compute a linear combination of two vectors, but the principle is the same. 
A time measurement is done to compare the execution time on the CPU and the GPU.