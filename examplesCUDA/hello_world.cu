/**
 * @file hello_world.cu
 * @author remzerrr (remi.helleboid@gmail.com)
 * @brief
 * @version 0.1
 * @date 2022-08-09
 *
 * @copyright Copyright (c) 2022
 *
 */

#include <stdio.h>   //printf
#include <stdlib.h>  //malloc

#include <iostream>

__global__ void cuda_hello() { printf("Hello World from GPU!\n"); }

int main() {
    std::cout << "Hello from CPU!" << std::endl;
    cuda_hello<<<2, 2>>>();
    cudaDeviceSynchronize();
    return 0;
}