//Consider 2 filter

#include <stdio.h>
#include "support.h"
#include "kernel.cu"

int main()
{
    Timer timer;

    // Initialize host variables ----------------------------------------------
    printf("Setting up the problem..."); fflush(stdout);
startTime(&timer);

    //M1: horizontal, M2: vertical
    Matrix M1_h,M2_h,N_h,P1_h,P2_h,P3_h; // M: filter, N: input image, P: output image
	Matrix N_d,P1_d,P2_d,P3_d;

	unsigned imageHeight = 1080; //for test_image2
	unsigned imageWidth = 1920; //for test_image2
	
	dim3 dim_grid, dim_block;

	/* Allocate host memory */
	M1_h = allocateMatrix(FILTER_SIZE,FILTER_SIZE);
    M2_h = allocateMatrix(FILTER_SIZE,FILTER_SIZE);
	N_h = allocateMatrix(imageHeight,imageWidth);
	P1_h = allocateMatrix(imageHeight,imageWidth);
    P2_h = allocateMatrix(imageHeight,imageWidth);
    P3_h = allocateMatrix(imageHeight,imageWidth);

	/* Initialize filter and images */
    loadData(M1_h,"Ysobel5_5.txt"); //horizontal filter
    loadData(M2_h,"Xsobel5_5.txt"); //vertical filter
    loadData(N_h,"test_image2.txt"); //image

stopTime(&timer); 
    printf("%f s\n", elapsedTime(timer));
    printf("Image: %u x %u\n",imageHeight,imageWidth);
    printf("Mask: %u x %u\n",FILTER_SIZE,FILTER_SIZE);

    // Allocate device variables ----------------------------------------------
    printf("Allocating device variables..."); fflush(stdout);
startTime(&timer);

    N_d = allocateDeviceMatrix(imageHeight,imageWidth);
	P1_d = allocateDeviceMatrix(imageHeight,imageWidth);
    P2_d = allocateDeviceMatrix(imageHeight,imageWidth);
    P3_d = allocateDeviceMatrix(imageHeight,imageWidth);

    cudaDeviceSynchronize();
stopTime(&timer); 
    printf("%f s\n",elapsedTime(timer));

    // Copy host variables to device ------------------------------------------
    printf("Copying data from host to device..."); fflush(stdout);
startTime(&timer);

   	/* Copy image to device global memory */
	copyToDeviceMatrix(N_d,N_h);

	/* Copy mask to device constant memory */
    cudaMemcpyToSymbol(M1_c, M1_h.elements, M1_h.height*M1_h.width*sizeof(float));
    cudaMemcpyToSymbol(M2_c, M2_h.elements, M2_h.height*M2_h.width*sizeof(float));
	
	cudaDeviceSynchronize();
stopTime(&timer); 
    printf("%f s\n",elapsedTime(timer));

    // Launch kernel ----------------------------------------------------------
    printf("Launching kernel..."); fflush(stdout);
startTime(&timer);
  
	dim_block.x = BLOCK_SIZE; dim_block.y = BLOCK_SIZE; dim_block.z = 1;
	
    dim_grid.x = imageWidth/TILE_SIZE;
	if(imageWidth%TILE_SIZE != 0) 
        dim_grid.x++;
	
    dim_grid.y = imageHeight/TILE_SIZE;
	if(imageHeight%TILE_SIZE != 0) 
        dim_grid.y++;
	
    dim_grid.z = 1;

	convolution<<<dim_grid,dim_block>>>(N_d,P1_d,P2_d,P3_d);

	cudaDeviceSynchronize();
stopTime(&timer); 
    printf("%f s\n", elapsedTime(timer));

    // Copy device variables from host ----------------------------------------
    printf("Copying data from device to host..."); fflush(stdout);
startTime(&timer);

    copyFromDeviceMatrix(P1_h,P1_d);
    copyFromDeviceMatrix(P2_h,P2_d);
    copyFromDeviceMatrix(P3_h,P3_d);
    
    cudaDeviceSynchronize();
stopTime(&timer); 
    printf("%f s\n", elapsedTime(timer));

    // Verify correctness -----------------------------------------------------
    printf("Executing the convolution in host..."); fflush(stdout);
startTime(&timer);

    printf("Verifying results..."); fflush(stdout);
    verify(M1_h,N_h,P1_h);
    verify(M2_h,N_h,P2_h);

stopTime(&timer); 
    printf("%f s\n", elapsedTime(timer));

    /* Saving Results */
    saveResult(P1_h,"testImage2_Results_horizontal.txt");
    saveResult(P2_h,"testImage2_Results_vertical.txt");
    saveResult(P3_h,"testImage2_Results_resultant.txt");

    // Free memory ------------------------------------------------------------
	freeMatrix(M1_h); freeMatrix(M2_h); freeMatrix(N_h); 
    freeMatrix(P1_h); freeMatrix(P2_h); freeMatrix(P3_h);
	freeDeviceMatrix(N_d); 
    freeDeviceMatrix(P1_d); freeDeviceMatrix(P2_d); freeDeviceMatrix(P3_d);

	return 0;
}
