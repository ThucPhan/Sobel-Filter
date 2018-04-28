//convention: Col - x - i - n
//            Row - y - j - m

__constant__ float M1_c[FILTER_SIZE][FILTER_SIZE]; //horizontal
__constant__ float M2_c[FILTER_SIZE][FILTER_SIZE]; //vertical

__global__ void convolution(Matrix N,Matrix P1,Matrix P2,Matrix P3)
{
	/********************************************************************
	Determine input and output indexes of each thread
	Load a tile of the input image to shared memory
	Apply the filter on the input image tile
	Write the compute values to the output image at the correct indexes
	********************************************************************/

	__shared__ float N_s[BLOCK_SIZE][BLOCK_SIZE]; //shared memory

	int outRow = blockIdx.y*TILE_SIZE + threadIdx.y;
	int outCol = blockIdx.x*TILE_SIZE + threadIdx.x;

	int halo_size = FILTER_SIZE/2;
	
	int inRow = outRow - halo_size;
	int inCol = outCol - halo_size;

	float output1 = 0.0f; float output2 = 0.0f;

	if(inRow >= 0 && inRow < N.height && inCol >= 0 && inCol < N.width)
		N_s[threadIdx.y][threadIdx.x] = N.elements[inRow*N.width + inCol];
	else
		N_s[threadIdx.y][threadIdx.x] = 0.0f;

	__syncthreads();

	if(threadIdx.x < TILE_SIZE && threadIdx.y < TILE_SIZE) 
	{
		for(int j=0; j<FILTER_SIZE; j++)
			for(int i=0; i<FILTER_SIZE; i++)
			{
				output1 += M1_c[j][i] * N_s[j+threadIdx.y][i+threadIdx.x];
				output2 += M2_c[j][i] * N_s[j+threadIdx.y][i+threadIdx.x];
			}

		if(outRow < P3.height && outCol < P3.width)
		{
			P1.elements[outRow*P1.width + outCol] = output1;
			P2.elements[outRow*P2.width + outCol] = output2;
			P3.elements[outRow*P3.width + outCol] = output1 + output2;
		}
	}
}
