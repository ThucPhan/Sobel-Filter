#include <stdlib.h>
#include <stdio.h>
#include "support.h"

Matrix allocateMatrix(unsigned height, unsigned width)
{
	Matrix mat;
	mat.height = height;
	mat.width = mat.pitch = width;
	mat.elements = (float*)malloc(height*width*sizeof(float));
	
	return mat;
}

void loadData(Matrix mat,const char *filename)
{
    float tmp; //stored value
    int ind = 0; //index

    FILE *fp;
    fp = fopen(filename,"r");
    while(!feof(fp))
    {
        fscanf(fp,"%f",&tmp);
        mat.elements[ind] = tmp;
        ind = ind+1;
    }
    fclose(fp);
}

void saveResult(Matrix mat,const char *filename)
{
    int m = mat.height;
    int n = mat.width;

    FILE *fp;
    fp = fopen(filename,"w");
    for(int j=0;j<m;j++)
    {
        for(int i=0;i<n;i++)
            fprintf(fp,"%f",mat.elements[j*n+i]);
        fprintf(fp,"\n");
    }
    fclose(fp);
}

Matrix allocateDeviceMatrix(unsigned height,unsigned width)
{
	Matrix mat;
	
	mat.height = height;
	mat.width = mat.pitch = width;
	cudaMalloc((void**)&(mat.elements), height*width*sizeof(float));

	return mat;
}

void copyToDeviceMatrix(Matrix dst, Matrix src)
{
	cudaMemcpy(dst.elements, src.elements, src.height*src.width*sizeof(float), cudaMemcpyHostToDevice);
}

void copyFromDeviceMatrix(Matrix dst, Matrix src)
{
	cudaMemcpy(dst.elements, src.elements, src.height*src.width*sizeof(float), cudaMemcpyDeviceToHost);	
}

void verify(Matrix M,Matrix N,Matrix P) 
{
    const float relativeTolerance = 1e-6;
    
    for(int row=0;row<N.height;++row) 
    {
        for(int col=0;col<N.width;++col) 
        {
            float sum = 0.0f;
            for(int j=0;j<M.height;++j) 
            {
                for(int i=0;i<M.width;++i) 
                {
                    int jN = row - M.height/2 + j;
                    int iN = col - M.width/2 + i;
                    if(jN>=0 && jN<N.height && iN>=0 && iN<N.width)
                        sum += M.elements[j*M.width+i] * N.elements[jN*N.width+iN];
                }
            }
            
            float relativeError = (sum - P.elements[row*P.width+col])/sum;
            if( (relativeError>relativeTolerance) || (relativeError<-relativeTolerance)) 
            {
                printf("TEST FAILED...");
                exit(0);
            }
        }
    }
    printf("TEST PASSED...");
}

void freeMatrix(Matrix mat)
{
	free(mat.elements);
	mat.elements = NULL;
}

void freeDeviceMatrix(Matrix mat)
{
	cudaFree(mat.elements);
	mat.elements = NULL;
}

void startTime(Timer* timer) 
{
    gettimeofday(&(timer->startTime),NULL);
}

void stopTime(Timer* timer) 
{
    gettimeofday(&(timer->endTime),NULL);
}

float elapsedTime(Timer timer) 
{
    return ( (float) ((timer.endTime.tv_sec-timer.startTime.tv_sec) 
        + (timer.endTime.tv_usec-timer.startTime.tv_usec)/1.0e6) );
}
