#ifndef __FILEH__
#define __FILEH__

#include <sys/time.h>

typedef struct {
    struct timeval startTime;
    struct timeval endTime;
} Timer;

typedef struct {
    unsigned int width;
    unsigned int height;
    unsigned int pitch;
    float* elements;
} Matrix;

#define FILTER_SIZE 5
#define TILE_SIZE 20
#define BLOCK_SIZE (TILE_SIZE + FILTER_SIZE - 1)

Matrix allocateMatrix(unsigned height,unsigned width);
void loadData(Matrix mat,const char *filename);
void saveResult(Matrix mat,const char *filename);
Matrix allocateDeviceMatrix(unsigned height,unsigned width);
void copyToDeviceMatrix(Matrix dst,Matrix src);
void copyFromDeviceMatrix(Matrix dst,Matrix src);
void verify(Matrix M,Matrix N,Matrix P);
void freeMatrix(Matrix mat);
void freeDeviceMatrix(Matrix mat);
void startTime(Timer* timer);
void stopTime(Timer* timer);
float elapsedTime(Timer timer);

#endif
