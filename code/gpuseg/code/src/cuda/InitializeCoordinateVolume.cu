#ifndef SRC_CUDA_INITIALIZE_COORDINATE_VOLUME_CU
#define SRC_CUDA_INITIALIZE_COORDINATE_VOLUME_CU

__global__ void  InitializeCoordinateVolumeKernel( CudaCompactElement* coordinateVolume,
                                                   dim3                volumeDimensions,
                                                   int                 logicalNumThreadBlocksX,
                                                   float               invNumThreadBlocksX );

extern "C" void CudaInitializeCoordinateVolume( CudaCompactElement* coordinateVolume, dim3 volumeDimensions )
{
    // set the thread block size to the maximum
    dim3 threadBlockDimensions( 8, 8, 8 );

    Assert( volumeDimensions.x % threadBlockDimensions.x == 0 );
    Assert( volumeDimensions.y % threadBlockDimensions.y == 0 );
    Assert( volumeDimensions.z % threadBlockDimensions.z == 0 );

    int numThreadBlocksX = volumeDimensions.x / threadBlockDimensions.x;
    int numThreadBlocksY = volumeDimensions.y / threadBlockDimensions.y;
    int numThreadBlocksZ = volumeDimensions.z / threadBlockDimensions.z;

    // since the grid dimensions are 2D only, we need to pack virtual XY dimensions into the actual X dimensions
    int   numThreadBlocksXY       = numThreadBlocksX * numThreadBlocksY;
    int   logicalNumThreadBlocksX = numThreadBlocksX;
    float invNumThreadBlocksX     = 1.0f / ( float ) numThreadBlocksX;

    // set the grid dimensions
    dim3 gridDimensions( numThreadBlocksXY, numThreadBlocksZ, 1 );

    // call our kernel
    InitializeCoordinateVolumeKernel<<< gridDimensions, threadBlockDimensions >>>( coordinateVolume, volumeDimensions, logicalNumThreadBlocksX, invNumThreadBlocksX );

    CudaSynchronize();
    CudaCheckErrors();
}

__global__ void  InitializeCoordinateVolumeKernel( CudaCompactElement* coordinateVolume,
                                                   dim3                volumeDimensions,
                                                   int                 logicalNumThreadBlocksX,
                                                   float               invNumThreadBlocksX )
{
    const dim3         elementCoordinates = ComputeIndexThread3DBlock2DTo3D( volumeDimensions, logicalNumThreadBlocksX, invNumThreadBlocksX );
    int                arrayIndex         = ComputeIndex3DToTiled1D( elementCoordinates, volumeDimensions );
    CudaCompactElement packedCoordinates  = PackCoordinates( elementCoordinates, 0, 0, 0, volumeDimensions );

    coordinateVolume[ arrayIndex ] = packedCoordinates;
}

#endif