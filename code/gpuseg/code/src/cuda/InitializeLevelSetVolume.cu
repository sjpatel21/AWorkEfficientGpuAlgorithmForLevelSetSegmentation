#ifndef SRC_CUDA_INITIALIZE_LEVEL_SET_VOLUME_CU
#define SRC_CUDA_INITIALIZE_LEVEL_SET_VOLUME_CU

__global__ void  InitializeLevelSetKernel( CudaLevelSetElement* deviceData,
                                           CudaTagElement*      deviceExportData,
                                           dim3                 volumeDimensions,
                                           int                  logicalNumThreadBlocksX,
                                           float                invNumThreadBlocksX,
                                           dim3                 seedRegion,
                                           int                  sphereSize,
                                           float                outOfPlaneAnisotropy );


extern "C" void CudaInitializeLevelSetVolume( CudaLevelSetElement* deviceData,
                                              CudaTagElement*      deviceExportData,
                                              dim3                 volumeDimensions,
                                              dim3                 seedCoordinates,
                                              int                  sphereSize,
                                              float                outOfPlaneAnisotropy )
{
    // set the thread block size to the maximum
    dim3 threadBlockDimensions( 8, 8, 4 );

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
    InitializeLevelSetKernel<<< gridDimensions, threadBlockDimensions >>>(
        deviceData,
        deviceExportData,
        volumeDimensions,
        logicalNumThreadBlocksX,
        invNumThreadBlocksX,
        seedCoordinates,
        sphereSize,
        outOfPlaneAnisotropy );

    CudaSynchronize();
    CudaCheckErrors();
}

__global__ void  InitializeLevelSetKernel( CudaLevelSetElement* deviceData,
                                           CudaTagElement*      deviceExportData,
                                           dim3                 volumeDimensions,
                                           int                  logicalNumThreadBlocksX,
                                           float                invNumThreadBlocksX,
                                           dim3                 seedRegion,
                                           int                  sphereSize,
                                           float                outOfPlaneAnisotropy )
{
    const dim3 elementCoordinates   = ComputeIndexThread3DBlock2DTo3D( volumeDimensions, logicalNumThreadBlocksX, invNumThreadBlocksX );
    int        arrayIndex           = ComputeIndex3DToTiled1D( elementCoordinates, volumeDimensions );
    int        arrayIndexUntiled    = ComputeIndex3DTo1D( elementCoordinates, volumeDimensions );
    int        elementToSeedVectorX = seedRegion.x - elementCoordinates.x;
    int        elementToSeedVectorY = seedRegion.y - elementCoordinates.y;
    float      elementToSeedVectorZ = ( __int2float_rz( seedRegion.z ) - __int2float_rz( elementCoordinates.z ) ) * outOfPlaneAnisotropy;
    float      distanceSquared      = Sqr( elementToSeedVectorX ) + Sqr( elementToSeedVectorY ) + Sqr( elementToSeedVectorZ );
    float      distance             = sqrt( distanceSquared );
    float      smoothDistance       = LEVEL_SET_INITIALIZE_SMOOTH_DISTANCE;
    float      currentLevelSetValue = 1.0f;

    if ( distance < sphereSize )
    {
        currentLevelSetValue = -1.0f;
    }
    else if ( distance < sphereSize + smoothDistance )
    {
        float outsideDistance = distance - sphereSize;

        currentLevelSetValue = -1.0f + ( outsideDistance / ( smoothDistance / 2.0f ) );
    }

    float oldLevelSetValue   = tex1Dfetch( CUDA_TEXTURE_REF_LEVEL_SET_1D, arrayIndex );
    float newLevelSetValue   = Clamp( oldLevelSetValue + currentLevelSetValue - 1.0f, -1.0f, 1.0f );

#ifdef LEVEL_SET_FIELD_FIXED_POINT
    int writeValue           = __float2int_rd( newLevelSetValue * LEVEL_SET_FIELD_MAX_VALUE );
    deviceData[ arrayIndex ] = writeValue;
#else
    deviceData[ arrayIndex ] = newLevelSetValue;
#endif

    int exportValue                       = __float2int_rd( newLevelSetValue * LEVEL_SET_FIELD_EXPORT_MAX_VALUE );
    deviceExportData[ arrayIndexUntiled ] = exportValue;
}










__global__ void  AddLevelSetKernel( CudaTagElement*      deviceResultData,
                                    CudaTagElement*      deviceAddData,
                                    dim3                 volumeDimensions,
                                    int                  logicalNumThreadBlocksX,
                                    float                invNumThreadBlocksX );


extern "C" void CudaAddLevelSetVolume( CudaTagElement*      deviceResultData,
                                       CudaTagElement*      deviceAddData,
                                       dim3                 volumeDimensions )
{
    // set the thread block size to the maximum
    dim3 threadBlockDimensions( 8, 8, 4 );

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
    AddLevelSetKernel<<< gridDimensions, threadBlockDimensions >>>(
        deviceResultData,
        deviceAddData,
        volumeDimensions,
        logicalNumThreadBlocksX,
        invNumThreadBlocksX );

    CudaSynchronize();
    CudaCheckErrors();
}

__global__ void  AddLevelSetKernel( CudaTagElement* deviceResultData,
                                    CudaTagElement* deviceAddData,
                                    dim3            volumeDimensions,
                                    int             logicalNumThreadBlocksX,
                                    float           invNumThreadBlocksX )
{
    const dim3 elementCoordinates   = ComputeIndexThread3DBlock2DTo3D( volumeDimensions, logicalNumThreadBlocksX, invNumThreadBlocksX );
    int        arrayIndexUntiled    = ComputeIndex3DTo1D( elementCoordinates, volumeDimensions );

    char  rawOldLevelSetValue = (char)deviceResultData[ arrayIndexUntiled ];
    float oldLevelSetValue    = rawOldLevelSetValue / LEVEL_SET_FIELD_EXPORT_MAX_VALUE;

    char  rawNewLevelSetValue = (char)deviceAddData[ arrayIndexUntiled ];
    float newLevelSetValue    = rawNewLevelSetValue / LEVEL_SET_FIELD_EXPORT_MAX_VALUE;

    float resultLevelSetValue             = Clamp( oldLevelSetValue + newLevelSetValue - 1.0f, -1.0f, 1.0f );
    int   rawResultLevelSetValue          = __float2int_rd( resultLevelSetValue * LEVEL_SET_FIELD_EXPORT_MAX_VALUE );
    deviceResultData[ arrayIndexUntiled ] = rawResultLevelSetValue;
}







#endif
