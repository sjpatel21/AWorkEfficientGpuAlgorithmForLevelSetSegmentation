#ifndef SRC_CUDA_EXPORT_LEVEL_SET_VOLUME_CU
#define SRC_CUDA_EXPORT_LEVEL_SET_VOLUME_CU

__global__ void  ExportLevelSetVolumeKernel( CudaTagElement* deviceExportData,
                                             dim3            volumeDimensions,
                                             float           invVolumeDimensionsX,
                                             float           invVolumeDimensionsXY );

extern "C" void CudaExportLevelSetVolume( CudaTagElement* deviceExportData,
                                          size_t          numActiveVoxels,
                                          dim3            volumeDimensions )
{
    // set the thread block size to the maximum
    dim3 threadBlockDimensions( 512, 1, 1 );

    int numElements     = volumeDimensions.x * volumeDimensions.y * volumeDimensions.z;
    int numThreadBlocks = static_cast< int >( ceil( numElements / ( 512.0f * 16.0f ) ) );

    float invVolumeDimensionsX  = 1.0f / (float)( volumeDimensions.x );
    float invVolumeDimensionsXY = 1.0f / (float)( volumeDimensions.x * volumeDimensions.y );

    // set the grid dimensions
    dim3 gridDimensions( numThreadBlocks, 1, 1 );

    // call our kernel
    ExportLevelSetVolumeKernel<<< gridDimensions, threadBlockDimensions >>>( deviceExportData, volumeDimensions, invVolumeDimensionsX, invVolumeDimensionsXY );

    CudaSynchronize();
    CudaCheckErrors();
}

__device__ void ExportLevelSetVolumeHelper( CudaCompactElement packedVoxelCoordinate,
                                            dim3               volumeDimensions,
                                            CudaTagElement*    deviceExportData )
{
    dim3         elementCoordinates        = UnpackCoordinates( packedVoxelCoordinate );
    unsigned int elementIndexTiled1D       = ComputeIndex3DToTiled1D( elementCoordinates, volumeDimensions );
    unsigned int elementIndex1D            = ComputeIndex3DTo1D( elementCoordinates, volumeDimensions );

    float levelSetValue                    = tex1Dfetch( CUDA_TEXTURE_REF_LEVEL_SET_1D, elementIndexTiled1D );
    char  levelSetValueChar                = __float2int_rd( levelSetValue * 127.0f );

    deviceExportData[ elementIndex1D ]     = levelSetValueChar;
}


__global__ void ExportLevelSetVolumeKernel( CudaTagElement* deviceExportData,
                                             dim3            volumeDimensions,
                                             float           invVolumeDimensionsX,
                                             float           invVolumeDimensionsXY )
{
    int   globalThreadIndex1D    = ComputeIndexThread1DBlock1DTo1D();
    int   elementIndex1D         = globalThreadIndex1D * 16;
    int4* deviceExportDataAsInt4 = (int4*)deviceExportData;

    __align__( 16 ) __shared__ char stagingChar [ 512 ][ 16 ];

    dim3  elementIndex3D;
    int   elementIndexTiled1D;
    float levelSetValue;

    elementIndex3D                           = ComputeIndex1DTo3D( elementIndex1D++, volumeDimensions, invVolumeDimensionsX, invVolumeDimensionsXY );
    elementIndexTiled1D                      = ComputeIndex3DToTiled1D( elementIndex3D, volumeDimensions );
    levelSetValue                            = tex1Dfetch( CUDA_TEXTURE_REF_LEVEL_SET_1D, elementIndexTiled1D );
    stagingChar[ threadIdx.x ][ 0 ]          = __float2int_rd( levelSetValue * 127.0f );

    elementIndex3D                           = ComputeIndex1DTo3D( elementIndex1D++, volumeDimensions, invVolumeDimensionsX, invVolumeDimensionsXY );
    elementIndexTiled1D                      = ComputeIndex3DToTiled1D( elementIndex3D, volumeDimensions );
    levelSetValue                            = tex1Dfetch( CUDA_TEXTURE_REF_LEVEL_SET_1D, elementIndexTiled1D );
    stagingChar[ threadIdx.x ][ 1 ]          = __float2int_rd( levelSetValue * 127.0f );

    elementIndex3D                           = ComputeIndex1DTo3D( elementIndex1D++, volumeDimensions, invVolumeDimensionsX, invVolumeDimensionsXY );
    elementIndexTiled1D                      = ComputeIndex3DToTiled1D( elementIndex3D, volumeDimensions );
    levelSetValue                            = tex1Dfetch( CUDA_TEXTURE_REF_LEVEL_SET_1D, elementIndexTiled1D );
    stagingChar[ threadIdx.x ][ 2 ]          = __float2int_rd( levelSetValue * 127.0f );

    elementIndex3D                           = ComputeIndex1DTo3D( elementIndex1D++, volumeDimensions, invVolumeDimensionsX, invVolumeDimensionsXY );
    elementIndexTiled1D                      = ComputeIndex3DToTiled1D( elementIndex3D, volumeDimensions );
    levelSetValue                            = tex1Dfetch( CUDA_TEXTURE_REF_LEVEL_SET_1D, elementIndexTiled1D );
    stagingChar[ threadIdx.x ][ 3 ]          = __float2int_rd( levelSetValue * 127.0f );

    elementIndex3D                           = ComputeIndex1DTo3D( elementIndex1D++, volumeDimensions, invVolumeDimensionsX, invVolumeDimensionsXY );
    elementIndexTiled1D                      = ComputeIndex3DToTiled1D( elementIndex3D, volumeDimensions );
    levelSetValue                            = tex1Dfetch( CUDA_TEXTURE_REF_LEVEL_SET_1D, elementIndexTiled1D );
    stagingChar[ threadIdx.x ][ 4 ]          = __float2int_rd( levelSetValue * 127.0f );

    elementIndex3D                           = ComputeIndex1DTo3D( elementIndex1D++, volumeDimensions, invVolumeDimensionsX, invVolumeDimensionsXY );
    elementIndexTiled1D                      = ComputeIndex3DToTiled1D( elementIndex3D, volumeDimensions );
    levelSetValue                            = tex1Dfetch( CUDA_TEXTURE_REF_LEVEL_SET_1D, elementIndexTiled1D );
    stagingChar[ threadIdx.x ][ 5 ]          = __float2int_rd( levelSetValue * 127.0f );

    elementIndex3D                           = ComputeIndex1DTo3D( elementIndex1D++, volumeDimensions, invVolumeDimensionsX, invVolumeDimensionsXY );
    elementIndexTiled1D                      = ComputeIndex3DToTiled1D( elementIndex3D, volumeDimensions );
    levelSetValue                            = tex1Dfetch( CUDA_TEXTURE_REF_LEVEL_SET_1D, elementIndexTiled1D );
    stagingChar[ threadIdx.x ][ 6 ]          = __float2int_rd( levelSetValue * 127.0f );

    elementIndex3D                           = ComputeIndex1DTo3D( elementIndex1D++, volumeDimensions, invVolumeDimensionsX, invVolumeDimensionsXY );
    elementIndexTiled1D                      = ComputeIndex3DToTiled1D( elementIndex3D, volumeDimensions );
    levelSetValue                            = tex1Dfetch( CUDA_TEXTURE_REF_LEVEL_SET_1D, elementIndexTiled1D );
    stagingChar[ threadIdx.x ][ 7 ]          = __float2int_rd( levelSetValue * 127.0f );

    elementIndex3D                           = ComputeIndex1DTo3D( elementIndex1D++, volumeDimensions, invVolumeDimensionsX, invVolumeDimensionsXY );
    elementIndexTiled1D                      = ComputeIndex3DToTiled1D( elementIndex3D, volumeDimensions );
    levelSetValue                            = tex1Dfetch( CUDA_TEXTURE_REF_LEVEL_SET_1D, elementIndexTiled1D );
    stagingChar[ threadIdx.x ][ 8 ]          = __float2int_rd( levelSetValue * 127.0f );

    elementIndex3D                           = ComputeIndex1DTo3D( elementIndex1D++, volumeDimensions, invVolumeDimensionsX, invVolumeDimensionsXY );
    elementIndexTiled1D                      = ComputeIndex3DToTiled1D( elementIndex3D, volumeDimensions );
    levelSetValue                            = tex1Dfetch( CUDA_TEXTURE_REF_LEVEL_SET_1D, elementIndexTiled1D );
    stagingChar[ threadIdx.x ][ 9 ]          = __float2int_rd( levelSetValue * 127.0f );

    elementIndex3D                           = ComputeIndex1DTo3D( elementIndex1D++, volumeDimensions, invVolumeDimensionsX, invVolumeDimensionsXY );
    elementIndexTiled1D                      = ComputeIndex3DToTiled1D( elementIndex3D, volumeDimensions );
    levelSetValue                            = tex1Dfetch( CUDA_TEXTURE_REF_LEVEL_SET_1D, elementIndexTiled1D );
    stagingChar[ threadIdx.x ][ 10 ]          = __float2int_rd( levelSetValue * 127.0f );

    elementIndex3D                           = ComputeIndex1DTo3D( elementIndex1D++, volumeDimensions, invVolumeDimensionsX, invVolumeDimensionsXY );
    elementIndexTiled1D                      = ComputeIndex3DToTiled1D( elementIndex3D, volumeDimensions );
    levelSetValue                            = tex1Dfetch( CUDA_TEXTURE_REF_LEVEL_SET_1D, elementIndexTiled1D );
    stagingChar[ threadIdx.x ][ 11 ]          = __float2int_rd( levelSetValue * 127.0f );

    elementIndex3D                           = ComputeIndex1DTo3D( elementIndex1D++, volumeDimensions, invVolumeDimensionsX, invVolumeDimensionsXY );
    elementIndexTiled1D                      = ComputeIndex3DToTiled1D( elementIndex3D, volumeDimensions );
    levelSetValue                            = tex1Dfetch( CUDA_TEXTURE_REF_LEVEL_SET_1D, elementIndexTiled1D );
    stagingChar[ threadIdx.x ][ 12 ]          = __float2int_rd( levelSetValue * 127.0f );

    elementIndex3D                           = ComputeIndex1DTo3D( elementIndex1D++, volumeDimensions, invVolumeDimensionsX, invVolumeDimensionsXY );
    elementIndexTiled1D                      = ComputeIndex3DToTiled1D( elementIndex3D, volumeDimensions );
    levelSetValue                            = tex1Dfetch( CUDA_TEXTURE_REF_LEVEL_SET_1D, elementIndexTiled1D );
    stagingChar[ threadIdx.x ][ 13 ]          = __float2int_rd( levelSetValue * 127.0f );

    elementIndex3D                           = ComputeIndex1DTo3D( elementIndex1D++, volumeDimensions, invVolumeDimensionsX, invVolumeDimensionsXY );
    elementIndexTiled1D                      = ComputeIndex3DToTiled1D( elementIndex3D, volumeDimensions );
    levelSetValue                            = tex1Dfetch( CUDA_TEXTURE_REF_LEVEL_SET_1D, elementIndexTiled1D );
    stagingChar[ threadIdx.x ][ 14 ]          = __float2int_rd( levelSetValue * 127.0f );

    elementIndex3D                           = ComputeIndex1DTo3D( elementIndex1D++, volumeDimensions, invVolumeDimensionsX, invVolumeDimensionsXY );
    elementIndexTiled1D                      = ComputeIndex3DToTiled1D( elementIndex3D, volumeDimensions );
    levelSetValue                            = tex1Dfetch( CUDA_TEXTURE_REF_LEVEL_SET_1D, elementIndexTiled1D );
    stagingChar[ threadIdx.x ][ 15 ]          = __float2int_rd( levelSetValue * 127.0f );


    int4* stagingInt4 = (int4*)( &( stagingChar[ threadIdx.x ][ 0 ] ) );

    int4 int4value = *stagingInt4;

    deviceExportDataAsInt4[ globalThreadIndex1D ] = int4value;
}






#endif