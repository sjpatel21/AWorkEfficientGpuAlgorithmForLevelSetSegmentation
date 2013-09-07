#ifndef SRC_CUDA_EXPORT_ACTIVE_ELEMENTS_VOLUME_CU
#define SRC_CUDA_EXPORT_ACTIVE_ELEMENTS_VOLUME_CU

__global__ void  ExportActiveElementsVolumeKernel( CudaTagElement* deviceExportData,
                                                   size_t          numActiveVoxels,
                                                   dim3            volumeDimensions );  

extern "C" void CudaExportActiveElementsVolume( CudaTagElement* deviceExportData,
                                                size_t          numActiveElementsHost,
                                                dim3            volumeDimensions )
{
    // set the thread block size to the maximum
    dim3 threadBlockDimensions( 512, 1, 1 );

    int numThreadBlocks = static_cast< int >( ceil( numActiveElementsHost / ( 512.0f * 4.0f ) ) );

    // set the grid dimensions
    dim3 gridDimensions( numThreadBlocks, 1, 1 );

    if ( numThreadBlocks > 0 )
    {
        // call our kernel
        ExportActiveElementsVolumeKernel<<< gridDimensions, threadBlockDimensions >>>(
            deviceExportData,
            numActiveElementsHost,
            volumeDimensions );

        CudaSynchronize();
        CudaCheckErrors();    
    }
}

__device__ void ExportActiveElementsVolumeHelper( CudaCompactElement packedVoxelCoordinate,
                                                  dim3               volumeDimensions,
                                                  CudaTagElement*    deviceExportData )
{
    dim3         elementCoordinates        = UnpackCoordinates( packedVoxelCoordinate );
    unsigned int arrayIndexInVolume        = ComputeIndex3DToTiled1D( elementCoordinates, volumeDimensions );
    deviceExportData[ arrayIndexInVolume ] = 127;
}

__global__ void ExportActiveElementsVolumeKernel( CudaTagElement*     deviceExportData,
                                                  size_t              numActiveVoxels,
                                                  dim3                volumeDimensions )
{
    int arrayIndexInActiveElementTexture = ComputeIndexThread1DBlock1DTo1D();
    int arrayIndexInActiveElementList    = arrayIndexInActiveElementTexture * 4;

    if ( arrayIndexInActiveElementList < numActiveVoxels )
    {
        CudaCompactElement4 packedCoordinates = tex1Dfetch( CUDA_TEXTURE_REF_ACTIVE_ELEMENTS_1D, arrayIndexInActiveElementTexture );

        ExportActiveElementsVolumeHelper( packedCoordinates.x, volumeDimensions, deviceExportData );
        arrayIndexInActiveElementList++;

        if ( arrayIndexInActiveElementList < numActiveVoxels )
        {
            ExportActiveElementsVolumeHelper( packedCoordinates.y, volumeDimensions, deviceExportData );
            arrayIndexInActiveElementList++;

            if ( arrayIndexInActiveElementList < numActiveVoxels )
            {
                ExportActiveElementsVolumeHelper( packedCoordinates.z, volumeDimensions, deviceExportData );
                arrayIndexInActiveElementList++;

                if ( arrayIndexInActiveElementList < numActiveVoxels )
                {
                    ExportActiveElementsVolumeHelper( packedCoordinates.w, volumeDimensions, deviceExportData );
                }
            }
        }
    }
}


#endif