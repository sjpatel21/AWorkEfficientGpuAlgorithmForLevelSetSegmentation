#ifndef SRC_CUDA_FILTER_DUPLICATES_CU
#define SRC_CUDA_FILTER_DUPLICATES_CU

__global__ void  FilterDuplicatesInitializeKernel( CudaCompactElement* deviceData,
                                                   CudaCompactElement* deviceKeep,
                                                   CudaTagElement*     deviceTag,
                                                   size_t              numElements,
                                                   dim3                volumeDimensions );

__global__ void  FilterDuplicatesFilterKernel( CudaCompactElement* deviceData,
                                               CudaCompactElement* deviceKeep,
                                               CudaTagElement*     deviceTag,
                                               size_t              numElements,
                                               dim3                volumeDimensions );

extern "C" void CudaFilterDuplicates( CudaCompactElement* deviceData,
                                      CudaCompactElement* deviceKeep,
                                      CudaTagElement*     deviceTag,
                                      size_t              numElements,
                                      dim3                volumeDimensions )
{
    // set the thread block size to the maximum
#ifdef CUDA_ARCH_SM_10   
    dim3 threadBlockDimensions( 128, 1, 1 );
    int numThreadBlocks = static_cast< int >( ceil( numElements / ( 128.0f * 4.0f ) ) );
#endif

#ifdef CUDA_ARCH_SM_13
    dim3 threadBlockDimensions( 512, 1, 1 );
    int numThreadBlocks = static_cast< int >( ceil( numElements / ( 512.0f * 4.0f ) ) );
#endif

    // set the grid dimensions
    dim3 gridDimensions( numThreadBlocks, 1, 1 );

    if ( numThreadBlocks > 0 )
    {
        CudaBindTextureToBuffer< CudaCompactElement, CudaCompactElement4 >( CUDA_TEXTURE_ACTIVE_ELEMENTS_1D, deviceData );
        CudaBindTextureToBuffer< CudaCompactElement, CudaCompactElement4 >( CUDA_TEXTURE_VALID_ELEMENTS_1D,  deviceKeep );

        // call our kernel
        FilterDuplicatesInitializeKernel<<< gridDimensions, threadBlockDimensions >>>(
            deviceData,
            deviceKeep,
            deviceTag,
            numElements,
            volumeDimensions );

        unsigned int numElementsWarpAligned = CudaGetWarpAlignedValue( numElements );

        for ( int i = 1; i < 7; i++ )
        {
            CudaCompactElement* currentOutputList = deviceData + ( numElementsWarpAligned * i );
            CudaCompactElement* currentValidList  = deviceKeep + ( numElementsWarpAligned * i );

            CudaBindTextureToBuffer< CudaCompactElement, CudaCompactElement4 >( CUDA_TEXTURE_ACTIVE_ELEMENTS_1D, currentOutputList );
            CudaBindTextureToBuffer< CudaCompactElement, CudaCompactElement4 >( CUDA_TEXTURE_VALID_ELEMENTS_1D,  currentValidList );

            // call our kernel
            FilterDuplicatesFilterKernel<<< gridDimensions, threadBlockDimensions >>>(
                currentOutputList,
                currentValidList,
                deviceTag,
                numElements,
                volumeDimensions );
        }

        CudaSynchronize();
        CudaCheckErrors();
    }
}

__device__ void FilterDuplicatesInitializeHelper( CudaCompactElement keep,
                                                  CudaCompactElement packedVoxelCoordinate,
                                                  int                arrayIndexInActiveElementList,
                                                  dim3               volumeDimensions,
                                                  CudaTagElement*    deviceTag )
{
    if ( keep == 1 )
    {
        dim3               tagCoordinates        = UnpackCoordinates( packedVoxelCoordinate );
        unsigned int       arrayIndexInTagVolume = ComputeIndex3DToTiled1D( tagCoordinates, volumeDimensions );
        deviceTag[ arrayIndexInTagVolume ]       = 1;
    }
}

__global__ void FilterDuplicatesInitializeKernel( CudaCompactElement* deviceData,
                                                  CudaCompactElement* deviceKeep,
                                                  CudaTagElement*     deviceTag,
                                                  size_t              numElements,
                                                  dim3                volumeDimensions )
{
    int arrayIndexInActiveElementTexture = ComputeIndexThread1DBlock1DTo1D();
    int arrayIndexInActiveElementList    = arrayIndexInActiveElementTexture * 4;

    if ( arrayIndexInActiveElementList < numElements )
    {
        CudaCompactElement4 keepElements = tex1Dfetch( CUDA_TEXTURE_REF_VALID_ELEMENTS_1D, arrayIndexInActiveElementTexture );
        
        if ( keepElements.x == 1 || keepElements.y == 1 || keepElements.z == 1 || keepElements.w == 1 )
        {
            CudaCompactElement4 packedCoordinates = tex1Dfetch( CUDA_TEXTURE_REF_ACTIVE_ELEMENTS_1D, arrayIndexInActiveElementTexture );

            FilterDuplicatesInitializeHelper( keepElements.x, packedCoordinates.x, arrayIndexInActiveElementList, volumeDimensions, deviceTag );
            arrayIndexInActiveElementList++;

            if ( arrayIndexInActiveElementList < numElements )
            {
                FilterDuplicatesInitializeHelper( keepElements.y, packedCoordinates.y, arrayIndexInActiveElementList, volumeDimensions, deviceTag );
                arrayIndexInActiveElementList++;

                if ( arrayIndexInActiveElementList < numElements )
                {
                    FilterDuplicatesInitializeHelper( keepElements.z, packedCoordinates.z, arrayIndexInActiveElementList, volumeDimensions, deviceTag );
                    arrayIndexInActiveElementList++;

                    if ( arrayIndexInActiveElementList < numElements )
                    {
                        FilterDuplicatesInitializeHelper( keepElements.w, packedCoordinates.w, arrayIndexInActiveElementList, volumeDimensions, deviceTag );
                    }
                }
            }
        }
    }
}

__device__ void FilterDuplicatesFilterHelper( CudaCompactElement  keep,
                                              CudaCompactElement  packedVoxelCoordinate,
                                              int                 arrayIndexInActiveElementList,
                                              dim3                volumeDimensions,
                                              CudaTagElement*     deviceTag,
                                              CudaCompactElement* deviceKeep )
{
    if ( keep == 1 )
    {
        dim3 currentCoordinatesDim3;
        int  arrayIndex;

        dim3 tagCoordinates = UnpackCoordinates( packedVoxelCoordinate );

        GET_TAG_NEIGHBORHOOD_HELPER_1D( alreadySet, currentCoordinatesDim3, arrayIndex, tagCoordinates, volumeDimensions, +, 0, +, 0, +, 0 );

        if ( alreadySet == 1 )
        {
            deviceKeep[ arrayIndexInActiveElementList ] = 0;
        }
        else
        {
            unsigned int arrayIndexInTagVolume = ComputeIndex3DToTiled1D( tagCoordinates, volumeDimensions );
            deviceTag[ arrayIndexInTagVolume ] = 1;
        }
    }
}

__global__ void FilterDuplicatesFilterKernel( CudaCompactElement* deviceData,
                                              CudaCompactElement* deviceKeep,
                                              CudaTagElement*     deviceTag,
                                              size_t              numElements,
                                              dim3                volumeDimensions )
{
    int arrayIndexInActiveElementTexture = ComputeIndexThread1DBlock1DTo1D();
    int arrayIndexInActiveElementList    = arrayIndexInActiveElementTexture * 4;

    if ( arrayIndexInActiveElementList < numElements )
    {
        CudaCompactElement4 keepElements = tex1Dfetch( CUDA_TEXTURE_REF_VALID_ELEMENTS_1D, arrayIndexInActiveElementTexture );

        if ( keepElements.x == 1 || keepElements.y == 1 || keepElements.z == 1 || keepElements.w == 1 )
        {
            CudaCompactElement4 packedCoordinates = tex1Dfetch( CUDA_TEXTURE_REF_ACTIVE_ELEMENTS_1D, arrayIndexInActiveElementTexture );
        
            FilterDuplicatesFilterHelper( keepElements.x, packedCoordinates.x, arrayIndexInActiveElementList, volumeDimensions, deviceTag, deviceKeep );
            arrayIndexInActiveElementList++;

            if ( arrayIndexInActiveElementList < numElements )
            {
                FilterDuplicatesFilterHelper( keepElements.y, packedCoordinates.y, arrayIndexInActiveElementList, volumeDimensions, deviceTag, deviceKeep );
                arrayIndexInActiveElementList++;

                if ( arrayIndexInActiveElementList < numElements )
                {
                    FilterDuplicatesFilterHelper( keepElements.z, packedCoordinates.z, arrayIndexInActiveElementList, volumeDimensions, deviceTag, deviceKeep );
                    arrayIndexInActiveElementList++;

                    if ( arrayIndexInActiveElementList < numElements )
                    {
                        FilterDuplicatesFilterHelper( keepElements.w, packedCoordinates.w, arrayIndexInActiveElementList, volumeDimensions, deviceTag, deviceKeep );
                    }
                }
            }
        }
    }
}

#endif