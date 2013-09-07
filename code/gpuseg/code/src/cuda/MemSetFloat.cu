__global__ void MemSetFloatKernel( float* deviceData,
                                   float  valueToSet,
                                   int    numElements );

extern "C" void CudaMemSetFloat( float* deviceData,
                                 float  valueToSet,
                                 int    numElements )
{
    CudaMemSetFloatAsync( deviceData, valueToSet, numElements );
    CudaSynchronize();
    CudaCheckErrors();
}

extern "C" void CudaMemSetFloatAsync( float* deviceData,
                                      float  valueToSet,
                                      int    numElements )
{
    unsigned int maxNumElements       = 256 * 256 * 256;
    unsigned int numFullPasses        = numElements / maxNumElements;
    unsigned int numLeftoverElements  = numElements % maxNumElements;
    bool         doLeftoverPass       = numLeftoverElements != 0;

    unsigned int currentIndex = 0;

    for ( unsigned int i = 0; i < numFullPasses; i++ )
    {
        dim3 threadBlockDimensions( 512, 1, 1 );
        int  numThreadBlocks = static_cast< int >( ceil( maxNumElements / 512.0f ) );
        dim3 gridDimensions( numThreadBlocks, 1, 1 );

        MemSetFloatKernel<<< gridDimensions, threadBlockDimensions >>>( deviceData + currentIndex, valueToSet, maxNumElements );

        currentIndex += maxNumElements;
    }

    if ( doLeftoverPass )
    {
        dim3 threadBlockDimensions( 512, 1, 1 );
        int  numThreadBlocks = static_cast< int >( ceil( numLeftoverElements / 512.0f ) );
        dim3 gridDimensions( numThreadBlocks, 1, 1 );

        MemSetFloatKernel<<< gridDimensions, threadBlockDimensions >>>( deviceData + currentIndex, valueToSet, numLeftoverElements );
    }
}

__global__ void MemSetFloatKernel( float* deviceData,
                                   float  valueToSet,
                                   int    numElements )
{
    int arrayIndex = ComputeIndexThread1DBlock1DTo1D();

    if ( arrayIndex < numElements )
    {
        deviceData[ arrayIndex ] = valueToSet;
    }
}










__global__ void MemSetIntKernel( int* deviceData,
                                 int  valueToSet,
                                 int  numElements );

extern "C" void CudaMemSetInt( int* deviceData,
                               int  valueToSet,
                               int  numElements )
{
    CudaMemSetIntAsync( deviceData, valueToSet, numElements );
    CudaSynchronize();
    CudaCheckErrors();
}

extern "C" void CudaMemSetIntAsync( int* deviceData,
                                    int  valueToSet,
                                    int  numElements )
{
    unsigned int maxNumElements       = 256 * 256 * 256;
    unsigned int numFullPasses        = numElements / maxNumElements;
    unsigned int numLeftoverElements  = numElements % maxNumElements;
    bool         doLeftoverPass       = numLeftoverElements != 0;

    unsigned int currentIndex = 0;

    for ( unsigned int i = 0; i < numFullPasses; i++ )
    {
        dim3 threadBlockDimensions( 512, 1, 1 );
        int  numThreadBlocks = static_cast< int >( ceil( maxNumElements / 512.0f ) );
        dim3 gridDimensions( numThreadBlocks, 1, 1 );

        MemSetIntKernel<<< gridDimensions, threadBlockDimensions >>>( deviceData + currentIndex, valueToSet, maxNumElements );

        currentIndex += maxNumElements;
    }

    if ( doLeftoverPass )
    {
        dim3 threadBlockDimensions( 512, 1, 1 );
        int  numThreadBlocks = static_cast< int >( ceil( numLeftoverElements / 512.0f ) );
        dim3 gridDimensions( numThreadBlocks, 1, 1 );

        MemSetIntKernel<<< gridDimensions, threadBlockDimensions >>>( deviceData + currentIndex, valueToSet, numLeftoverElements );
    }
}

__global__ void MemSetIntKernel( int* deviceData,
                                 int  valueToSet,
                                 int  numElements )
{
    int arrayIndex = ComputeIndexThread1DBlock1DTo1D();

    if ( arrayIndex < numElements )
    {
        deviceData[ arrayIndex ] = valueToSet;
    }
}





template< typename T >
__global__ void MemSetSparseKernel( T*     deviceData,
                                    T      valueToSet,
                                    size_t numElements,
                                    dim3   volumeDimensions );

extern "C" void CudaMemSetCharSparse( unsigned char* deviceData,
                                      unsigned char  valueToSet,
                                      size_t         numElements,
                                      dim3           volumeDimensions )
{
        CudaMemSetCharSparseAsync( deviceData,
                                   valueToSet,
                                   numElements,
                                   volumeDimensions );

        CudaSynchronize();
        CudaCheckErrors();
}

extern "C" void CudaMemSetCharSparseAsync( unsigned char* deviceData,
                                           unsigned char  valueToSet,
                                           size_t         numElements,
                                           dim3           volumeDimensions )
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
        // call our kernel
        MemSetSparseKernel< unsigned char ><<< gridDimensions, threadBlockDimensions >>>(
            deviceData,
            valueToSet,
            numElements,
            volumeDimensions );
    }
}




extern "C" void CudaMemSetIntSparse( unsigned int* deviceData,
                                     unsigned int  valueToSet,
                                      size_t       numElements,
                                      dim3         volumeDimensions )
{
        CudaMemSetIntSparseAsync( deviceData,
                                  valueToSet,
                                  numElements,
                                  volumeDimensions );

        CudaSynchronize();
        CudaCheckErrors();
}

extern "C" void CudaMemSetIntSparseAsync( unsigned int* deviceData,
                                          unsigned int  valueToSet,
                                          size_t        numElements,
                                          dim3          volumeDimensions )
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
        // call our kernel
        MemSetSparseKernel< unsigned int ><<< gridDimensions, threadBlockDimensions >>>(
            deviceData,
            valueToSet,
            numElements,
            volumeDimensions );
    }
}




template< typename T >
__device__ void MemSetSparseKernelHelper( CudaCompactElement packedVoxelCoordinate,
                                          T*                 deviceData,
                                          T                  valueToSet,
                                          dim3               volumeDimensions )
{
    dim3 elementCoordinates    = UnpackCoordinates( packedVoxelCoordinate );
    int  elementIndex          = ComputeIndex3DToTiled1D( elementCoordinates, volumeDimensions );
    deviceData[ elementIndex ] = valueToSet;
}

template< typename T >
__global__ void MemSetSparseKernel( T*     deviceData,
                                    T      valueToSet,
                                    size_t numElements,
                                    dim3   volumeDimensions )
{
    int arrayIndexInActiveElementTexture = ComputeIndexThread1DBlock1DTo1D();
    int arrayIndexInActiveElementList    = arrayIndexInActiveElementTexture * 4;

    if ( arrayIndexInActiveElementList < numElements )
    {
        CudaCompactElement4 packedCoordinates = tex1Dfetch( CUDA_TEXTURE_REF_ACTIVE_ELEMENTS_1D, arrayIndexInActiveElementTexture );

        MemSetSparseKernelHelper< T >(
            packedCoordinates.x,
            deviceData,
            valueToSet,
            volumeDimensions );
       
        arrayIndexInActiveElementList++;

        if ( arrayIndexInActiveElementList < numElements )
        {
            MemSetSparseKernelHelper< T >(
                packedCoordinates.y,
                deviceData,
                valueToSet,
                volumeDimensions );
            
            arrayIndexInActiveElementList++;

            if ( arrayIndexInActiveElementList < numElements )
            {
                MemSetSparseKernelHelper< T >(
                    packedCoordinates.z,
                    deviceData,
                    valueToSet,
                    volumeDimensions );
          
                arrayIndexInActiveElementList++;

                if ( arrayIndexInActiveElementList < numElements )
                {
                    MemSetSparseKernelHelper< T >(
                        packedCoordinates.w,
                        deviceData,
                        valueToSet,
                        volumeDimensions );
                }
            }
        }
    }
}