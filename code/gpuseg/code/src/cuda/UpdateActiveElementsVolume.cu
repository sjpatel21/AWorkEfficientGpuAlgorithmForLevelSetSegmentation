#ifndef SRC_CUDA_UPDATE_ACTIVE_ELEMENTS_VOLUME_CU
#define SRC_CUDA_UPDATE_ACTIVE_ELEMENTS_VOLUME_CU

__global__ void  UpdateActiveElementsVolumeKernel( CudaCompactElement*  newActiveElementVolume,
                                                   dim3                 volumeDimensions,
                                                   size_t               oldNumActiveElements );

extern "C" void CudaUpdateActiveElementsVolume( CudaCompactElement* newActiveElementVolume,
                                                size_t              oldNumActiveElements,
                                                dim3                volumeDimensions )
{
    // set the thread block size to the maximum
#ifdef CUDA_ARCH_SM_10   
    dim3 threadBlockDimensions( 128, 1, 1 );
    int numThreadBlocks = static_cast< int >( ceil( oldNumActiveElements / ( 128.0f * 4.0f ) ) );
#endif

#ifdef CUDA_ARCH_SM_13
    dim3 threadBlockDimensions( 256, 1, 1 );
    int numThreadBlocks = static_cast< int >( ceil( oldNumActiveElements / ( 256.0f * 4.0f ) ) );
#endif

    // set the grid dimensions
    dim3 gridDimensions( numThreadBlocks, 1, 1 );

    if ( numThreadBlocks > 0 )
    {
        // call our kernel
        UpdateActiveElementsVolumeKernel<<< gridDimensions, threadBlockDimensions >>>(
            newActiveElementVolume,
            volumeDimensions,
            oldNumActiveElements );

        CudaSynchronize();
        CudaCheckErrors();
    }
}

__device__ void UpdateActiveElementsVolumeHelper( CudaCompactElement packedOldVoxelCoordinate, dim3 volumeDimensions, CudaCompactElement* deviceNewKeepElementsData )
{
    dim3 oldElementCoordinates = UnpackCoordinates( packedOldVoxelCoordinate );

    bool  outputU4negativeZ;
    bool  outputU1;
    bool  outputU3;
    bool  outputU4;
    bool  outputU5;
    bool  outputU7;
    bool  outputU4positiveZ;

    float tolerance = TEMPORAL_DERIVATIVE_THRESHOLD;

    ComputeActiveElementOutputs(
         oldElementCoordinates,
         volumeDimensions,
         tolerance,
         outputU4negativeZ,
         outputU1,
         outputU3,
         outputU4,
         outputU5,
         outputU7,
         outputU4positiveZ );

    if ( outputU4 ) { int i = ComputeIndex3DToTiled1D( oldElementCoordinates, volumeDimensions ); deviceNewKeepElementsData[ i ] = 1; }

    if ( outputU1 )          { int i = ComputeIndex3DToTiled1D( OffsetCoordinates( oldElementCoordinates,  0, -1,  0, volumeDimensions ), volumeDimensions ); deviceNewKeepElementsData[ i ] = 1; }
    if ( outputU3 )          { int i = ComputeIndex3DToTiled1D( OffsetCoordinates( oldElementCoordinates, -1,  0,  0, volumeDimensions ), volumeDimensions ); deviceNewKeepElementsData[ i ] = 1; }
    if ( outputU5 )          { int i = ComputeIndex3DToTiled1D( OffsetCoordinates( oldElementCoordinates, +1,  0,  0, volumeDimensions ), volumeDimensions ); deviceNewKeepElementsData[ i ] = 1; }
    if ( outputU7 )          { int i = ComputeIndex3DToTiled1D( OffsetCoordinates( oldElementCoordinates,  0, +1,  0, volumeDimensions ), volumeDimensions ); deviceNewKeepElementsData[ i ] = 1; }
    if ( outputU4negativeZ ) { int i = ComputeIndex3DToTiled1D( OffsetCoordinates( oldElementCoordinates,  0,  0, -1, volumeDimensions ), volumeDimensions ); deviceNewKeepElementsData[ i ] = 1; }
    if ( outputU4positiveZ ) { int i = ComputeIndex3DToTiled1D( OffsetCoordinates( oldElementCoordinates,  0,  0, +1, volumeDimensions ), volumeDimensions ); deviceNewKeepElementsData[ i ] = 1; }
}

__global__ void UpdateActiveElementsVolumeKernel( CudaCompactElement* deviceNewKeepElementsData,
                                                  dim3                volumeDimensions,
                                                  size_t              numActiveVoxels )
{
    int arrayIndexInActiveElementTexture = ComputeIndexThread1DBlock1DTo1D();
    int arrayIndexInActiveElementList    = arrayIndexInActiveElementTexture * 4;

    for ( unsigned int i = 0; i < 3; i++ )
    {
        if ( arrayIndexInActiveElementList < numActiveVoxels )
        {
            CudaCompactElement4 packedCoordinates = tex1Dfetch( CUDA_TEXTURE_REF_ACTIVE_ELEMENTS_1D, arrayIndexInActiveElementTexture );

            CudaCompactElement  packedCoordinate;

            if ( i == 0 ) packedCoordinate = packedCoordinates.x;
            if ( i == 1 ) packedCoordinate = packedCoordinates.y;
            if ( i == 2 ) packedCoordinate = packedCoordinates.z;
            if ( i == 3 ) packedCoordinate = packedCoordinates.w;

            UpdateActiveElementsVolumeHelper( packedCoordinate, volumeDimensions, deviceNewKeepElementsData );
        }
        
        arrayIndexInActiveElementList++;
    }
}









__global__ void  OutputNewActiveElementsKernel( CudaCompactElement*  u4negativeZOutputList,
                                                CudaCompactElement*  u1OutputList,         
                                                CudaCompactElement*  u3OutputList,         
                                                CudaCompactElement*  u4OutputList,         
                                                CudaCompactElement*  u5OutputList,         
                                                CudaCompactElement*  u7OutputList,         
                                                CudaCompactElement*  u4positiveZOutputList,

                                                CudaCompactElement*  u4negativeZValidList,
                                                CudaCompactElement*  u1ValidList,
                                                CudaCompactElement*  u3ValidList,
                                                CudaCompactElement*  u4ValidList,
                                                CudaCompactElement*  u5ValidList,
                                                CudaCompactElement*  u7ValidList,
                                                CudaCompactElement*  u4positiveZValidList,

                                                dim3                 volumeDimensions,
                                                size_t               oldNumActiveElements );

extern "C" void CudaOutputNewActiveElements( CudaCompactElement* newActiveElementList,
                                             CudaCompactElement* newValidElementList,
                                             size_t              oldNumActiveElements,
                                             dim3                volumeDimensions )
{
    // set the thread block size to the maximum
#ifdef SIX_CONNECTED_VOXEL_CULLING

#ifdef CUDA_ARCH_SM_10
    dim3 threadBlockDimensions( 256, 1, 1 );
    int numThreadBlocks = static_cast< int >( ceil( oldNumActiveElements / ( 256.0f ) ) );
#endif

#ifdef CUDA_ARCH_SM_13
    dim3 threadBlockDimensions( 512, 1, 1 );
    int numThreadBlocks = static_cast< int >( ceil( oldNumActiveElements / ( 512.0f ) ) );
#endif

#else

#ifdef CUDA_ARCH_SM_10
    dim3 threadBlockDimensions( 128, 1, 1 );
    int numThreadBlocks = static_cast< int >( ceil( oldNumActiveElements / ( 128.0f ) ) );
#endif

#ifdef CUDA_ARCH_SM_13
    dim3 threadBlockDimensions( 256, 1, 1 );
    int numThreadBlocks = static_cast< int >( ceil( oldNumActiveElements / ( 256.0f ) ) );
#endif

#endif

    // set the grid dimensions
    dim3 gridDimensions( numThreadBlocks, 1, 1 );

    if ( numThreadBlocks > 0 )
    {
        unsigned int numElementsWarpAligned = CudaGetWarpAlignedValue( oldNumActiveElements );

        CudaCompactElement* u4negativeZOutputList = newActiveElementList + ( numElementsWarpAligned * 0 );
        CudaCompactElement* u1OutputList          = newActiveElementList + ( numElementsWarpAligned * 1 );
        CudaCompactElement* u3OutputList          = newActiveElementList + ( numElementsWarpAligned * 2 );
        CudaCompactElement* u4OutputList          = newActiveElementList + ( numElementsWarpAligned * 3 );
        CudaCompactElement* u5OutputList          = newActiveElementList + ( numElementsWarpAligned * 4 );
        CudaCompactElement* u7OutputList          = newActiveElementList + ( numElementsWarpAligned * 5 );
        CudaCompactElement* u4positiveZOutputList = newActiveElementList + ( numElementsWarpAligned * 6 );

        CudaCompactElement* u4negativeZValidList = newValidElementList + ( numElementsWarpAligned * 0 );
        CudaCompactElement* u1ValidList          = newValidElementList + ( numElementsWarpAligned * 1 );
        CudaCompactElement* u3ValidList          = newValidElementList + ( numElementsWarpAligned * 2 );
        CudaCompactElement* u4ValidList          = newValidElementList + ( numElementsWarpAligned * 3 );
        CudaCompactElement* u5ValidList          = newValidElementList + ( numElementsWarpAligned * 4 );
        CudaCompactElement* u7ValidList          = newValidElementList + ( numElementsWarpAligned * 5 );
        CudaCompactElement* u4positiveZValidList = newValidElementList + ( numElementsWarpAligned * 6 );

        // call our kernel
        OutputNewActiveElementsKernel<<< gridDimensions, threadBlockDimensions >>>(
            u4negativeZOutputList,
            u1OutputList,         
            u3OutputList,         
            u4OutputList,         
            u5OutputList,         
            u7OutputList,         
            u4positiveZOutputList,

            u4negativeZValidList,
            u1ValidList,
            u3ValidList,
            u4ValidList,
            u5ValidList,
            u7ValidList,
            u4positiveZValidList,

            volumeDimensions,
            oldNumActiveElements );

        CudaSynchronize();
        CudaCheckErrors();
    }
}

__device__ void  OutputNewActiveElementsHelper( CudaCompactElement   packedOldVoxelCoordinate,
                                                int                  arrayIndexInOldActiveElementList,
                                                CudaCompactElement*  u4negativeZOutputList,
                                                CudaCompactElement*  u1OutputList,         
                                                CudaCompactElement*  u3OutputList,         
                                                CudaCompactElement*  u4OutputList,         
                                                CudaCompactElement*  u5OutputList,         
                                                CudaCompactElement*  u7OutputList,         
                                                CudaCompactElement*  u4positiveZOutputList,

                                                CudaCompactElement*  u4negativeZValidList,
                                                CudaCompactElement*  u1ValidList,
                                                CudaCompactElement*  u3ValidList,
                                                CudaCompactElement*  u4ValidList,
                                                CudaCompactElement*  u5ValidList,
                                                CudaCompactElement*  u7ValidList,
                                                CudaCompactElement*  u4positiveZValidList,

                                                dim3                 volumeDimensions )
{
    dim3 oldElementCoordinates = UnpackCoordinates( packedOldVoxelCoordinate );

    bool  outputU4negativeZ;
    bool  outputU1;
    bool  outputU3;
    bool  outputU4;
    bool  outputU5;
    bool  outputU7;
    bool  outputU4positiveZ;

    float tolerance  = TEMPORAL_DERIVATIVE_THRESHOLD;

    ComputeActiveElementOutputs( oldElementCoordinates,
                                 volumeDimensions,
                                 tolerance,
                                 outputU4negativeZ,
                                 outputU1,
                                 outputU3,
                                 outputU4,
                                 outputU5,
                                 outputU7,
                                 outputU4positiveZ );

    if ( outputU4 )
    {
        int packedOutputCoordinates                      = PackCoordinates( oldElementCoordinates, 0, 0,  0, volumeDimensions );
        u4OutputList[ arrayIndexInOldActiveElementList ] = packedOutputCoordinates;
        u4ValidList[ arrayIndexInOldActiveElementList ]  = 1;
    }

    if ( outputU1 )
    {
        int packedOutputCoordinates                      = PackCoordinates( oldElementCoordinates,  0, -1,  0, volumeDimensions );
        u1OutputList[ arrayIndexInOldActiveElementList ] = packedOutputCoordinates;
        u1ValidList[ arrayIndexInOldActiveElementList ]  = 1;
    }

    if ( outputU3 )
    {
        int packedOutputCoordinates                      = PackCoordinates( oldElementCoordinates, -1,  0,  0, volumeDimensions );
        u3OutputList[ arrayIndexInOldActiveElementList ] = packedOutputCoordinates;
        u3ValidList[ arrayIndexInOldActiveElementList ]  = 1;
    }

    if ( outputU5 )
    {
        int packedOutputCoordinates                      = PackCoordinates( oldElementCoordinates, +1,  0,  0, volumeDimensions );
        u5OutputList[ arrayIndexInOldActiveElementList ] = packedOutputCoordinates;
        u5ValidList[ arrayIndexInOldActiveElementList ]  = 1;

    }

    if ( outputU7 )
    {
        int packedOutputCoordinates                      = PackCoordinates( oldElementCoordinates,  0, +1,  0, volumeDimensions );
        u7OutputList[ arrayIndexInOldActiveElementList ] = packedOutputCoordinates;
        u7ValidList[ arrayIndexInOldActiveElementList ]  = 1;
    }

    if ( outputU4negativeZ )
    {
        int packedOutputCoordinates                               = PackCoordinates( oldElementCoordinates,  0,  0, -1, volumeDimensions );
        u4negativeZOutputList[ arrayIndexInOldActiveElementList ] = packedOutputCoordinates;
        u4negativeZValidList[ arrayIndexInOldActiveElementList ]  = 1;
    }

    if ( outputU4positiveZ )
    {
        int packedOutputCoordinates                               = PackCoordinates( oldElementCoordinates,  0,  0, +1, volumeDimensions );
        u4positiveZOutputList[ arrayIndexInOldActiveElementList ] = packedOutputCoordinates;
        u4positiveZValidList[ arrayIndexInOldActiveElementList ]  = 1;
    }
}

__global__ void  OutputNewActiveElementsKernel( CudaCompactElement*  u4negativeZOutputList,
                                                CudaCompactElement*  u1OutputList,         
                                                CudaCompactElement*  u3OutputList,         
                                                CudaCompactElement*  u4OutputList,         
                                                CudaCompactElement*  u5OutputList,         
                                                CudaCompactElement*  u7OutputList,         
                                                CudaCompactElement*  u4positiveZOutputList,

                                                CudaCompactElement*  u4negativeZValidList,
                                                CudaCompactElement*  u1ValidList,
                                                CudaCompactElement*  u3ValidList,
                                                CudaCompactElement*  u4ValidList,
                                                CudaCompactElement*  u5ValidList,
                                                CudaCompactElement*  u7ValidList,
                                                CudaCompactElement*  u4positiveZValidList,

                                                dim3                 volumeDimensions,
                                                size_t               oldNumActiveElements )
{
    int arrayIndexInActiveElementTexture = ComputeIndexThread1DBlock1DTo1D();
    int arrayIndexInActiveElementList    = arrayIndexInActiveElementTexture * 4;

    CudaCompactElement4 packedCoordinates = tex1Dfetch( CUDA_TEXTURE_REF_ACTIVE_ELEMENTS_1D, arrayIndexInActiveElementTexture );
    CudaCompactElement  packedCoordinate;

    for ( unsigned int i = 0; i < 3; i++ )
    {
        if ( arrayIndexInActiveElementList < oldNumActiveElements )
        {
            if ( i == 0 ) packedCoordinate = packedCoordinates.x;
            if ( i == 1 ) packedCoordinate = packedCoordinates.y;
            if ( i == 2 ) packedCoordinate = packedCoordinates.z;
            if ( i == 3 ) packedCoordinate = packedCoordinates.w;

            OutputNewActiveElementsHelper(
                packedCoordinate,
                arrayIndexInActiveElementList,

                u4negativeZOutputList,
                u1OutputList,         
                u3OutputList,         
                u4OutputList,         
                u5OutputList,         
                u7OutputList,         
                u4positiveZOutputList,

                u4negativeZValidList,
                u1ValidList,
                u3ValidList,
                u4ValidList,
                u5ValidList,
                u7ValidList,
                u4positiveZValidList,

                volumeDimensions );

            arrayIndexInActiveElementList++;
        }
    }
}

#endif