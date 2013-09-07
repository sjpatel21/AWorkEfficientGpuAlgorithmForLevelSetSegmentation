#ifndef SRC_CUDA_INITIALIZE_ACTIVE_ELEMENTS_VOLUME_CU
#define SRC_CUDA_INITIALIZE_ACTIVE_ELEMENTS_VOLUME_CU

__global__ void  InitializeActiveElementsVolumeKernel( CudaCompactElement* keepElementsVolume,
                                                       dim3                volumeDimensions,
                                                       int                 logicalNumThreadBlocksX,
                                                       float               invNumThreadBlocksX );

extern "C" void CudaInitializeActiveElementsVolume( CudaCompactElement* keepElementsVolume, dim3 volumeDimensions )
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
    InitializeActiveElementsVolumeKernel<<< gridDimensions, threadBlockDimensions >>>( keepElementsVolume, volumeDimensions, logicalNumThreadBlocksX, invNumThreadBlocksX );

    CudaSynchronize();
    CudaCheckErrors();
}

__global__ void InitializeActiveElementsVolumeKernel( CudaCompactElement* keepElementsVolume,
                                                      dim3                volumeDimensions,
                                                      int                 logicalNumThreadBlocksX,
                                                      float               invNumThreadBlocksX )
{
    const dim3         elementCoordinates = ComputeIndexThread3DBlock2DTo3D( volumeDimensions, logicalNumThreadBlocksX, invNumThreadBlocksX );
    CudaCompactElement keepElement;
    dim3               currentCoordinatesDim3;
    int                arrayIndex;

    GET_LEVEL_SET_NEIGHBORHOOD_HELPER_1D( u4negativeZ, currentCoordinatesDim3, arrayIndex, elementCoordinates, volumeDimensions, +, 0, +, 0, -, 1 );
    GET_LEVEL_SET_NEIGHBORHOOD_HELPER_1D( u1,          currentCoordinatesDim3, arrayIndex, elementCoordinates, volumeDimensions, +, 0, -, 1, +, 0 );
    GET_LEVEL_SET_NEIGHBORHOOD_HELPER_1D( u3,          currentCoordinatesDim3, arrayIndex, elementCoordinates, volumeDimensions, -, 1, +, 0, +, 0 );
    GET_LEVEL_SET_NEIGHBORHOOD_HELPER_1D( u4,          currentCoordinatesDim3, arrayIndex, elementCoordinates, volumeDimensions, +, 0, +, 0, +, 0 );
    GET_LEVEL_SET_NEIGHBORHOOD_HELPER_1D( u5,          currentCoordinatesDim3, arrayIndex, elementCoordinates, volumeDimensions, +, 1, +, 0, +, 0 );
    GET_LEVEL_SET_NEIGHBORHOOD_HELPER_1D( u7,          currentCoordinatesDim3, arrayIndex, elementCoordinates, volumeDimensions, +, 0, +, 1, +, 0 );
    GET_LEVEL_SET_NEIGHBORHOOD_HELPER_1D( u4positiveZ, currentCoordinatesDim3, arrayIndex, elementCoordinates, volumeDimensions, +, 0, +, 0, +, 1 );

    //
    // if gradient is 0
    //
    if ( Equals( u4, u4negativeZ, SPATIAL_DERIVATIVE_THRESHOLD ) &&
         Equals( u4, u1,          SPATIAL_DERIVATIVE_THRESHOLD ) &&
         Equals( u4, u3,          SPATIAL_DERIVATIVE_THRESHOLD ) &&
         Equals( u4, u5,          SPATIAL_DERIVATIVE_THRESHOLD ) &&
         Equals( u4, u7,          SPATIAL_DERIVATIVE_THRESHOLD ) &&
         Equals( u4, u4positiveZ, SPATIAL_DERIVATIVE_THRESHOLD ) )      
    {
        keepElement = 0;
    }
    else
    {
        keepElement = 1;
    }

    arrayIndex                       = ComputeIndex3DToTiled1D( elementCoordinates, volumeDimensions );
    keepElementsVolume[ arrayIndex ] = keepElement;
}




#ifdef COMPUTE_PERFORMANCE_METRICS


__device__ void ComputeActiveElementOutputsUntiled( dim3  elementCoordinates,
                                                    dim3  volumeDimensions,
                                                    float tolerance,
                                                    bool& outputU4negativeZ,
                                                    bool& outputU1,
                                                    bool& outputU3,
                                                    bool& outputU4,
                                                    bool& outputU5,
                                                    bool& outputU7,
                                                    bool& outputU4positiveZ )
{
    dim3 currentCoordinatesDim3;
    int  arrayIndex;

    outputU4negativeZ = false;
    outputU1          = false;
    outputU3          = false;
    outputU4          = false;
    outputU5          = false;
    outputU7          = false;
    outputU4positiveZ = false;

#ifdef TEMPORAL_DERIVATIVE_VOXEL_CULLING
    CudaTagElement u4timeDerivativeScratch;

#ifndef SIX_CONNECTED_VOXEL_CULLING
    CudaTagElement u0timeDerivativeScratch;
    CudaTagElement u1timeDerivativeScratch;
    CudaTagElement u2timeDerivativeScratch;
    CudaTagElement u3timeDerivativeScratch;
    CudaTagElement u5timeDerivativeScratch;
    CudaTagElement u6timeDerivativeScratch;
    CudaTagElement u7timeDerivativeScratch;
    CudaTagElement u8timeDerivativeScratch;
#endif

#endif

    float u4Scratch;
    float uOtherScratch;

    GET_LEVEL_SET_NEIGHBORHOOD_HELPER_1D_PREDEFINE_VARIABLE( U4, currentCoordinatesDim3, arrayIndex, elementCoordinates, volumeDimensions, +, 0, +, 0, +, 0 );

#ifdef TEMPORAL_DERIVATIVE_VOXEL_CULLING

    GET_TAG_NEIGHBORHOOD_HELPER_1D_PREDEFINE_VARIABLE( U4_TIME_DERIVATIVE, currentCoordinatesDim3, arrayIndex, elementCoordinates, volumeDimensions, +, 0, +, 0, +, 0 );

    if ( U4_TIME_DERIVATIVE == 1 )
    {

#endif

        GET_LEVEL_SET_NEIGHBORHOOD_HELPER_1D_PREDEFINE_VARIABLE( U4_NEGATIVE_Z, currentCoordinatesDim3, arrayIndex, elementCoordinates, volumeDimensions, +, 0, +, 0, -, 1 );
        if ( !Equals( U4, U4_NEGATIVE_Z, tolerance ) ) { outputU4 = true; outputU4negativeZ = true; }

        GET_LEVEL_SET_NEIGHBORHOOD_HELPER_1D_PREDEFINE_VARIABLE( U1, currentCoordinatesDim3, arrayIndex, elementCoordinates, volumeDimensions, +, 0, -, 1, +, 0 );
        if ( !Equals( U4, U1, tolerance ) ) { outputU4 = true; outputU1 = true; }

        GET_LEVEL_SET_NEIGHBORHOOD_HELPER_1D_PREDEFINE_VARIABLE( U3, currentCoordinatesDim3, arrayIndex, elementCoordinates, volumeDimensions, -, 1, +, 0, +, 0 );
        if ( !Equals( U4, U3, tolerance ) ) { outputU4 = true; outputU3 = true; }

        GET_LEVEL_SET_NEIGHBORHOOD_HELPER_1D_PREDEFINE_VARIABLE( U5, currentCoordinatesDim3, arrayIndex, elementCoordinates, volumeDimensions, +, 1, +, 0, +, 0 );
        if ( !Equals( U4, U5, tolerance ) ) { outputU4 = true; outputU5 = true; }

        GET_LEVEL_SET_NEIGHBORHOOD_HELPER_1D_PREDEFINE_VARIABLE( U7, currentCoordinatesDim3, arrayIndex, elementCoordinates, volumeDimensions, +, 0, +, 1, +, 0 );
        if ( !Equals( U4, U7, tolerance ) ) { outputU4 = true; outputU7 = true; }

        GET_LEVEL_SET_NEIGHBORHOOD_HELPER_1D_PREDEFINE_VARIABLE( U4_POSITIVE_Z, currentCoordinatesDim3, arrayIndex, elementCoordinates, volumeDimensions, +, 0, +, 0, +, 1 );
        if ( !Equals( U4, U4_POSITIVE_Z, tolerance ) ) { outputU4 = true; outputU4positiveZ = true; }

#ifdef TEMPORAL_DERIVATIVE_VOXEL_CULLING
    }
#ifndef SIX_CONNECTED_VOXEL_CULLING
    else
    {
        GET_TAG_NEIGHBORHOOD_HELPER_1D_PREDEFINE_VARIABLE( U1_TIME_DERIVATIVE_NEGATIVE_Z, currentCoordinatesDim3, arrayIndex, elementCoordinates, volumeDimensions, +, 0, -, 1, -, 1 );
        GET_TAG_NEIGHBORHOOD_HELPER_1D_PREDEFINE_VARIABLE( U3_TIME_DERIVATIVE_NEGATIVE_Z, currentCoordinatesDim3, arrayIndex, elementCoordinates, volumeDimensions, -, 1, +, 0, -, 1 );
        GET_TAG_NEIGHBORHOOD_HELPER_1D_PREDEFINE_VARIABLE( U4_TIME_DERIVATIVE_NEGATIVE_Z, currentCoordinatesDim3, arrayIndex, elementCoordinates, volumeDimensions, +, 0, +, 0, -, 1 );
        GET_TAG_NEIGHBORHOOD_HELPER_1D_PREDEFINE_VARIABLE( U5_TIME_DERIVATIVE_NEGATIVE_Z, currentCoordinatesDim3, arrayIndex, elementCoordinates, volumeDimensions, +, 1, +, 0, -, 1 );
        GET_TAG_NEIGHBORHOOD_HELPER_1D_PREDEFINE_VARIABLE( U7_TIME_DERIVATIVE_NEGATIVE_Z, currentCoordinatesDim3, arrayIndex, elementCoordinates, volumeDimensions, +, 0, +, 1, -, 1 );

        //
        // testing -z layer of time derivative neighborhood
        //
        if ( U1_TIME_DERIVATIVE_NEGATIVE_Z == 1 ||
             U3_TIME_DERIVATIVE_NEGATIVE_Z == 1 ||
             U4_TIME_DERIVATIVE_NEGATIVE_Z == 1 ||
             U5_TIME_DERIVATIVE_NEGATIVE_Z == 1 ||
             U7_TIME_DERIVATIVE_NEGATIVE_Z == 1 )
        {
            GET_LEVEL_SET_NEIGHBORHOOD_HELPER_1D_PREDEFINE_VARIABLE( U4_NEGATIVE_Z, currentCoordinatesDim3, arrayIndex, elementCoordinates, volumeDimensions, +, 0, +, 0, -, 1 );

            if ( !Equals( U4, U4_NEGATIVE_Z, tolerance ) ) { outputU4 = true; outputU4negativeZ = true; }
        }

        if ( U1_TIME_DERIVATIVE_NEGATIVE_Z == 1 )
        {
            GET_LEVEL_SET_NEIGHBORHOOD_HELPER_1D_PREDEFINE_VARIABLE( U1, currentCoordinatesDim3, arrayIndex, elementCoordinates, volumeDimensions, +, 0, -, 1, +, 0 );

            if ( !Equals( U4, U1, tolerance ) ) { outputU4 = true; outputU1 = true; }
        }

        if ( U3_TIME_DERIVATIVE_NEGATIVE_Z == 1 )
        {
            GET_LEVEL_SET_NEIGHBORHOOD_HELPER_1D_PREDEFINE_VARIABLE( U3, currentCoordinatesDim3, arrayIndex, elementCoordinates, volumeDimensions, -, 1, +, 0, +, 0 );

            if ( !Equals( U4, U3, tolerance ) ) { outputU4 = true; outputU3 = true; }

        }

        if ( U5_TIME_DERIVATIVE_NEGATIVE_Z == 1 )
        {
            GET_LEVEL_SET_NEIGHBORHOOD_HELPER_1D_PREDEFINE_VARIABLE( U5, currentCoordinatesDim3, arrayIndex, elementCoordinates, volumeDimensions, +, 1, +, 0, +, 0 );

            if ( !Equals( U4, U5, tolerance ) ) { outputU4 = true; outputU5 = true; }
        }


        if ( U7_TIME_DERIVATIVE_NEGATIVE_Z == 1 )
        {
            GET_LEVEL_SET_NEIGHBORHOOD_HELPER_1D_PREDEFINE_VARIABLE( U7, currentCoordinatesDim3, arrayIndex, elementCoordinates, volumeDimensions, +, 0, +, 1, +, 0 );

            if ( !Equals( U4, U7, tolerance ) ) { outputU4 = true; outputU7 = true; }
        }

        GET_TAG_NEIGHBORHOOD_HELPER_1D_PREDEFINE_VARIABLE( U0_TIME_DERIVATIVE, currentCoordinatesDim3, arrayIndex, elementCoordinates, volumeDimensions, -, 1, -, 1, +, 0 );
        GET_TAG_NEIGHBORHOOD_HELPER_1D_PREDEFINE_VARIABLE( U1_TIME_DERIVATIVE, currentCoordinatesDim3, arrayIndex, elementCoordinates, volumeDimensions, +, 0, -, 1, +, 0 );
        GET_TAG_NEIGHBORHOOD_HELPER_1D_PREDEFINE_VARIABLE( U2_TIME_DERIVATIVE, currentCoordinatesDim3, arrayIndex, elementCoordinates, volumeDimensions, +, 1, -, 1, +, 0 );
        GET_TAG_NEIGHBORHOOD_HELPER_1D_PREDEFINE_VARIABLE( U3_TIME_DERIVATIVE, currentCoordinatesDim3, arrayIndex, elementCoordinates, volumeDimensions, -, 1, +, 0, +, 0 );
                                                           
        GET_TAG_NEIGHBORHOOD_HELPER_1D_PREDEFINE_VARIABLE( U5_TIME_DERIVATIVE, currentCoordinatesDim3, arrayIndex, elementCoordinates, volumeDimensions, +, 1, +, 0, +, 0 );
        GET_TAG_NEIGHBORHOOD_HELPER_1D_PREDEFINE_VARIABLE( U6_TIME_DERIVATIVE, currentCoordinatesDim3, arrayIndex, elementCoordinates, volumeDimensions, -, 1, +, 1, +, 0 );
        GET_TAG_NEIGHBORHOOD_HELPER_1D_PREDEFINE_VARIABLE( U7_TIME_DERIVATIVE, currentCoordinatesDim3, arrayIndex, elementCoordinates, volumeDimensions, +, 0, +, 1, +, 0 );
        GET_TAG_NEIGHBORHOOD_HELPER_1D_PREDEFINE_VARIABLE( U8_TIME_DERIVATIVE, currentCoordinatesDim3, arrayIndex, elementCoordinates, volumeDimensions, +, 1, +, 1, +, 0 );

        //
        // testing middle layer of time derivative neighborhood
        //
        if ( U0_TIME_DERIVATIVE == 1 ||
             U1_TIME_DERIVATIVE == 1 ||
             U2_TIME_DERIVATIVE == 1 )
        {
            GET_LEVEL_SET_NEIGHBORHOOD_HELPER_1D_PREDEFINE_VARIABLE( U1, currentCoordinatesDim3, arrayIndex, elementCoordinates, volumeDimensions, +, 0, -, 1, +, 0 );

            if ( !Equals( U4, U1, tolerance ) ) { outputU4 = true; outputU1 = true; }
        }


        if ( U0_TIME_DERIVATIVE == 1 ||
             U3_TIME_DERIVATIVE == 1 ||
             U6_TIME_DERIVATIVE == 1 )
        {
            GET_LEVEL_SET_NEIGHBORHOOD_HELPER_1D_PREDEFINE_VARIABLE( U3, currentCoordinatesDim3, arrayIndex, elementCoordinates, volumeDimensions, -, 1, +, 0, +, 0 );

            if ( !Equals( U4, U3, tolerance ) ) { outputU4 = true; outputU3 = true; }
        }

        if ( U2_TIME_DERIVATIVE == 1 ||
             U5_TIME_DERIVATIVE == 1 ||
             U8_TIME_DERIVATIVE == 1 )
        {
            GET_LEVEL_SET_NEIGHBORHOOD_HELPER_1D_PREDEFINE_VARIABLE( U5, currentCoordinatesDim3, arrayIndex, elementCoordinates, volumeDimensions, +, 1, +, 0, +, 0 );

            if ( !Equals( U4, U5, tolerance ) ) { outputU4 = true; outputU5 = true; }
        }

        if ( U6_TIME_DERIVATIVE == 1 ||
             U7_TIME_DERIVATIVE == 1 || 
             U8_TIME_DERIVATIVE == 1 )
        {
            GET_LEVEL_SET_NEIGHBORHOOD_HELPER_1D_PREDEFINE_VARIABLE( U7, currentCoordinatesDim3, arrayIndex, elementCoordinates, volumeDimensions, +, 0, +, 1, +, 0 );

            if ( !Equals( U4, U7, tolerance ) ) { outputU4 = true; outputU7 = true; }
        }

        GET_TAG_NEIGHBORHOOD_HELPER_1D_PREDEFINE_VARIABLE( U1_TIME_DERIVATIVE_POSITIVE_Z, currentCoordinatesDim3, arrayIndex, elementCoordinates, volumeDimensions, +, 0, -, 1, +, 1 );
        GET_TAG_NEIGHBORHOOD_HELPER_1D_PREDEFINE_VARIABLE( U3_TIME_DERIVATIVE_POSITIVE_Z, currentCoordinatesDim3, arrayIndex, elementCoordinates, volumeDimensions, -, 1, +, 0, +, 1 );
        GET_TAG_NEIGHBORHOOD_HELPER_1D_PREDEFINE_VARIABLE( U4_TIME_DERIVATIVE_POSITIVE_Z, currentCoordinatesDim3, arrayIndex, elementCoordinates, volumeDimensions, +, 0, +, 0, +, 1 );
        GET_TAG_NEIGHBORHOOD_HELPER_1D_PREDEFINE_VARIABLE( U5_TIME_DERIVATIVE_POSITIVE_Z, currentCoordinatesDim3, arrayIndex, elementCoordinates, volumeDimensions, +, 1, +, 0, +, 1 );
        GET_TAG_NEIGHBORHOOD_HELPER_1D_PREDEFINE_VARIABLE( U7_TIME_DERIVATIVE_POSITIVE_Z, currentCoordinatesDim3, arrayIndex, elementCoordinates, volumeDimensions, +, 0, +, 1, +, 1 );

        //
        // testing +z layer of time derivative neighborhood
        //
        if ( U1_TIME_DERIVATIVE_POSITIVE_Z == 1 ||
             U3_TIME_DERIVATIVE_POSITIVE_Z == 1 ||
             U4_TIME_DERIVATIVE_POSITIVE_Z == 1 ||
             U5_TIME_DERIVATIVE_POSITIVE_Z == 1 ||
             U7_TIME_DERIVATIVE_POSITIVE_Z == 1 )
        {
            GET_LEVEL_SET_NEIGHBORHOOD_HELPER_1D_PREDEFINE_VARIABLE( U4_POSITIVE_Z, currentCoordinatesDim3, arrayIndex, elementCoordinates, volumeDimensions, +, 0, +, 0, +, 1 );

            if ( !Equals( U4, U4_POSITIVE_Z, tolerance ) ) { outputU4 = true; outputU4positiveZ = true; }
        }

        if ( U1_TIME_DERIVATIVE_POSITIVE_Z == 1 )
        {
            GET_LEVEL_SET_NEIGHBORHOOD_HELPER_1D_PREDEFINE_VARIABLE( U1, currentCoordinatesDim3, arrayIndex, elementCoordinates, volumeDimensions, +, 0, -, 1, +, 0 );

            if ( !Equals( U4, U1, tolerance ) ) { outputU4 = true; outputU1 = true; }
        }


        if ( U3_TIME_DERIVATIVE_POSITIVE_Z == 1 )
        {
            GET_LEVEL_SET_NEIGHBORHOOD_HELPER_1D_PREDEFINE_VARIABLE( U3, currentCoordinatesDim3, arrayIndex, elementCoordinates, volumeDimensions, -, 1, +, 0, +, 0 );

            if ( !Equals( U4, U3, tolerance ) ) { outputU4 = true; outputU3 = true; }
        }

        if ( U5_TIME_DERIVATIVE_POSITIVE_Z == 1 )
        {
            GET_LEVEL_SET_NEIGHBORHOOD_HELPER_1D_PREDEFINE_VARIABLE( U5, currentCoordinatesDim3, arrayIndex, elementCoordinates, volumeDimensions, +, 1, +, 0, +, 0 );

            if ( !Equals( U4, U5, tolerance ) ) { outputU4 = true; outputU5 = true; }
        }


        if ( U7_TIME_DERIVATIVE_POSITIVE_Z == 1 )
        {
            GET_LEVEL_SET_NEIGHBORHOOD_HELPER_1D_PREDEFINE_VARIABLE( U7, currentCoordinatesDim3, arrayIndex, elementCoordinates, volumeDimensions, +, 0, +, 1, +, 0 );

            if ( !Equals( U4, U7, tolerance ) ) { outputU4 = true; outputU7 = true; }
        }
    }
#endif
#endif
}


template< typename T >
__global__ void  InitializeActiveElementsVolumeConditionalMemoryWriteKernel( CudaLevelSetElement* keepElementsVolume1, CudaLevelSetElement* keepElementsVolume2, CudaTagElement* tagElementsVolume,
                                                                             dim3                 volumeDimensions,
                                                                             int                  logicalNumThreadBlocksX,
                                                                             float                invNumThreadBlocksX );

extern "C" void CudaInitializeActiveElementsVolumeConditionalMemoryWrite( CudaLevelSetElement*  keepElementsVolume1, CudaLevelSetElement* keepElementsVolume2, CudaTagElement* tagElementsVolume, dim3 volumeDimensions )
{
    // set the thread block size to the maximum
    dim3 threadBlockDimensions( 1, 16, 16 );

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
    InitializeActiveElementsVolumeConditionalMemoryWriteKernel< unsigned char ><<< gridDimensions, threadBlockDimensions >>>( keepElementsVolume1, keepElementsVolume2, tagElementsVolume, volumeDimensions, logicalNumThreadBlocksX, invNumThreadBlocksX );

    CudaSynchronize();
    CudaCheckErrors();
}

template< typename T >
__global__ void InitializeActiveElementsVolumeConditionalMemoryWriteKernel( CudaLevelSetElement* keepElementsVolume1, CudaLevelSetElement*  keepElementsVolume2, CudaTagElement* tagElementsVolume,
                                                                            dim3                 volumeDimensions,
                                                                            int                  logicalNumThreadBlocksX,
                                                                            float                invNumThreadBlocksX )
{
    const dim3         elementCoordinates = ComputeIndexThread3DBlock2DTo3D( volumeDimensions, logicalNumThreadBlocksX, invNumThreadBlocksX );
    dim3               currentCoordinatesDim3;

    bool  outputU4negativeZ;
    bool  outputU1;
    bool  outputU3;
    bool  outputU4;
    bool  outputU5;
    bool  outputU7;
    bool  outputU4positiveZ;

    float tolerance  = TEMPORAL_DERIVATIVE_THRESHOLD;

    ComputeActiveElementOutputs( elementCoordinates,
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
            //
            // get 3x3x3 neighborhood
            //
            int3 currentCoordinates;
            dim3 currentCoordinatesDim3;
            int  arrayIndex;
            int  elementIndex;
            int  elementIndexUntiled;

            elementIndex        = ComputeIndex3DToTiled1D( elementCoordinates, volumeDimensions );
            elementIndexUntiled = ComputeIndex3DTo1D( elementCoordinates, volumeDimensions );

            GET_LEVEL_SET_NEIGHBORHOOD_HELPER_1D( u0nZ, currentCoordinatesDim3, arrayIndex, elementCoordinates, volumeDimensions, -, 1, +, 1, -, 1 );
            GET_LEVEL_SET_NEIGHBORHOOD_HELPER_1D( u1nZ, currentCoordinatesDim3, arrayIndex, elementCoordinates, volumeDimensions, +, 0, +, 1, -, 1 );
            GET_LEVEL_SET_NEIGHBORHOOD_HELPER_1D( u2nZ, currentCoordinatesDim3, arrayIndex, elementCoordinates, volumeDimensions, +, 1, +, 1, -, 1 );
            GET_LEVEL_SET_NEIGHBORHOOD_HELPER_1D( u3nZ, currentCoordinatesDim3, arrayIndex, elementCoordinates, volumeDimensions, -, 1, +, 0, -, 1 );
            GET_LEVEL_SET_NEIGHBORHOOD_HELPER_1D( u4negativeZ, currentCoordinatesDim3, arrayIndex, elementCoordinates, volumeDimensions, +, 0, +, 0, -, 1 );
            GET_LEVEL_SET_NEIGHBORHOOD_HELPER_1D( u5nZ, currentCoordinatesDim3, arrayIndex, elementCoordinates, volumeDimensions, +, 1, +, 0, -, 1 );
            GET_LEVEL_SET_NEIGHBORHOOD_HELPER_1D( u6nZ, currentCoordinatesDim3, arrayIndex, elementCoordinates, volumeDimensions, -, 1, -, 1, -, 1 );
            GET_LEVEL_SET_NEIGHBORHOOD_HELPER_1D( u7nZ, currentCoordinatesDim3, arrayIndex, elementCoordinates, volumeDimensions, +, 0, -, 1, -, 1 );
            GET_LEVEL_SET_NEIGHBORHOOD_HELPER_1D( u8nZ, currentCoordinatesDim3, arrayIndex, elementCoordinates, volumeDimensions, +, 1, -, 1, -, 1 );
                                                                         
            GET_LEVEL_SET_NEIGHBORHOOD_HELPER_1D( u0, currentCoordinatesDim3, arrayIndex, elementCoordinates, volumeDimensions, -, 1, +, 1, +, 0 );
            GET_LEVEL_SET_NEIGHBORHOOD_HELPER_1D( u1, currentCoordinatesDim3, arrayIndex, elementCoordinates, volumeDimensions, +, 0, +, 1, +, 0 );
            GET_LEVEL_SET_NEIGHBORHOOD_HELPER_1D( u2, currentCoordinatesDim3, arrayIndex, elementCoordinates, volumeDimensions, +, 1, +, 1, +, 0 );
            GET_LEVEL_SET_NEIGHBORHOOD_HELPER_1D( u3, currentCoordinatesDim3, arrayIndex, elementCoordinates, volumeDimensions, -, 1, +, 0, +, 0 );
            GET_LEVEL_SET_NEIGHBORHOOD_HELPER_1D( u4, currentCoordinatesDim3, arrayIndex, elementCoordinates, volumeDimensions, +, 0, +, 0, +, 0 );
            GET_LEVEL_SET_NEIGHBORHOOD_HELPER_1D( u5, currentCoordinatesDim3, arrayIndex, elementCoordinates, volumeDimensions, +, 1, +, 0, +, 0 );
            GET_LEVEL_SET_NEIGHBORHOOD_HELPER_1D( u6, currentCoordinatesDim3, arrayIndex, elementCoordinates, volumeDimensions, -, 1, -, 1, +, 0 );
            GET_LEVEL_SET_NEIGHBORHOOD_HELPER_1D( u7, currentCoordinatesDim3, arrayIndex, elementCoordinates, volumeDimensions, +, 0, -, 1, +, 0 );
            GET_LEVEL_SET_NEIGHBORHOOD_HELPER_1D( u8, currentCoordinatesDim3, arrayIndex, elementCoordinates, volumeDimensions, +, 1, -, 1, +, 0 );
                                                                         
            GET_LEVEL_SET_NEIGHBORHOOD_HELPER_1D( u0pZ, currentCoordinatesDim3, arrayIndex, elementCoordinates, volumeDimensions, -, 1, +, 1, +, 1 );
            GET_LEVEL_SET_NEIGHBORHOOD_HELPER_1D( u1pZ, currentCoordinatesDim3, arrayIndex, elementCoordinates, volumeDimensions, +, 0, +, 1, +, 1 );
            GET_LEVEL_SET_NEIGHBORHOOD_HELPER_1D( u2pZ, currentCoordinatesDim3, arrayIndex, elementCoordinates, volumeDimensions, +, 1, +, 1, +, 1 );
            GET_LEVEL_SET_NEIGHBORHOOD_HELPER_1D( u3pZ, currentCoordinatesDim3, arrayIndex, elementCoordinates, volumeDimensions, -, 1, +, 0, +, 1 );
            GET_LEVEL_SET_NEIGHBORHOOD_HELPER_1D( u4positiveZ, currentCoordinatesDim3, arrayIndex, elementCoordinates, volumeDimensions, +, 0, +, 0, +, 1 );
            GET_LEVEL_SET_NEIGHBORHOOD_HELPER_1D( u5pZ, currentCoordinatesDim3, arrayIndex, elementCoordinates, volumeDimensions, +, 1, +, 0, +, 1 );
            GET_LEVEL_SET_NEIGHBORHOOD_HELPER_1D( u6pZ, currentCoordinatesDim3, arrayIndex, elementCoordinates, volumeDimensions, -, 1, -, 1, +, 1 );
            GET_LEVEL_SET_NEIGHBORHOOD_HELPER_1D( u7pZ, currentCoordinatesDim3, arrayIndex, elementCoordinates, volumeDimensions, +, 0, -, 1, +, 1 );
            GET_LEVEL_SET_NEIGHBORHOOD_HELPER_1D( u8pZ, currentCoordinatesDim3, arrayIndex, elementCoordinates, volumeDimensions, +, 1, -, 1, +, 1 );

            GET_SOURCE_NEIGHBORHOOD_HELPER( u4Source, currentCoordinates, elementCoordinates, volumeDimensions, +, 0, +, 0, +, 0, T );

            // compute derivatives
            float derivativeCentralX          = ( u5 - u3 )                   / 2.0f;
            float derivativeCentralY          = ( u7 - u1 )                   / 2.0f;
            float derivativeCentralZ          = ( u4positiveZ - u4negativeZ ) / 2.0f; 

            float derivativeForwardX          = ( u5          - u4 );
            float derivativeForwardY          = ( u7          - u4 );
            float derivativeForwardZ          = ( u4positiveZ - u4 );

            float derivativeBackwardX         = ( u4 - u3 );
            float derivativeBackwardY         = ( u4 - u1 );
            float derivativeBackwardZ         = ( u4 - u4negativeZ );

            float secondDerivativeX = derivativeForwardX - derivativeBackwardX;
            float secondDerivativeY = derivativeForwardY - derivativeBackwardY;
            float secondDerivativeZ = derivativeForwardZ - derivativeBackwardZ;

            // compute speed function terms
            float connectivityTerm = 0.0f;

            if ( Equals( u4positiveZ, u4negativeZ, 0.1f ) &&            
                 Equals( u4negativeZ, u1,          0.1f ) &&
                 Equals( u1, u3,                   0.1f ) &&
                 Equals( u3, u5,                   0.1f ) &&
                 Equals( u5, u7,                   0.1f ) &&
                 Equals( u7, u4positiveZ,          0.1f ) )
            {
                float average = 0.0f;
                
                average  += u0nZ;
                average  += u1nZ;
                average  += u2nZ;
                average  += u3nZ;
                average  += u4negativeZ;
                average  += u5nZ;
                average  += u6nZ;
                average  += u7nZ;
                average  += u8nZ;
                    
                average  += u0;
                average  += u1; 
                average  += u2; 
                average  += u3; 
                average  += u4; 
                average  += u5; 
                average  += u6; 
                average  += u7; 
                average  += u8; 
                    
                average  += u0pZ;
                average  += u1pZ;
                average  += u2pZ;
                average  += u3pZ;
                average  += u4positiveZ;
                average  += u5pZ;
                average  += u6pZ;
                average  += u7pZ;
                average  += u8pZ;

                average  /= 27.0f;

                connectivityTerm = average - u4;
            }

            float curvatureInfluence = 0.15f;
            float timeStep           = 0.1f;

            float curvatureTerm  = ( secondDerivativeX + secondDerivativeY + secondDerivativeZ );
            float densityTerm    = - ComputeDensityTerm( 100, u4Source, 100 );
            float speedFunction  = ( ( 1 - curvatureInfluence ) * ( densityTerm ) ) + ( ( curvatureInfluence ) * curvatureTerm );

            // gradient magnitude
            float gradientLength = sqrt( Sqr( derivativeCentralX ) + Sqr( derivativeCentralY ) + Sqr( derivativeCentralZ ) );

            // compute new value
            float levelSetDelta    = ( ( timeStep * speedFunction * gradientLength ) ) + connectivityTerm;
            float newLevelSetValue = min( 1.0f, max( -1.0f, u4 + levelSetDelta ) );

            // rescale
            newLevelSetValue = min( 1.0f, max( -1.0f, ( ( 1.01f * newLevelSetValue ) + 0.01f ) ) );

            // assign new value
#ifdef LEVEL_SET_FIELD_FIXED_POINT
            int writeValue                            = __float2int_rd( newLevelSetValue * LEVEL_SET_FIELD_MAX_VALUE );
            keepElementsVolume1[ elementIndex ]        = writeValue;
            keepElementsVolume2[ elementIndexUntiled ] = writeValue;
#else
            keepElementsVolume1[ elementIndex ]        = newLevelSetValue;
            keepElementsVolume2[ elementIndexUntiled ] = newLevelSetValue;
#endif

            // compute and assign time derivative
            float timeDerivative = newLevelSetValue - u4;

            int timeDerivativeValue           = abs( timeDerivative ) > TEMPORAL_DERIVATIVE_THRESHOLD ? 1 : 0;
            tagElementsVolume[ elementIndex ] = timeDerivativeValue;

    }
}


template< typename T >
__global__ void  InitializeActiveElementsVolumeUnconditionalMemoryWriteKernel( CudaLevelSetElement* keepElementsVolume1, CudaLevelSetElement* keepElementsVolume2, CudaTagElement* tagElementsVolume,
                                                                             dim3                 volumeDimensions,
                                                                             int                  logicalNumThreadBlocksX,
                                                                             float                invNumThreadBlocksX );

extern "C" void CudaInitializeActiveElementsVolumeUnconditionalMemoryWrite( CudaLevelSetElement*  keepElementsVolume1, CudaLevelSetElement* keepElementsVolume2, CudaTagElement* tagElementsVolume, dim3 volumeDimensions )
{
    // set the thread block size to the maximum
    dim3 threadBlockDimensions( 1, 16, 16 );

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
    InitializeActiveElementsVolumeUnconditionalMemoryWriteKernel< unsigned char ><<< gridDimensions, threadBlockDimensions >>>( keepElementsVolume1, keepElementsVolume2, tagElementsVolume, volumeDimensions, logicalNumThreadBlocksX, invNumThreadBlocksX );

    CudaSynchronize();
    CudaCheckErrors();
}

template< typename T >
__global__ void InitializeActiveElementsVolumeUnconditionalMemoryWriteKernel( CudaLevelSetElement* keepElementsVolume1, CudaLevelSetElement*  keepElementsVolume2, CudaTagElement* tagElementsVolume,
                                                                              dim3                 volumeDimensions,
                                                                              int                  logicalNumThreadBlocksX,
                                                                              float                invNumThreadBlocksX )
{
    const dim3         elementCoordinates = ComputeIndexThread3DBlock2DTo3D( volumeDimensions, logicalNumThreadBlocksX, invNumThreadBlocksX );
    
    //
    // get 3x3x3 neighborhood
    //
    int3 currentCoordinates;
    dim3 currentCoordinatesDim3;
    int  arrayIndex;
    int  elementIndex;
    int  elementIndexUntiled;

    elementIndex        = ComputeIndex3DToTiled1D( elementCoordinates, volumeDimensions );
    elementIndexUntiled = ComputeIndex3DTo1D( elementCoordinates, volumeDimensions );

    GET_LEVEL_SET_NEIGHBORHOOD_HELPER_1D( u0nZ, currentCoordinatesDim3, arrayIndex, elementCoordinates, volumeDimensions, -, 1, +, 1, -, 1 );
    GET_LEVEL_SET_NEIGHBORHOOD_HELPER_1D( u1nZ, currentCoordinatesDim3, arrayIndex, elementCoordinates, volumeDimensions, +, 0, +, 1, -, 1 );
    GET_LEVEL_SET_NEIGHBORHOOD_HELPER_1D( u2nZ, currentCoordinatesDim3, arrayIndex, elementCoordinates, volumeDimensions, +, 1, +, 1, -, 1 );
    GET_LEVEL_SET_NEIGHBORHOOD_HELPER_1D( u3nZ, currentCoordinatesDim3, arrayIndex, elementCoordinates, volumeDimensions, -, 1, +, 0, -, 1 );
    GET_LEVEL_SET_NEIGHBORHOOD_HELPER_1D( u4negativeZ, currentCoordinatesDim3, arrayIndex, elementCoordinates, volumeDimensions, +, 0, +, 0, -, 1 );
    GET_LEVEL_SET_NEIGHBORHOOD_HELPER_1D( u5nZ, currentCoordinatesDim3, arrayIndex, elementCoordinates, volumeDimensions, +, 1, +, 0, -, 1 );
    GET_LEVEL_SET_NEIGHBORHOOD_HELPER_1D( u6nZ, currentCoordinatesDim3, arrayIndex, elementCoordinates, volumeDimensions, -, 1, -, 1, -, 1 );
    GET_LEVEL_SET_NEIGHBORHOOD_HELPER_1D( u7nZ, currentCoordinatesDim3, arrayIndex, elementCoordinates, volumeDimensions, +, 0, -, 1, -, 1 );
    GET_LEVEL_SET_NEIGHBORHOOD_HELPER_1D( u8nZ, currentCoordinatesDim3, arrayIndex, elementCoordinates, volumeDimensions, +, 1, -, 1, -, 1 );
                                                                 
    GET_LEVEL_SET_NEIGHBORHOOD_HELPER_1D( u0, currentCoordinatesDim3, arrayIndex, elementCoordinates, volumeDimensions, -, 1, +, 1, +, 0 );
    GET_LEVEL_SET_NEIGHBORHOOD_HELPER_1D( u1, currentCoordinatesDim3, arrayIndex, elementCoordinates, volumeDimensions, +, 0, +, 1, +, 0 );
    GET_LEVEL_SET_NEIGHBORHOOD_HELPER_1D( u2, currentCoordinatesDim3, arrayIndex, elementCoordinates, volumeDimensions, +, 1, +, 1, +, 0 );
    GET_LEVEL_SET_NEIGHBORHOOD_HELPER_1D( u3, currentCoordinatesDim3, arrayIndex, elementCoordinates, volumeDimensions, -, 1, +, 0, +, 0 );
    GET_LEVEL_SET_NEIGHBORHOOD_HELPER_1D( u4, currentCoordinatesDim3, arrayIndex, elementCoordinates, volumeDimensions, +, 0, +, 0, +, 0 );
    GET_LEVEL_SET_NEIGHBORHOOD_HELPER_1D( u5, currentCoordinatesDim3, arrayIndex, elementCoordinates, volumeDimensions, +, 1, +, 0, +, 0 );
    GET_LEVEL_SET_NEIGHBORHOOD_HELPER_1D( u6, currentCoordinatesDim3, arrayIndex, elementCoordinates, volumeDimensions, -, 1, -, 1, +, 0 );
    GET_LEVEL_SET_NEIGHBORHOOD_HELPER_1D( u7, currentCoordinatesDim3, arrayIndex, elementCoordinates, volumeDimensions, +, 0, -, 1, +, 0 );
    GET_LEVEL_SET_NEIGHBORHOOD_HELPER_1D( u8, currentCoordinatesDim3, arrayIndex, elementCoordinates, volumeDimensions, +, 1, -, 1, +, 0 );
                                                                 
    GET_LEVEL_SET_NEIGHBORHOOD_HELPER_1D( u0pZ, currentCoordinatesDim3, arrayIndex, elementCoordinates, volumeDimensions, -, 1, +, 1, +, 1 );
    GET_LEVEL_SET_NEIGHBORHOOD_HELPER_1D( u1pZ, currentCoordinatesDim3, arrayIndex, elementCoordinates, volumeDimensions, +, 0, +, 1, +, 1 );
    GET_LEVEL_SET_NEIGHBORHOOD_HELPER_1D( u2pZ, currentCoordinatesDim3, arrayIndex, elementCoordinates, volumeDimensions, +, 1, +, 1, +, 1 );
    GET_LEVEL_SET_NEIGHBORHOOD_HELPER_1D( u3pZ, currentCoordinatesDim3, arrayIndex, elementCoordinates, volumeDimensions, -, 1, +, 0, +, 1 );
    GET_LEVEL_SET_NEIGHBORHOOD_HELPER_1D( u4positiveZ, currentCoordinatesDim3, arrayIndex, elementCoordinates, volumeDimensions, +, 0, +, 0, +, 1 );
    GET_LEVEL_SET_NEIGHBORHOOD_HELPER_1D( u5pZ, currentCoordinatesDim3, arrayIndex, elementCoordinates, volumeDimensions, +, 1, +, 0, +, 1 );
    GET_LEVEL_SET_NEIGHBORHOOD_HELPER_1D( u6pZ, currentCoordinatesDim3, arrayIndex, elementCoordinates, volumeDimensions, -, 1, -, 1, +, 1 );
    GET_LEVEL_SET_NEIGHBORHOOD_HELPER_1D( u7pZ, currentCoordinatesDim3, arrayIndex, elementCoordinates, volumeDimensions, +, 0, -, 1, +, 1 );
    GET_LEVEL_SET_NEIGHBORHOOD_HELPER_1D( u8pZ, currentCoordinatesDim3, arrayIndex, elementCoordinates, volumeDimensions, +, 1, -, 1, +, 1 );

    GET_SOURCE_NEIGHBORHOOD_HELPER( u4Source, currentCoordinates, elementCoordinates, volumeDimensions, +, 0, +, 0, +, 0, T );

    // compute derivatives
    float derivativeCentralX          = ( u5 - u3 )                   / 2.0f;
    float derivativeCentralY          = ( u7 - u1 )                   / 2.0f;
    float derivativeCentralZ          = ( u4positiveZ - u4negativeZ ) / 2.0f; 

    float derivativeForwardX          = ( u5          - u4 );
    float derivativeForwardY          = ( u7          - u4 );
    float derivativeForwardZ          = ( u4positiveZ - u4 );

    float derivativeBackwardX         = ( u4 - u3 );
    float derivativeBackwardY         = ( u4 - u1 );
    float derivativeBackwardZ         = ( u4 - u4negativeZ );

    float secondDerivativeX = derivativeForwardX - derivativeBackwardX;
    float secondDerivativeY = derivativeForwardY - derivativeBackwardY;
    float secondDerivativeZ = derivativeForwardZ - derivativeBackwardZ;

    // compute speed function terms
    float connectivityTerm = 0.0f;

    if ( Equals( u4positiveZ, u4negativeZ, 0.1f ) &&            
         Equals( u4negativeZ, u1,          0.1f ) &&
         Equals( u1, u3,                   0.1f ) &&
         Equals( u3, u5,                   0.1f ) &&
         Equals( u5, u7,                   0.1f ) &&
         Equals( u7, u4positiveZ,          0.1f ) )
    {
        float average = 0.0f;
        
        average  += u0nZ;
        average  += u1nZ;
        average  += u2nZ;
        average  += u3nZ;
        average  += u4negativeZ;
        average  += u5nZ;
        average  += u6nZ;
        average  += u7nZ;
        average  += u8nZ;
            
        average  += u0;
        average  += u1; 
        average  += u2; 
        average  += u3; 
        average  += u4; 
        average  += u5; 
        average  += u6; 
        average  += u7; 
        average  += u8; 
            
        average  += u0pZ;
        average  += u1pZ;
        average  += u2pZ;
        average  += u3pZ;
        average  += u4positiveZ;
        average  += u5pZ;
        average  += u6pZ;
        average  += u7pZ;
        average  += u8pZ;

        average  /= 27.0f;

        connectivityTerm = average - u4;
    }

    float curvatureInfluence = 0.15f;
    float timeStep           = 0.1f;

    float curvatureTerm  = ( secondDerivativeX + secondDerivativeY + secondDerivativeZ );
    float densityTerm    = - ComputeDensityTerm( 100, u4Source, 100 );
    float speedFunction  = ( ( 1 - curvatureInfluence ) * ( densityTerm ) ) + ( ( curvatureInfluence ) * curvatureTerm );

    // gradient magnitude
    float gradientLength = sqrt( Sqr( derivativeCentralX ) + Sqr( derivativeCentralY ) + Sqr( derivativeCentralZ ) );

    // compute new value
    float levelSetDelta    = ( ( timeStep * speedFunction * gradientLength ) ) + connectivityTerm;
    float newLevelSetValue = min( 1.0f, max( -1.0f, u4 + levelSetDelta ) );

    // rescale
    newLevelSetValue = min( 1.0f, max( -1.0f, ( ( 1.01f * newLevelSetValue ) + 0.01f ) ) );

    // assign new value
#ifdef LEVEL_SET_FIELD_FIXED_POINT
    int writeValue                            = __float2int_rd( newLevelSetValue * LEVEL_SET_FIELD_MAX_VALUE );
    keepElementsVolume1[ elementIndex ]        = writeValue;
    keepElementsVolume2[ elementIndexUntiled ] = writeValue;
#else
    keepElementsVolume1[ elementIndex ]        = newLevelSetValue;
    keepElementsVolume2[ elementIndexUntiled ] = newLevelSetValue;
#endif
}


#endif


#endif