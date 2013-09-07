#ifndef SRC_CUDA_UPDATE_ACTIVE_ELEMENTS_CU
#define SRC_CUDA_UPDATE_ACTIVE_ELEMENTS_CU

#define U4_NEGATIVE_Z                 uOtherScratch
#define U1                            uOtherScratch
#define U3                            uOtherScratch
#define U4                            u4Scratch
#define U5                            uOtherScratch
#define U7                            uOtherScratch
#define U4_POSITIVE_Z                 uOtherScratch

#define U1_TIME_DERIVATIVE_NEGATIVE_Z u1timeDerivativeScratch
#define U3_TIME_DERIVATIVE_NEGATIVE_Z u3timeDerivativeScratch
#define U4_TIME_DERIVATIVE_NEGATIVE_Z u4timeDerivativeScratch
#define U5_TIME_DERIVATIVE_NEGATIVE_Z u5timeDerivativeScratch
#define U7_TIME_DERIVATIVE_NEGATIVE_Z u7timeDerivativeScratch

#define U0_TIME_DERIVATIVE            u0timeDerivativeScratch
#define U1_TIME_DERIVATIVE            u1timeDerivativeScratch
#define U2_TIME_DERIVATIVE            u2timeDerivativeScratch
#define U3_TIME_DERIVATIVE            u3timeDerivativeScratch
#define U4_TIME_DERIVATIVE            u4timeDerivativeScratch
#define U5_TIME_DERIVATIVE            u5timeDerivativeScratch
#define U6_TIME_DERIVATIVE            u6timeDerivativeScratch
#define U7_TIME_DERIVATIVE            u7timeDerivativeScratch
#define U8_TIME_DERIVATIVE            u8timeDerivativeScratch

#define U1_TIME_DERIVATIVE_POSITIVE_Z u1timeDerivativeScratch
#define U3_TIME_DERIVATIVE_POSITIVE_Z u3timeDerivativeScratch
#define U4_TIME_DERIVATIVE_POSITIVE_Z u4timeDerivativeScratch
#define U5_TIME_DERIVATIVE_POSITIVE_Z u5timeDerivativeScratch
#define U7_TIME_DERIVATIVE_POSITIVE_Z u7timeDerivativeScratch

__device__ void ComputeActiveElementOutputs( dim3  elementCoordinates,
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

#endif