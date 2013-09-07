#ifndef SRC_CUDA_UPDATE_ACTIVE_ELEMENT_COUNT_LIST_CU
#define SRC_CUDA_UPDATE_ACTIVE_ELEMENT_COUNT_LIST_CU

__global__ void  UpdateActiveElementOutputCountListKernel( CudaCompactElement*  oldActiveElementList,
                                                           CudaCompactElement*  newActiveElementCountList,
                                                           dim3                 volumeDimensions,
                                                           size_t               oldNumActiveElements );

extern "C" void CudaUpdateActiveElementOutputCountList( CudaCompactElement* oldActiveElementList,
                                                        CudaCompactElement* newActiveElementCountList,
                                                        size_t              oldNumActiveElements,
                                                        dim3                volumeDimensions )
{
    // set the thread block size to the maximum
#ifdef CUDA_ARCH_SM_10   
    dim3 threadBlockDimensions( 128, 1, 1 );
    int numThreadBlocks = static_cast< int >( ceil( oldNumActiveElements / 128.0f ) );
#endif

#ifdef CUDA_ARCH_SM_13
    dim3 threadBlockDimensions( 512, 1, 1 );
    int numThreadBlocks = static_cast< int >( ceil( oldNumActiveElements / 512.0f ) );
#endif

    // set the grid dimensions
    dim3 gridDimensions( numThreadBlocks, 1, 1 );

    if ( numThreadBlocks > 0 )
    {
        // call our kernel
        UpdateActiveElementOutputCountListKernel<<< gridDimensions, threadBlockDimensions >>>(
            oldActiveElementList,
            newActiveElementCountList,
            volumeDimensions,
            oldNumActiveElements );

        CudaSynchronize();
        CudaCheckErrors();
    }
}

__global__ void UpdateActiveElementOutputCountListKernel( CudaCompactElement* oldActiveElementList,
                                                          CudaCompactElement* newActiveElementCountList,
                                                          dim3                volumeDimensions,
                                                          size_t              numActiveVoxels )
{
    int arrayIndexInOldActiveElementList  = ComputeElementCoordinates1D();

    if ( arrayIndexInOldActiveElementList < numActiveVoxels )
    {
        CudaCompactElement packedOldVoxelCoordinate = oldActiveElementList[ arrayIndexInOldActiveElementList ];
        dim3 oldElementCoordinates                  = UnpackCoordinates( packedOldVoxelCoordinate );

        int3 currentCoordinates;
        dim3 currentCoordinatesDim3;
        int  arrayIndex;

        GET_LEVEL_SET_NEIGHBORHOOD_HELPER_1D( u4negativeZ, currentCoordinates, currentCoordinatesDim3, arrayIndex, oldElementCoordinates, volumeDimensions, +, 0, +, 0, -, 1 );
        GET_LEVEL_SET_NEIGHBORHOOD_HELPER_1D( u1,          currentCoordinates, currentCoordinatesDim3, arrayIndex, oldElementCoordinates, volumeDimensions, +, 0, -, 1, +, 0 );
        GET_LEVEL_SET_NEIGHBORHOOD_HELPER_1D( u3,          currentCoordinates, currentCoordinatesDim3, arrayIndex, oldElementCoordinates, volumeDimensions, -, 1, +, 0, +, 0 );
        GET_LEVEL_SET_NEIGHBORHOOD_HELPER_1D( u4,          currentCoordinates, currentCoordinatesDim3, arrayIndex, oldElementCoordinates, volumeDimensions, +, 0, +, 0, +, 0 );
        GET_LEVEL_SET_NEIGHBORHOOD_HELPER_1D( u5,          currentCoordinates, currentCoordinatesDim3, arrayIndex, oldElementCoordinates, volumeDimensions, +, 1, +, 0, +, 0 );
        GET_LEVEL_SET_NEIGHBORHOOD_HELPER_1D( u7,          currentCoordinates, currentCoordinatesDim3, arrayIndex, oldElementCoordinates, volumeDimensions, +, 0, +, 1, +, 0 );
        GET_LEVEL_SET_NEIGHBORHOOD_HELPER_1D( u4positiveZ, currentCoordinates, currentCoordinatesDim3, arrayIndex, oldElementCoordinates, volumeDimensions, +, 0, +, 0, +, 1 );

        GET_TAG_NEIGHBORHOOD_HELPER_1D( u4timeDerivative, currentCoordinates, currentCoordinatesDim3, arrayIndex, oldElementCoordinates, volumeDimensions, +, 0, +, 0, +, 0 );

        bool  outputSelf = false;
        float tolerance  = 0.001f;
        int   numOutputs = 0;

#ifdef TEMPORAL_VOXEL_CULLING
        if ( u4timeDerivative == 1 )
        {
#endif

            if ( !Equals( u4, u1,          tolerance ) ) { outputSelf = true; numOutputs++; }
            if ( !Equals( u4, u3,          tolerance ) ) { outputSelf = true; numOutputs++; }
            if ( !Equals( u4, u5,          tolerance ) ) { outputSelf = true; numOutputs++; }
            if ( !Equals( u4, u7,          tolerance ) ) { outputSelf = true; numOutputs++; }
            if ( !Equals( u4, u4negativeZ, tolerance ) ) { outputSelf = true; numOutputs++; }
            if ( !Equals( u4, u4positiveZ, tolerance ) ) { outputSelf = true; numOutputs++; }

#ifdef TEMPORAL_VOXEL_CULLING
        }
        else
        {
            GET_TAG_NEIGHBORHOOD_HELPER_1D( u1negativeZtimeDerivative, currentCoordinates, currentCoordinatesDim3, arrayIndex, oldElementCoordinates, volumeDimensions, +, 0, -, 1, -, 1 );
            GET_TAG_NEIGHBORHOOD_HELPER_1D( u3negativeZtimeDerivative, currentCoordinates, currentCoordinatesDim3, arrayIndex, oldElementCoordinates, volumeDimensions, -, 1, +, 0, -, 1 );
            GET_TAG_NEIGHBORHOOD_HELPER_1D( u4negativeZtimeDerivative, currentCoordinates, currentCoordinatesDim3, arrayIndex, oldElementCoordinates, volumeDimensions, +, 0, +, 0, -, 1 );
            GET_TAG_NEIGHBORHOOD_HELPER_1D( u5negativeZtimeDerivative, currentCoordinates, currentCoordinatesDim3, arrayIndex, oldElementCoordinates, volumeDimensions, +, 1, +, 0, -, 1 );
            GET_TAG_NEIGHBORHOOD_HELPER_1D( u7negativeZtimeDerivative, currentCoordinates, currentCoordinatesDim3, arrayIndex, oldElementCoordinates, volumeDimensions, +, 0, +, 1, -, 1 );

            GET_TAG_NEIGHBORHOOD_HELPER_1D( u0timeDerivative,          currentCoordinates, currentCoordinatesDim3, arrayIndex, oldElementCoordinates, volumeDimensions, -, 1, -, 1, +, 0 );
            GET_TAG_NEIGHBORHOOD_HELPER_1D( u1timeDerivative,          currentCoordinates, currentCoordinatesDim3, arrayIndex, oldElementCoordinates, volumeDimensions, +, 0, -, 1, +, 0 );
            GET_TAG_NEIGHBORHOOD_HELPER_1D( u2timeDerivative,          currentCoordinates, currentCoordinatesDim3, arrayIndex, oldElementCoordinates, volumeDimensions, +, 1, -, 1, +, 0 );
            GET_TAG_NEIGHBORHOOD_HELPER_1D( u3timeDerivative,          currentCoordinates, currentCoordinatesDim3, arrayIndex, oldElementCoordinates, volumeDimensions, -, 1, +, 0, +, 0 );

            GET_TAG_NEIGHBORHOOD_HELPER_1D( u5timeDerivative,          currentCoordinates, currentCoordinatesDim3, arrayIndex, oldElementCoordinates, volumeDimensions, +, 1, +, 0, +, 0 );
            GET_TAG_NEIGHBORHOOD_HELPER_1D( u6timeDerivative,          currentCoordinates, currentCoordinatesDim3, arrayIndex, oldElementCoordinates, volumeDimensions, -, 1, +, 1, +, 0 );
            GET_TAG_NEIGHBORHOOD_HELPER_1D( u7timeDerivative,          currentCoordinates, currentCoordinatesDim3, arrayIndex, oldElementCoordinates, volumeDimensions, +, 0, +, 1, +, 0 );
            GET_TAG_NEIGHBORHOOD_HELPER_1D( u8timeDerivative,          currentCoordinates, currentCoordinatesDim3, arrayIndex, oldElementCoordinates, volumeDimensions, +, 1, +, 1, +, 0 );

            GET_TAG_NEIGHBORHOOD_HELPER_1D( u1positiveZtimeDerivative, currentCoordinates, currentCoordinatesDim3, arrayIndex, oldElementCoordinates, volumeDimensions, +, 0, -, 1, +, 1 );
            GET_TAG_NEIGHBORHOOD_HELPER_1D( u3positiveZtimeDerivative, currentCoordinates, currentCoordinatesDim3, arrayIndex, oldElementCoordinates, volumeDimensions, -, 1, +, 0, +, 1 );
            GET_TAG_NEIGHBORHOOD_HELPER_1D( u4positiveZtimeDerivative, currentCoordinates, currentCoordinatesDim3, arrayIndex, oldElementCoordinates, volumeDimensions, +, 0, +, 0, +, 1 );
            GET_TAG_NEIGHBORHOOD_HELPER_1D( u5positiveZtimeDerivative, currentCoordinates, currentCoordinatesDim3, arrayIndex, oldElementCoordinates, volumeDimensions, +, 1, +, 0, +, 1 );
            GET_TAG_NEIGHBORHOOD_HELPER_1D( u7positiveZtimeDerivative, currentCoordinates, currentCoordinatesDim3, arrayIndex, oldElementCoordinates, volumeDimensions, +, 0, +, 1, +, 1 );

            //
            // testing -z layer of time derivative neighborhood
            //
            if ( u1negativeZtimeDerivative == 1 )
            {
                if ( !Equals( u4, u4negativeZ, tolerance ) )
                {
                    outputSelf = true;
                    numOutputs++;
                }
                if ( !Equals( u4, u1, tolerance ) )
                {
                    outputSelf = true;
                    numOutputs++;
                }
            }

            if ( u3negativeZtimeDerivative == 1 )
            {
                if ( !Equals( u4, u4negativeZ, tolerance ) )
                {
                    outputSelf = true;
                    numOutputs++;
                }
                if ( !Equals( u4, u3, tolerance ) )
                {
                    outputSelf = true;
                    numOutputs++;
                }
            }


            if ( u4negativeZtimeDerivative == 1 )
            {
                if ( !Equals( u4, u4negativeZ, tolerance ) )
                {
                    outputSelf = true;
                    numOutputs++;
                }
            }


            if ( u5negativeZtimeDerivative == 1 )
            {
                if ( !Equals( u4, u4negativeZ, tolerance ) )
                {
                    outputSelf = true;
                    numOutputs++;
                }
                if ( !Equals( u4, u5, tolerance ) )
                {
                    outputSelf = true;
                    numOutputs++;
                }
            }


            if ( u7negativeZtimeDerivative == 1 )
            {
                if ( !Equals( u4, u4negativeZ, tolerance ) )
                {
                    outputSelf = true;
                    numOutputs++;
                }
                if ( !Equals( u4, u7, tolerance ) )
                {
                    outputSelf = true;
                    numOutputs++;
                }
            }

            //
            // testing middle layer of time derivative neighborhood
            //
            if ( u0timeDerivative == 1 )
            {
                if ( !Equals( u4, u1, tolerance ) )
                {
                    outputSelf = true;
                    numOutputs++;
                }
                if ( !Equals( u4, u3, tolerance ) )
                {
                    outputSelf = true;
                    numOutputs++;
                }
            }


            if ( u1timeDerivative == 1 )
            {
                if ( !Equals( u4, u1, tolerance ) )
                {
                    outputSelf = true;
                    numOutputs++;
                }
            }


            if ( u2timeDerivative == 1 )
            {
                if ( !Equals( u4, u1, tolerance ) )
                {
                    outputSelf = true;
                    numOutputs++;
                }
                if ( !Equals( u4, u5, tolerance ) )
                {
                    outputSelf = true;
                    numOutputs++;
                }
            }


            if ( u3timeDerivative == 1 )
            {
                if ( !Equals( u4, u3, tolerance ) )
                {
                    outputSelf = true;
                    numOutputs++;
                }
            }


            if ( u5timeDerivative == 1 )
            {
                if ( !Equals( u4, u5, tolerance ) )
                {
                    outputSelf = true;
                    numOutputs++;
                }
            }


            if ( u6timeDerivative == 1 )
            {
                if ( !Equals( u4, u3, tolerance ) )
                {
                    outputSelf = true;
                    numOutputs++;
                }
                if ( !Equals( u4, u7, tolerance ) )
                {
                    outputSelf = true;
                    numOutputs++;
                }
            }


            if ( u7timeDerivative == 1 )
            {
                if ( !Equals( u4, u7, tolerance ) )
                {
                    outputSelf = true;
                    numOutputs++;
                }
            }

            if ( u8timeDerivative == 1 )
            {
                if ( !Equals( u4, u5, tolerance ) )
                {
                    outputSelf = true;
                    numOutputs++;
                }
                if ( !Equals( u4, u7, tolerance ) )
                {
                    outputSelf = true;
                    numOutputs++;
                }
            }


            //
            // testing +z layer of time derivative layer
            //
            if ( u1positiveZtimeDerivative == 1 )
            {
                if ( !Equals( u4, u4positiveZ, tolerance ) )
                {
                    outputSelf = true;
                    numOutputs++;
                }
                if ( !Equals( u4, u1, tolerance ) )
                {
                    outputSelf = true;
                    numOutputs++;
                }
            }


            if ( u3positiveZtimeDerivative == 1 )
            {
                if ( !Equals( u4, u4positiveZ, tolerance ) )
                {
                    outputSelf = true;
                    numOutputs++;
                }
                if ( !Equals( u4, u3, tolerance ) )
                {
                    outputSelf = true;
                    numOutputs++;
                }
            }


            if ( u4positiveZtimeDerivative == 1 )
            {
                if ( !Equals( u4, u4positiveZ, tolerance ) )
                {
                    outputSelf = true;
                    numOutputs++;
                }
            }


            if ( u5positiveZtimeDerivative == 1 )
            {
                if ( !Equals( u4, u4positiveZ, tolerance ) )
                {
                    outputSelf = true;
                    numOutputs++;
                }
                if ( !Equals( u4, u5, tolerance ) )
                {
                    outputSelf = true;
                    numOutputs++;
                }
            }


            if ( u7positiveZtimeDerivative == 1 )
            {
                if ( !Equals( u4, u4positiveZ, tolerance ) )
                {
                    outputSelf = true;
                    numOutputs++;
                }
                if ( !Equals( u4, u7, tolerance ) )
                {
                    outputSelf = true;
                    numOutputs++;
                }
            }
        }
#endif

        if ( outputSelf )
        {
            numOutputs++;
        }

        newActiveElementCountList[ arrayIndexInOldActiveElementList ] = numOutputs;
    }
}


#endif