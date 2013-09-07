#ifndef SRC_CUDA_TEXTURES_CU
#define SRC_CUDA_TEXTURES_CU

texture< bool,                3, cudaReadModeElementType >     CUDA_TEXTURE_REF_SOURCE_3D_UNSUPPORTED;
texture< unsigned char,       3, cudaReadModeElementType >     CUDA_TEXTURE_REF_SOURCE_3D_UI8;
texture< char,                3, cudaReadModeElementType >     CUDA_TEXTURE_REF_SOURCE_3D_I8;
texture< unsigned short,      3, cudaReadModeElementType >     CUDA_TEXTURE_REF_SOURCE_3D_UI16;
texture< short,               3, cudaReadModeElementType >     CUDA_TEXTURE_REF_SOURCE_3D_I16;
texture< unsigned int,        3, cudaReadModeElementType >     CUDA_TEXTURE_REF_SOURCE_3D_UI32;
texture< int,                 3, cudaReadModeElementType >     CUDA_TEXTURE_REF_SOURCE_3D_I32;
texture< CudaTagElement,      1, cudaReadModeElementType >     CUDA_TEXTURE_REF_TAG_1D;
texture< CudaLevelSetElement, 1, cudaReadModeNormalizedFloat > CUDA_TEXTURE_REF_LEVEL_SET_1D;
texture< CudaCompactElement4, 1, cudaReadModeElementType >     CUDA_TEXTURE_REF_ACTIVE_ELEMENTS_1D;
texture< CudaCompactElement4, 1, cudaReadModeElementType >     CUDA_TEXTURE_REF_VALID_ELEMENTS_1D;

template< typename T > __device__ texture< T,              3, cudaReadModeElementType > GetSourceVolumeTextureReference() { return CUDA_TEXTURE_REF_SOURCE_3D_UNSUPPORTED; }
template<>             __device__ texture< unsigned char,  3, cudaReadModeElementType > GetSourceVolumeTextureReference() { return CUDA_TEXTURE_REF_SOURCE_3D_UI8;  }
template<>             __device__ texture< char,           3, cudaReadModeElementType > GetSourceVolumeTextureReference() { return CUDA_TEXTURE_REF_SOURCE_3D_I8;   }
template<>             __device__ texture< unsigned short, 3, cudaReadModeElementType > GetSourceVolumeTextureReference() { return CUDA_TEXTURE_REF_SOURCE_3D_UI16; }
template<>             __device__ texture< short,          3, cudaReadModeElementType > GetSourceVolumeTextureReference() { return CUDA_TEXTURE_REF_SOURCE_3D_I16;  }
template<>             __device__ texture< unsigned int,   3, cudaReadModeElementType > GetSourceVolumeTextureReference() { return CUDA_TEXTURE_REF_SOURCE_3D_UI32; }
template<>             __device__ texture< int,            3, cudaReadModeElementType > GetSourceVolumeTextureReference() { return CUDA_TEXTURE_REF_SOURCE_3D_I32;  }



#define GET_SOURCE_NEIGHBORHOOD_HELPER( sampleVariable, currentCoordinates, elementCoordinates, volumeDimensions, signI, i, signJ, j, signK, k, templateType )    \
    currentCoordinates . x = elementCoordinates . x signI i ;                                                                                       \
    currentCoordinates . y = elementCoordinates . y signJ j ;                                                                                       \
    currentCoordinates . z = elementCoordinates . z signK k ;                                                                                       \
                                                                                                                                                    \
                                                                                                                                                    \
    templateType sampleVariable = tex3D( GetSourceVolumeTextureReference< templateType >(), currentCoordinates . x, currentCoordinates . y, currentCoordinates . z );         \


#define GET_TAG_NEIGHBORHOOD_HELPER_1D( sampleVariable, currentCoordinatesDim3, arrayIndex, elementCoordinates, volumeDimensions, signI, i, signJ, j, signK, k )    \
                                                                                                                                                    \
    currentCoordinatesDim3 = OffsetCoordinatesUnsafe( elementCoordinates , signI i , signJ j , signK k , volumeDimensions );                        \
                                                                                                                                                    \
    arrayIndex = ComputeIndex3DToTiled1D( currentCoordinatesDim3, volumeDimensions );                                                               \
                                                                                                                                                    \
    CudaTagElement sampleVariable = tex1Dfetch( CUDA_TEXTURE_REF_TAG_1D, arrayIndex );                                                              \


#define GET_TAG_NEIGHBORHOOD_HELPER_1D_PREDEFINE_VARIABLE( sampleVariable, currentCoordinatesDim3, arrayIndex, elementCoordinates, volumeDimensions, signI, i, signJ, j, signK, k )    \
                                                                                                                                                    \
    currentCoordinatesDim3 = OffsetCoordinatesUnsafe( elementCoordinates , signI i , signJ j , signK k , volumeDimensions );                        \
                                                                                                                                                    \
    arrayIndex = ComputeIndex3DToTiled1D( currentCoordinatesDim3, volumeDimensions );                                                               \
                                                                                                                                                    \
    sampleVariable = tex1Dfetch( CUDA_TEXTURE_REF_TAG_1D, arrayIndex );                                                                             \


#define GET_LEVEL_SET_NEIGHBORHOOD_HELPER_1D( sampleVariable, currentCoordinatesDim3, arrayIndex, elementCoordinates, volumeDimensions, signI, i, signJ, j, signK, k ) \
                                                                                                                                                    \
    currentCoordinatesDim3 = OffsetCoordinatesUnsafe( elementCoordinates , signI i , signJ j , signK k , volumeDimensions );                        \
                                                                                                                                                    \
    arrayIndex = ComputeIndex3DToTiled1D( currentCoordinatesDim3, volumeDimensions );                                                               \
                                                                                                                                                    \
    float sampleVariable = tex1Dfetch( CUDA_TEXTURE_REF_LEVEL_SET_1D, arrayIndex );                                                                 \


#define GET_LEVEL_SET_NEIGHBORHOOD_HELPER_1D_PREDEFINE_VARIABLE( sampleVariable, currentCoordinatesDim3, arrayIndex, elementCoordinates, volumeDimensions, signI, i, signJ, j, signK, k ) \
                                                                                                                                                    \
    currentCoordinatesDim3 = OffsetCoordinatesUnsafe( elementCoordinates , signI i , signJ j , signK k , volumeDimensions );                        \
                                                                                                                                                    \
    arrayIndex = ComputeIndex3DToTiled1D( currentCoordinatesDim3, volumeDimensions );                                                               \
                                                                                                                                                    \
    sampleVariable = tex1Dfetch( CUDA_TEXTURE_REF_LEVEL_SET_1D, arrayIndex );                                                                       \





#define GET_LEVEL_SET_NEIGHBORHOOD_HELPER_1D_UNTILED( sampleVariable, currentCoordinatesDim3, arrayIndex, elementCoordinates, volumeDimensions, signI, i, signJ, j, signK, k ) \
                                                                                                                                                    \
    currentCoordinatesDim3 = OffsetCoordinatesUnsafe( elementCoordinates , signI i , signJ j , signK k , volumeDimensions );                        \
                                                                                                                                                    \
    arrayIndex = ComputeIndex3DTo1D( currentCoordinatesDim3, volumeDimensions );                                                                    \
                                                                                                                                                    \
    float sampleVariable = tex1Dfetch( CUDA_TEXTURE_REF_LEVEL_SET_1D, arrayIndex );                                                                 \


#define GET_TAG_NEIGHBORHOOD_HELPER_1D_PREDEFINE_VARIABLE_UNTILED( sampleVariable, currentCoordinatesDim3, arrayIndex, elementCoordinates, volumeDimensions, signI, i, signJ, j, signK, k )    \
                                                                                                                                                    \
    currentCoordinatesDim3 = OffsetCoordinatesUnsafe( elementCoordinates , signI i , signJ j , signK k , volumeDimensions );                        \
                                                                                                                                                    \
    arrayIndex = ComputeIndex3DTo1D( currentCoordinatesDim3, volumeDimensions );                                                                    \
                                                                                                                                                    \
    sampleVariable = tex1Dfetch( CUDA_TEXTURE_REF_TAG_1D, arrayIndex );                                                                             \


#define GET_LEVEL_SET_NEIGHBORHOOD_HELPER_1D_PREDEFINE_VARIABLE_UNTILED( sampleVariable, currentCoordinatesDim3, arrayIndex, elementCoordinates, volumeDimensions, signI, i, signJ, j, signK, k ) \
                                                                                                                                                    \
    currentCoordinatesDim3 = OffsetCoordinatesUnsafe( elementCoordinates , signI i , signJ j , signK k , volumeDimensions );                        \
                                                                                                                                                    \
    arrayIndex = ComputeIndex3DTo1D( currentCoordinatesDim3, volumeDimensions );                                                                    \
                                                                                                                                                    \
    sampleVariable = tex1Dfetch( CUDA_TEXTURE_REF_LEVEL_SET_1D, arrayIndex );                                                                       \


#endif