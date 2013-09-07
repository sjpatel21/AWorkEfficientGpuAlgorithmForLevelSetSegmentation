#ifndef CUDA_CUDA_KERNELS_HPP
#define CUDA_CUDA_KERNELS_HPP

#include "CudaTypes.hpp"

//
// kernel entry points
//
extern "C" void CudaMemSetFloat                   ( float*              deviceData,
                                                    float               valueToSet,
                                                    int                 numElements );

extern "C" void CudaMemSetFloatAsync              ( float*              deviceData,
                                                    float               valueToSet,
                                                    int                 numElements );

extern "C" void CudaMemSetInt                     ( int*                deviceData,
                                                    int                 valueToSet,
                                                    int                 numElements );

extern "C" void CudaMemSetIntAsync                ( int*                deviceData,
                                                    int                 valueToSet,
                                                    int                 numElements );

extern "C" void CudaMemSetCharSparse              ( unsigned char*      deviceData,
                                                    unsigned char       valueToSet,
                                                    size_t              numElements,
                                                    dim3                volumeDimensions );

extern "C" void CudaMemSetCharSparseAsync         ( unsigned char*      deviceData,
                                                    unsigned char       valueToSet,
                                                    size_t              numElements,
                                                    dim3                volumeDimensions );

extern "C" void CudaMemSetIntSparse               ( unsigned int*       deviceData,
                                                    unsigned int        valueToSet,
                                                    size_t              numElements,
                                                    dim3                volumeDimensions );

extern "C" void CudaMemSetIntSparseAsync          ( unsigned int*       deviceData,
                                                    unsigned int        valueToSet,
                                                    size_t              numElements,
                                                    dim3                volumeDimensions );

extern "C" void CudaInitializeCoordinateVolume    ( CudaCompactElement*  coordinateVolume,
                                                    dim3                 volumeDimensions );

extern "C" void CudaInitializeActiveElementsVolume( CudaCompactElement*  keepElementsVolume,
                                                    dim3                 volumeDimensions );

extern "C" void CudaInitializeLevelSetVolume      ( CudaLevelSetElement* levelSetVolume,
                                                    CudaTagElement*      levelSetExportVolume,
                                                    dim3                 volumeDimensions,
                                                    dim3                 seedCoordinates,
                                                    int                  sphereSize,
                                                    float                outOfPlaneAnisotropy );

extern "C" void CudaAddLevelSetVolume( CudaTagElement*      deviceResultData,
                                       CudaTagElement*      deviceAddData,
                                       dim3                 volumeDimensions );

extern "C" void CudaUpdateLevelSetVolumeAsync     ( CudaLevelSetElement* levelSetVolume,
                                                    CudaTagElement*      levelSetExportVolume,
                                                    CudaTagElement*      timeDerivativeVolume,
                                                    size_t               numActiveElements,
                                                    dim3                 volumeDimensions,
                                                    int                  target,
                                                    int                  maxDistanceBeforeShrink,
                                                    float                curvatureInfluence,
                                                    float                timeStep,
                                                    unsigned int         numBytesPerVoxel,
                                                    bool                 isSigned );

extern "C" void CudaUpdateActiveElementsVolume    ( CudaCompactElement* newActiveElementVolume,
                                                    size_t              oldNumActiveElements,
                                                    dim3                volumeDimensions );

extern "C" void CudaFilterDuplicates( CudaCompactElement* deviceData,
                                      CudaCompactElement* deviceKeep,
                                      CudaTagElement*     deviceTag,
                                      size_t              numElements,
                                      dim3                volumeDimensions );

extern "C" void CudaOutputNewActiveElements( CudaCompactElement* newActiveElementList,
                                             CudaCompactElement* newValidElementList,
                                             size_t              oldNumActiveElements,
                                             dim3                volumeDimensions );

extern "C" void CudaExportActiveElementsVolume    ( CudaTagElement*     deviceExportData,
                                                    size_t              numActiveElementsHost,
                                                    dim3                volumeDimensions );

extern "C" void CudaExportLevelSetVolume          ( CudaTagElement*     deviceExportData,
                                                    size_t              numActiveElementsHost,
                                                    dim3                volumeDimensions );



#ifdef COMPUTE_PERFORMANCE_METRICS

extern "C" void CudaInitializeActiveElementsVolumeConditionalMemoryWrite( CudaLevelSetElement*  keepElementsVolume1, CudaLevelSetElement*  keepElementsVolume2, CudaTagElement* tagElementsVolume,
                                                                          dim3                  volumeDimensions );

extern "C" void CudaInitializeActiveElementsVolumeUnconditionalMemoryWrite( CudaLevelSetElement*  keepElementsVolume1, CudaLevelSetElement*  keepElementsVolume2, CudaTagElement* tagElementsVolume,
                                                                            dim3                  volumeDimensions );

#endif

#endif