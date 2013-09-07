#ifndef SEGMENTER_HPP
#define SEGMENTER_HPP

#include <builtin_types.h>

#include <cudpp.h>

#include "core/RefCounted.hpp"

#include "container/List.hpp"
#include "container/Array.hpp"

#include "rendering/rtgi/Texture.hpp"

#include "VolumeDesc.hpp"
#include "Engine.hpp"
#include "cuda/CudaTypes.hpp"

namespace math
{
    class Vector3;
}

namespace rendering
{
namespace rtgi
{
    class VertexBuffer;
    class PixelBuffer;
    class FrameBufferObject;
    class Effect;
}
}

class Engine;

class Segmenter : public core::RefCounted
{
public:
    Segmenter();
    ~Segmenter();

    void                                SetEngine( Engine* engine );
    void                                SetAutomaticParameterAdjustEnabled( bool );

    VolumeDesc                          LoadVolume( const VolumeDesc& volumeDesc );
    void                                UnloadVolume();

    VolumeDesc                          GetSaveableSegmentation();
                                  
    void                                InitializeSeed( const math::Vector3& seedCoordinates, unsigned int sphereSize );
    void                                ClearCurrentSegmentation();
    void                                ClearAllSegmentations();
    void                                FreezeCurrentSegmentation();

    void                                FinishedSegmentationSession();
                                  
    void                                RequestUpdateSegmentation( double timeDeltaSeconds );
    void                                RequestPlaySegmentation();
    void                                RequestPauseSegmentation();
                                  
    bool                                IsSegmentationInitialized();
    bool                                IsSegmentationFinished();
    bool                                IsSegmentationInProgress();
    bool                                IsAutomaticParameterAdjustEnabled();
                                  
    rendering::rtgi::Texture*           GetSourceVolumeTexture();
    rendering::rtgi::Texture*           GetCurrentLevelSetVolumeTexture();
    rendering::rtgi::Texture*           GetFrozenLevelSetVolumeTexture();
    rendering::rtgi::Texture*           GetActiveElementsVolumeTexture();
                                  
    size_t                              GetActiveElementCount();
    VolumeDesc                          GetVolumeDesc();
                                  
private:
    VolumeDesc                          LoadVolumeHostPadded( const VolumeDesc& volumeDesc );
    VolumeDesc                          UnloadVolumeHostPadded( const VolumeDesc& volumeDesc );

    // Unpads volumeDesc according to its difference in size (padding) from mVolumeBeforePaddingDesc
    VolumeDesc                            UnpadVolume( const VolumeDesc& volumeDesc );

    void                                UpdateRenderingTextures();
    void                                ClearTexture( rendering::rtgi::Texture* texture );
    void                                TagTextureSparse( rendering::rtgi::Texture* texture, float tagValue );

    void                                SwapLevelSetBuffers();

    bool                                ShouldRestartSegmentation();
    bool                                ShouldUpdateSegmentation();
    bool                                ShouldOptimizeForFewActiveVoxels();
    bool                                ShouldInitializeActiveElements();
    bool                                ShouldInitializeCoordinates();
    bool                                ShouldUpdateRenderingTextures();

    void                                UpdateHostStateBeforeRequestUpdate();
    void                                UpdateHostStateAfterRequestUpdate( double timeDeltaSeconds );

    void                                UpdateHostStateRestartSegmentation();

    void                                UpdateHostStateBeforeUpdateSegmentation();
    void                                UpdateHostStateAfterUpdateSegmentation();
    void                                UpdateHostStateDoNotUpdateSegmentation();

    void                                UpdateHostStateBeforeUpdateSegmentationIteration();
    void                                UpdateHostStateAfterUpdateSegmentationIteration( const CudaTagElement* levelSetExportBuffer );

    void                                UpdateHostStateOptimizeForFewActiveVoxels();
    void                                UpdateHostStateOptimizeForManyActiveVoxels();

    void                                UpdateHostStateInitializeActiveElements();
    void                                UpdateHostStateInitializeCoordinates();

    rendering::rtgi::TexturePixelFormat GetTexturePixelFormat( const VolumeDesc& volumeDesc );
    const char*                         GetCudaSourceTexture ( const VolumeDesc& volumeDesc );
    unsigned int                        GetNumActiveElementsAligned();

    void                                BeforeLoadVolumeDebug( const VolumeDesc& volumeDesc );
    void                                AfterLoadVolumeDebug();

    void                                WriteSegmentationToFile();
    void                                ComputeSegmentationAccuracy( const CudaTagElement* levelSetExportBuffer );
    void                                PrintSegmentationDetails();

    void                                CalculateSegmentationParameters();
    // Sets outCollectedValues to be full of the voxel values inside the current level set and
    // returns the total value of all voxels collected
    float                                CollectCurrentLevelSetVoxels( container::Array< float > &outCollectedValues );


    // device data global memory
    CudaLevelSetElement*             mLevelSetVolumeDeviceX;
    CudaLevelSetElement*             mLevelSetVolumeDeviceY;
    CudaLevelSetElement*             mLevelSetVolumeDeviceRead;
    CudaLevelSetElement*             mLevelSetVolumeDeviceWrite;

    CudaCompactElement*              mCoordinatesVolumeDevice;
    CudaCompactElement*              mValidElementsVolumeDevice;

    CudaTagElement*                  mTagVolumeDevice;

    cudaArray*                       mSourceVolumeArray3DDevice;

    size_t*                          mNumValidActiveElementsDevice;

    // compact
    CUDPPHandle                      mCompactPlanHandle;

    // rtgi objects
    rendering::rtgi::Texture*           mSourceVolumeTexture;
    rendering::rtgi::Texture*           mActiveElementsTexture;
    rendering::rtgi::Texture*           mCurrentLevelSetVolumeTexture;
    rendering::rtgi::Texture*           mFrozenLevelSetVolumeTexture;

    rendering::rtgi::PixelBuffer*       mLevelSetExportPixelBuffer;
    rendering::rtgi::PixelBuffer*       mFrozenLevelSetExportPixelBuffer;

    rendering::rtgi::VertexBuffer*      mActiveElementsCompactedVertexBuffer;

    rendering::rtgi::FrameBufferObject* mFrameBufferObject;

    rendering::rtgi::Effect*            mTagVolumeEffect;

    // sizes of volumes
    dim3                             mVolumeDimensions;
                               
    int                              mSourceVolumeNumBytes;
    int                              mTagVolumeNumBytes;
    int                              mLevelSetVolumeNumBytes;
    int                              mCompactVolumeNumBytes;

    // simulation state
    size_t                           mNumValidActiveElementsHost;
    int                              mVolumeNumElements;
    int                              mNumIterations;
    float                            mTargetPrevious;
    float                            mMaxDistanceBeforeShrinkPrevious;
    float                            mCurvatureInfluencePrevious;
    float                            mTimeStepPrevious;

    // volume desc
    VolumeDesc                       mVolumeDesc;
    VolumeDesc                         mVolumeBeforePaddingDesc;

    // engine
    Engine*                          mEngine;

    // playback state
    bool                             mLevelSetInitialized;
    bool                             mCoordinatesVolumeInitialized;
    bool                             mComputedSegmentationDetailsOnce;
    bool                             mPaused;
    bool                             mCallEngineOnResumeSegmentation;
    bool                             mCallEngineOnStopSegmentation;
    bool                             mAutomaticParameterAdjustEnabled;

#ifdef COMPUTE_PERFORMANCE_METRICS

    // counters and buffers
    CudaLevelSetElement*       mLevelSetVolumeDummy1;
    CudaLevelSetElement*       mLevelSetVolumeDummy2;
    CudaTagElement*            mTagVolumeDummy;
    CudaCompactElement*        mCompactVolumeDummy;
    container::Array< int >    mActiveTileCounter;
    container::Array< int >    mActiveVoxelCounter;
    container::Array< double > mLevelSetUpdateTimer;
    container::Array< double > mOutputNewActiveVoxelsTimer;
    container::Array< double > mInitializeActiveVoxelsConditionalMemoryWriteTimer;
    container::Array< double > mInitializeActiveVoxelsUnconditionalMemoryWriteTimer;
    container::Array< double > mWriteAndCompactEntireVolumeTimer;
    container::Array< double > mFilterDuplicatesTimer;
    container::Array< double > mCompactTimer;
    container::Array< double > mClearTagVolumeTimer;
    container::Array< double > mClearValidVolumeTimer;

#endif

};

#endif
