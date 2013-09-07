#ifndef GPUSEG_ENGINE_HPP
#define GPUSEG_ENGINE_HPP

#if defined(PLATFORM_WIN32)

#define NOMINMAX
#include <windows.h>

typedef HWND WINDOW_HANDLE;

#elif defined(PLATFORM_OSX)

#include <Carbon/Carbon.h>

typedef HIViewRef WINDOW_HANDLE;

#endif

#include "core/RefCounted.hpp"
#include "core/Functor.hpp"
#include "core/Printf.hpp"

#include "math/Vector3.hpp"

#include "GPUSegRenderStrategy.hpp"
#include "VolumeDesc.hpp"

namespace content
{
    class Inventory;
}

namespace rendering
{
    class RenderStrategy;
    class Camera;
}

class  GPUSegRenderStrategy;
class  Segmenter;
struct VolumeFileDesc;

class Dummy
{};

class Engine : public core::RefCounted
{
public:

    Engine( WINDOW_HANDLE windowHandle );
    ~Engine();

    void Update();

    //
    // volume loading
    //
    VolumeDesc LoadVolume( const VolumeFileDesc& volumeFileDesc, const core::String& parameterFile );
    void UnloadVolume();

    //
    // segmentation saving
    //
    void SaveSegmentationAs( const core::String& fileName );

    //
    // parameter loading
    //
    void SaveParametersAs( const core::String& fileName );
    core::String GetCurrentParametersFileName();

    //
    // parameter change state (ie: auto calculate, etc.)
    //
    void SetAutomaticParameterAdjustEnabled( bool enabled );

    //
    // segmentation state
    //
    void BeginPlaceSeed();
    void AddSeedPoint( int screenX, int screenY );
    void EndPlaceSeed();

    void InitializeSeed( const math::Vector3& seedCoordinates, unsigned int sphereSize );
    void ClearCurrentSegmentation();
    void ClearAllSegmentations();
    void FreezeCurrentSegmentation();

    void FinishedSegmentationSession();

    void PlaySegmentation();
    void StopSegmentation();
    
    //
    // scripting
    //
    void LoadScript( const core::String& fileName );
    void UnloadScript();
    bool IsScriptLoaded();
    void Reload();

    //
    // give the GUI a chance to reload after we reload a file
    //
    void SetAssetChangedBeforeReloadCallback( core::IFunctor* assetChangedBeforeReloadCallback );
    void SetAssetChangedAfterReloadCallback( core::IFunctor* assetChangedAfterReloadCallback );
    void SetSegmentationStartedCallback( core::IFunctor* assetChangedAfterReloadCallback );
    void SetSegmentationFinishedCallback( core::IFunctor* assetChangedAfterReloadCallback );

    //
    // when parameters are automatically changed we must notify the rest of the program (ie: GUI)
    //
    void SetParametersChangedCallback( core::IFunctor* parametersChangedCallback );
    
#if defined(PLATFORM_WIN32)
    
    CORE_SET_CALLBACK_METHOD( Engine::SetAssetChangedBeforeReloadCallback );
    CORE_SET_CALLBACK_METHOD( Engine::SetAssetChangedAfterReloadCallback );
    CORE_SET_CALLBACK_METHOD( Engine::SetSegmentationStartedCallback );
    CORE_SET_CALLBACK_METHOD( Engine::SetSegmentationFinishedCallback );
    CORE_SET_CALLBACK_METHOD( Engine::SetParametersChangedCallback );
    
#elif defined(PLATFORM_OSX)
    
    CORE_SET_CALLBACK_METHOD( SetAssetChangedBeforeReloadCallback );
    CORE_SET_CALLBACK_METHOD( SetAssetChangedAfterReloadCallback );
    CORE_SET_CALLBACK_METHOD( SetSegmentationStartedCallback );
    CORE_SET_CALLBACK_METHOD( SetSegmentationFinishedCallback );
    CORE_SET_CALLBACK_METHOD( SetParametersChangedCallback );
    
#endif
    

    void SegmentationStarted();
    void SegmentationFinished();

    //
    // Call to notify ParametersChangedCallback that some parameters have changed
    //
    void ParametersChanged();

    //
    // mouse input functions
    //
    void SetViewport( int width, int height );
    void RotateCamera( float horizontalAngleRadians, float verticalAngleRadians );
    void MoveCameraAlongViewVector( float distance );
    void BeginSketch();
    void EndSketch();
    void ClearSketches();
    void AddSketchPoint   ( int screenX,    int screenY );
    void RemoveSketchPoint( int screenX,    int screenY );
    void MoveSketchPoint  ( int oldScreenX, int oldScreenY, int newScreenX, int newScreenY );

    //
    // allow clients to set the ground normal used to rotate Maya cameras
    //
    void SetCameraGroundNormal( const math::Vector3& cameraGroundNormal );

    //
    // render strategy calls back the engine at an appropriate time to render stuff
    //
    void RenderCallback( void* userData );

    //
    // load manager calls back the engine when we reload the scene
    //
    void AssetChangedBeforeReloadCallback( void* );

    void AssetChangedAfterReloadCallback( void* );
    
#if defined(PLATFORM_WIN32)
    
    CORE_CALLBACK_METHOD( Engine, Engine::RenderCallback );
    CORE_CALLBACK_METHOD( Engine, Engine::AssetChangedBeforeReloadCallback );
    CORE_CALLBACK_METHOD( Engine, Engine::AssetChangedAfterReloadCallback );
    
#elif defined(PLATFORM_OSX)
    
    CORE_CALLBACK_METHOD( Engine, RenderCallback, Engine::RenderCallback );
    CORE_CALLBACK_METHOD( Engine, AssetChangedBeforeReloadCallback, Engine::AssetChangedBeforeReloadCallback );
    CORE_CALLBACK_METHOD( Engine, AssetChangedAfterReloadCallback, Engine::AssetChangedAfterReloadCallback );
    
#endif
    

private:

    GPUSegRenderStrategy*      mGPUSegRenderStrategy;

    Segmenter*                 mSegmenter;

    core::IFunctor*            mAssetChangedBeforeReloadCallback;
    core::IFunctor*            mAssetChangedAfterReloadCallback;
    core::IFunctor*            mSegmentationFinishedCallback;
    core::IFunctor*            mSegmentationStartedCallback;
    core::IFunctor*            mParametersChangedCallback;

    double                     mTimeDeltaSeconds;
    double                     mTotalUserInteractionTimeSeconds;
    
    bool                       mFinishedSegmentation;
    bool                       mScriptLoaded;
    bool                       mVolumeLoaded;
    bool                       mSaveParameters;
    VolumeDesc                 mVolumeDesc;
    core::String               mCurrentScript;
    core::String               mCurrentParametersFile;

    math::Vector3              mCameraGroundNormal;
    int                        mCameraGroundNormalNonZeroCoordinate;
    int                        mCameraGroundNormalSign;
};

#endif