#include "python/ScriptManager.hpp"
#include "python/Macros.hpp"

#include "Engine.hpp"

#include "core/MemoryMonitor.hpp"
#include "core/Time.hpp"
#include "core/String.hpp"
#include "core/Functor.hpp"
#include "core/Printf.hpp"

#include "math/Vector3.hpp"
#include "math/Vector4.hpp"
#include "math/Matrix44.hpp"

#include "container/Map.hpp"

#include "content/ParameterManager.hpp"
#include "content/LoadManager.hpp"
#include "content/Parameter.hpp"
#include "content/Ref.hpp"

#include "rendering/rtgi/RTGI.hpp"
#include "rendering/DebugDraw.hpp"
#include "rendering/Context.hpp"
#include "rendering/TextConsole.hpp"
#include "rendering/DebugDraw.hpp"
#include "rendering/Scene.hpp"
#include "rendering/Camera.hpp"
#include "rendering/renderstrategies/SketchRenderStrategy.hpp"

#include "Config.hpp"
#include "GPUSegRenderStrategy.hpp"
#include "Segmenter.hpp"
#include "VolumeLoader.hpp"
#include "VolumeSaver.hpp"
#include "VolumeFileDesc.hpp"
#include "VolumeDesc.hpp"

#include "cuda/Cuda.hpp"

#ifdef LEFOHN_NO_DEBUG_BENCHMARK

// Hack to get rid of the TILE_SIZE defined in Config.hpp since mock lefohn
// uses it's on TILE_SIZE static const.
const int __MangledTileSize = TILE_SIZE;

#ifdef TILE_SIZE
#undef TILE_SIZE
#endif

#include "lefohn/lefohnSegmentationSimulator.hpp"

#define TILE_SIZE ( __MangledTileSize )
#endif

PYTHON_DECLARE_CLASS_FACTORY( CreateGPUSegRenderStrategy, GPUSegRenderStrategy )

PYTHON_MODULE_BEGIN( gpuseg )

    PYTHON_CLASS_FACTORY( "CreateGPUSegRenderStrategy", CreateGPUSegRenderStrategy, GPUSegRenderStrategy )

    PYTHON_CLASS_BEGIN( "VolumeFileDesc", VolumeFileDesc )
        PYTHON_CLASS_METHOD_CONSTRUCTOR()
        PYTHON_CLASS_FIELD_READWRITE( "upDirection",      VolumeFileDesc::upDirection )
        PYTHON_CLASS_FIELD_READWRITE( "fileName",         VolumeFileDesc::fileName )
        PYTHON_CLASS_FIELD_READWRITE( "zAnisotropy",      VolumeFileDesc::zAnisotropy )
        PYTHON_CLASS_FIELD_READWRITE( "isSigned",         VolumeFileDesc::isSigned )
        PYTHON_CLASS_FIELD_READWRITE( "numBytesPerVoxel", VolumeFileDesc::numBytesPerVoxel )
        PYTHON_CLASS_FIELD_READWRITE( "numVoxelsX",       VolumeFileDesc::numVoxelsX )
        PYTHON_CLASS_FIELD_READWRITE( "numVoxelsY",       VolumeFileDesc::numVoxelsY )
        PYTHON_CLASS_FIELD_READWRITE( "numVoxelsZ",       VolumeFileDesc::numVoxelsZ )
    PYTHON_CLASS_END

    PYTHON_CLASS_BEGIN( "VolumeDesc", VolumeDesc )
    PYTHON_CLASS_END

    PYTHON_CLASS_BEGIN_DERIVED( "GPUSegRenderStrategy", GPUSegRenderStrategy, rendering::RenderStrategy )
        PYTHON_CLASS_METHOD( "SetEngine",    GPUSegRenderStrategy::SetEngine )
        PYTHON_CLASS_METHOD( "LoadVolume",   GPUSegRenderStrategy::LoadVolume )
        PYTHON_CLASS_METHOD( "UnloadVolume", GPUSegRenderStrategy::UnloadVolume )
    PYTHON_CLASS_END

    PYTHON_CLASS_BEGIN( "Engine", Engine )
        PYTHON_CLASS_METHOD_CALLBACK( "RenderCallback", Engine::RenderCallback )
    PYTHON_CLASS_END

    PYTHON_CLASS_BEGIN_NON_COPYABLE( "VolumeLoader", VolumeLoader )
        PYTHON_CLASS_METHOD_STATIC( "LoadVolume",    VolumeLoader::LoadVolume )
        PYTHON_CLASS_METHOD_STATIC( "UnloadVolume",  VolumeLoader::UnloadVolume )
    PYTHON_CLASS_END

    PYTHON_SHARED_CLASS( Engine )
        
PYTHON_MODULE_END


Engine::Engine( WINDOW_HANDLE windowHandle ) :
    mGPUSegRenderStrategy               ( NULL ),
    mSegmenter                          ( NULL ),
    mAssetChangedBeforeReloadCallback   ( NULL ),
    mAssetChangedAfterReloadCallback    ( NULL ),
    mSegmentationFinishedCallback       ( NULL ),
    mSegmentationStartedCallback        ( NULL ),
    mParametersChangedCallback          ( NULL ),
    mTimeDeltaSeconds                   ( 0.0l ),
    mScriptLoaded                       ( false ),
    mVolumeLoaded                       ( false ),
    mSaveParameters                     ( false ),
    mCameraGroundNormal                 ( 0, 0, 1 ),
    mCameraGroundNormalNonZeroCoordinate( math::Z ),
    mCameraGroundNormalSign             ( 1 )
{
    core::MemoryMonitor::Initialize();
    content::LoadManager::Initialize();
    content::ParameterManager::Initialize();
    rendering::Context::Initialize( windowHandle );
    python::ScriptManager::Initialize();

    PYTHON_INITIALIZE_MODULE( gpuseg )

    content::LoadManager::SetAssetChangedBeforeReloadCallback( new core::Functor< Engine >( this, &Engine::AssetChangedBeforeReloadCallback ) );
    content::LoadManager::SetAssetChangedAfterReloadCallback ( new core::Functor< Engine >( this, &Engine::AssetChangedAfterReloadCallback ) );

    mSegmenter = new Segmenter();
    mSegmenter->AddRef();
    mSegmenter->SetEngine( this );

    mGPUSegRenderStrategy = new GPUSegRenderStrategy();
    mGPUSegRenderStrategy->AddRef();
    mGPUSegRenderStrategy->SetEngine( this );
    mGPUSegRenderStrategy->SetRenderCallback( new core::Functor< Engine >( this, &Engine::RenderCallback ) );

#ifdef WHITE_BACKGROUND
    mGPUSegRenderStrategy->SetClearColor( rendering::rtgi::ColorRGBA( 1, 1, 1, 1 ) );
#else
    mGPUSegRenderStrategy->SetClearColor( rendering::rtgi::ColorRGBA( 0, 0, 0, 1 ) );
#endif

    rendering::Context::SetCurrentRenderStrategy( mGPUSegRenderStrategy );

#ifdef LEFOHN_NO_DEBUG_BENCHMARK
    lefohn::RunLefohnSimulation();
#endif
}


Engine::~Engine()
{
    // if a volume is loaded then unload it
    if ( mVolumeLoaded )
    {
        UnloadVolume();
    }

    AssignRef( mAssetChangedBeforeReloadCallback, NULL );
    AssignRef( mAssetChangedAfterReloadCallback,  NULL );
    AssignRef( mSegmentationStartedCallback,      NULL );
    AssignRef( mSegmentationFinishedCallback,     NULL );
    AssignRef( mParametersChangedCallback,        NULL );

    rendering::Context::SetCurrentRenderStrategy( NULL );

    mGPUSegRenderStrategy->Release();
    mGPUSegRenderStrategy = NULL;

    mSegmenter->Release();
    mSegmenter = NULL;

    PYTHON_TERMINATE_MODULE( gpuseg )

    python::ScriptManager::Terminate();
    rendering::Context::Terminate();
    content::ParameterManager::Terminate();
    content::LoadManager::Terminate();
    core::MemoryMonitor::Terminate();
}

VolumeDesc Engine::LoadVolume( const VolumeFileDesc& volumeFileDesc, const core::String& parameterFile )
{
    // inform UI that we're about to change
    if ( mAssetChangedBeforeReloadCallback != NULL )
    {
        mAssetChangedBeforeReloadCallback->Call( NULL );
    }

    // if a volume is loaded then unload it
    if ( mVolumeLoaded )
    {
        UnloadVolume();
    }

    // load volume on CPU
    VolumeDesc volumeDesc = VolumeLoader::LoadVolume( volumeFileDesc );

    // load parameters
    if ( parameterFile == "" )
    {
        mSaveParameters = false;

        if ( volumeDesc.maxValue <= 255.0f )
        {
            mCurrentParametersFile = "defaults/UnsignedChar.par";
        }
        else
        if ( volumeDesc.maxValue <= 4095.0f )
        {
            mCurrentParametersFile = "defaults/UnsignedShort.par";
        }
        else
        if ( volumeDesc.maxValue <= 65535.0f )
        {
            mCurrentParametersFile = "defaults/UnsignedShortHDR.par";
        }
        else
        {
            ReleaseAssert( 0 );
        }
    }
    else
    {
        mCurrentParametersFile = parameterFile;
        mSaveParameters        = true;
    }

    content::ParameterManager::LoadParameters( mCurrentParametersFile );

    // load volume on GPU
    mVolumeDesc = mSegmenter->LoadVolume( volumeDesc );
    mGPUSegRenderStrategy->LoadVolume( mVolumeDesc );

    mGPUSegRenderStrategy->SetSourceVolumeTexture( mSegmenter->GetSourceVolumeTexture() );
    mGPUSegRenderStrategy->SetCurrentLevelSetVolumeTexture( mSegmenter->GetCurrentLevelSetVolumeTexture() );
    mGPUSegRenderStrategy->SetFrozenLevelSetVolumeTexture( mSegmenter->GetFrozenLevelSetVolumeTexture() );
    mGPUSegRenderStrategy->SetActiveElementsVolumeTexture( mSegmenter->GetActiveElementsVolumeTexture() );

    SetCameraGroundNormal( mVolumeDesc.upDirection );

    // unload volume on CPU
    VolumeLoader::UnloadVolume( volumeDesc );

    mTotalUserInteractionTimeSeconds = 0.0;
    mVolumeLoaded                    = true;
    mFinishedSegmentation            = false;

#ifdef PRINT_USER_INTERACTION_TIME
    rendering::TextConsole::PrintToStaticConsole( "userInteractionTime", "" );
#endif

    // inform UI that we've changed
    if ( mAssetChangedAfterReloadCallback != NULL )
    {
        mAssetChangedAfterReloadCallback->Call( NULL );
    }

    return mVolumeDesc;
}

void Engine::UnloadVolume()
{
    // inform UI that we're about to change
    if ( mAssetChangedBeforeReloadCallback != NULL )
    {
        mAssetChangedBeforeReloadCallback->Call( NULL );
    }

    // unload volume
    if ( mVolumeLoaded )
    {
        mGPUSegRenderStrategy->UnloadVolume();
        mSegmenter->UnloadVolume();

        SetCameraGroundNormal( math::Vector3( 0, 1, 0 ) );
    }

    // save parameters
    if ( mSaveParameters )
    {
        content::ParameterManager::SaveParameters( mCurrentParametersFile );
    }

    mCurrentParametersFile = "";

    // inform UI that we've changed
    if ( mAssetChangedAfterReloadCallback != NULL )
    {
        mAssetChangedAfterReloadCallback->Call( NULL );
    }
}

void Engine::SaveParametersAs( const core::String& fileName )
{
    boost::filesystem::path fromFilePath( mCurrentParametersFile.ToStdString() );
    boost::filesystem::path toFilePath( fileName.ToStdString() );

    if ( boost::filesystem::exists( toFilePath ) )
    {
        boost::filesystem::remove( toFilePath );
    }

    boost::filesystem::copy_file( fromFilePath, toFilePath );

    content::ParameterManager::SaveParameters( fileName );

    mCurrentParametersFile = fileName;
    mSaveParameters        = true;
}

void Engine::SaveSegmentationAs( const core::String& fileName )
{
    mSegmenter->RequestPauseSegmentation();

    VolumeDesc segmentationVolume = mSegmenter->GetSaveableSegmentation();

    VolumeSaver::SaveVolume( fileName, segmentationVolume );

    delete [] segmentationVolume.volumeData;
}

core::String Engine::GetCurrentParametersFileName()
{
    if ( mSaveParameters )
    {
        return mCurrentParametersFile;
    }
    else
    {
        return "";
    }
}

void Engine::LoadScript( const core::String& fileName )
{
    Assert( !mScriptLoaded );

    mCurrentScript = fileName;

    AssetChangedBeforeReloadCallback( NULL );

    python::ScriptManager::Load( "gpuseg", fileName );
    python::ScriptManager::Call< Engine >( "gpuseg", "Initialize", this );

    AssetChangedAfterReloadCallback( NULL );

    mScriptLoaded = true;
}

void Engine::UnloadScript()
{
    if ( mScriptLoaded )
    {
        mScriptLoaded = false;

        python::ScriptManager::Call( "gpuseg", "Terminate" );
        python::ScriptManager::Unload( "gpuseg", mCurrentScript );

        mCurrentScript  = "";
    }
}

bool Engine::IsScriptLoaded()
{
    return mScriptLoaded;
}

void Engine::Update()
{
    //
    // get time
    //
    mTimeDeltaSeconds = core::TimeGetTimeDeltaSeconds();

    if ( !mFinishedSegmentation )
    {
        mTotalUserInteractionTimeSeconds += mTimeDeltaSeconds;
    }

    //
    // draw fps
    //
    //double       framesPerSecond           = 1.0f / mTimeDeltaSeconds;
    //double       frameTimeMilliseconds     = mTimeDeltaSeconds * 1000.0f;
    //core::String frameTimeString           = core::String( "Frame Time: %1 ms ( %2 fps )" ).arg( frameTimeMilliseconds, 6 ).arg( framesPerSecond, 6 );
    //rendering::TextConsole::PrintToStaticConsole( "frameTime",           frameTimeString );

    //
    // reload any files that have changed
    //
    content::LoadManager::Update();
    python::ScriptManager::Update( mTimeDeltaSeconds );

    //
    // if there is a script loaded, give it a chance to update
    //
    if ( mScriptLoaded )
    {
        python::ScriptManager::Call( "gpuseg", "Update", mTimeDeltaSeconds );
    }

    //
    // update the segmenter
    //
    if ( mVolumeLoaded )
    {
        mSegmenter->RequestUpdateSegmentation( mTimeDeltaSeconds );
    }

    //
    // update the rendering context
    //
    rendering::Context::Update( mTimeDeltaSeconds );
    rendering::Context::Render();

    //
    // if there is no render strategy, then perform the sandbox render callback,
    // otherwise we let the rendering strategy perform the callback, since it will
    // know the right time to do it (typically after all geometry passes, but before
    // any full-screen passes)
    //
    if ( rendering::Context::GetCurrentRenderStrategy() == NULL )
    {
        Engine::RenderCallback( NULL );
    }
}

void Engine::Reload()
{
    if ( mScriptLoaded )
    {
        core::String script = mCurrentScript;
        
        UnloadScript();

        // sanity check in case the current camera was a loaded camera
        rendering::Context::SetCurrentCamera( rendering::Context::GetDebugCamera() );

        LoadScript( script );
    }
}

void Engine::AssetChangedBeforeReloadCallback( void* data )
{
    if ( mAssetChangedBeforeReloadCallback != NULL )
    {
        mAssetChangedBeforeReloadCallback->Call( data );
    }
}

void Engine::AssetChangedAfterReloadCallback( void* data )
{
    if ( mAssetChangedAfterReloadCallback != NULL )
    {
        mAssetChangedAfterReloadCallback->Call( data );
    }
}

void Engine::SetAssetChangedBeforeReloadCallback( core::IFunctor* assetChangedAfterReloadCallback )
{
    AssignRef( mAssetChangedAfterReloadCallback, assetChangedAfterReloadCallback );
}

void Engine::SetAssetChangedAfterReloadCallback( core::IFunctor* assetChangedAfterReloadCallback )
{
    AssignRef( mAssetChangedAfterReloadCallback, assetChangedAfterReloadCallback );
}

void Engine::SetSegmentationStartedCallback( core::IFunctor* segmentationStartedCallback )
{
    AssignRef( mSegmentationStartedCallback, segmentationStartedCallback );
}

void Engine::SetSegmentationFinishedCallback( core::IFunctor* segmentationFinishedCallback )
{
    AssignRef( mSegmentationFinishedCallback, segmentationFinishedCallback );
}

void Engine::SetParametersChangedCallback( core::IFunctor* parametersChangedCallback )
{
    AssignRef( mParametersChangedCallback, parametersChangedCallback );
}

void Engine::ParametersChanged()
{
    if( mParametersChangedCallback  != NULL )
    {
        mParametersChangedCallback->Call( NULL );
    }
}

void Engine::SegmentationStarted()
{
    if ( mSegmentationStartedCallback != NULL )
    {
        mSegmentationStartedCallback->Call( NULL );
    }
}

void Engine::SegmentationFinished()
{
    if ( mSegmentationFinishedCallback != NULL )
    {
        mSegmentationFinishedCallback->Call( NULL );
    }
}

void Engine::SetCameraGroundNormal(  const math::Vector3& cameraGroundNormal )
{
    mCameraGroundNormal = cameraGroundNormal;

    math::Vector3 position, target, up;
    rendering::Context::GetCurrentCamera()->GetLookAtVectors( position, target, up );
    rendering::Context::GetCurrentCamera()->SetLookAtVectors( position, target, cameraGroundNormal );

    if ( mCameraGroundNormal == math::Vector3( 0, 0, 1 ) )
    {
        mCameraGroundNormalNonZeroCoordinate = math::Z;
        mCameraGroundNormalSign = 1;
        return;
    }
    else
    if ( mCameraGroundNormal == math::Vector3( 0, 0, -1 ) )
    {
        mCameraGroundNormalNonZeroCoordinate = math::Z;
        mCameraGroundNormalSign = -1;
    }
    else
    if ( mCameraGroundNormal == math::Vector3( 0, 1, 0 ) )
    {
        mCameraGroundNormalNonZeroCoordinate = math::Y;
        mCameraGroundNormalSign = 1;
        return;
    }
    else
    if ( mCameraGroundNormal == math::Vector3( 0, -1, 0 ) )
    {
        mCameraGroundNormalNonZeroCoordinate = math::Y;
        mCameraGroundNormalSign = -1;
        return;
    }
    else
    if ( mCameraGroundNormal == math::Vector3( 1, 0, 0 ) )
    {
        mCameraGroundNormalNonZeroCoordinate = math::X;
        mCameraGroundNormalSign = 1;
        return;
    }
    else
    if ( mCameraGroundNormal == math::Vector3( -1, 0, 0 ) )
    {
        mCameraGroundNormalNonZeroCoordinate = math::X;
        mCameraGroundNormalSign = -1;
        return;
    }
    else
    {
        Assert( 0 );
    }
}

void Engine::RotateCamera( float horizontalAngleRadians, float verticalAngleRadians )
{
    math::Vector3 oldPosition, oldTarget, oldUp, oldTargetToOldPosition, oldTargetToOldPositionUnit;
    math::Vector3 planarOldPosition, planarOldTarget, planarOldTargetToOldPosition, planarOldRight, planarNewRight;
    math::Vector3 newPositionParallel, newPositionPerpendicular, newPosition, newUp, newPositionToOldTarget;
    math::Vector3 newPositionToOldTargetLengthAdjusted;

    math::Matrix44 rotationMatrix;

    //
    // compute the vertical component of the rotation first
    //
    rendering::Context::GetCurrentCamera()->GetLookAtVectors( oldPosition, oldTarget, oldUp );

    newPositionParallel      = cos( verticalAngleRadians ) * ( oldPosition - oldTarget );
    newPositionPerpendicular = sin( verticalAngleRadians ) * ( oldPosition - oldTarget ).Length() * oldUp;
    newPosition              = oldTarget + newPositionParallel + newPositionPerpendicular;

    rendering::Context::GetCurrentCamera()->SetLookAtVectors( newPosition, oldTarget, oldUp );

    //
    // horizontally rotate the camera's position by temporarily discarding the y coordinates
    // of the old position and target, compute the rotation in 2D, and set the new position
    // y coordinate to equal the old y coordinate.  in other words, do a 2D rotation on the
    // horizontal plane.
    //
    rendering::Context::GetCurrentCamera()->GetLookAtVectors( oldPosition, oldTarget, oldUp );

    if ( mCameraGroundNormalSign * oldUp[ mCameraGroundNormalNonZeroCoordinate ] < 0 )
    {
        horizontalAngleRadians *= -1;
    }

    planarOldPosition                                         = oldPosition;
    planarOldTarget                                           = oldTarget;
    planarOldTarget[ mCameraGroundNormalNonZeroCoordinate ]   = 0;
    planarOldPosition[ mCameraGroundNormalNonZeroCoordinate ] = 0;    
    planarOldTargetToOldPosition                              = planarOldPosition - planarOldTarget;
    oldTargetToOldPosition                                    = oldPosition - oldTarget;
    oldTargetToOldPositionUnit                                = oldPosition - oldTarget;
    oldTargetToOldPositionUnit.Normalize();

    newPositionParallel =
        cos( horizontalAngleRadians ) *
        planarOldTargetToOldPosition;


    math::Vector3 oldRightUnit = math::CrossProduct( oldTargetToOldPositionUnit, oldUp );
    oldRightUnit.Normalize();

    newPositionPerpendicular = sin( horizontalAngleRadians ) * planarOldTargetToOldPosition.Length() * oldRightUnit;

    newPosition                                         = planarOldTarget + newPositionParallel + newPositionPerpendicular;
    newPosition[ mCameraGroundNormalNonZeroCoordinate ] = oldPosition[ mCameraGroundNormalNonZeroCoordinate ];

    //
    // now rotate the camera's orientation.  assume that the camera's right vector always lies
    // in the horizontal plane.  compute the new right vector by rotating the old right vector
    // in 2D on the horizontal plane.
    //
    Assert( math::Equals( math::DotProduct( oldTargetToOldPositionUnit, oldUp ), 0 ) );

    planarOldRight = math::CrossProduct( oldTargetToOldPositionUnit, oldUp );
    planarOldRight.Normalize();

    Assert( math::Equals( planarOldRight[ mCameraGroundNormalNonZeroCoordinate ], 0, 0.001f ) );

    rotationMatrix.SetToIdentity();
    
    if ( mCameraGroundNormalSign * oldUp[ mCameraGroundNormalNonZeroCoordinate ] >= 0 )
    {
        rotationMatrix.SetToRotation( - horizontalAngleRadians, mCameraGroundNormal );
    }
    else
    {
        rotationMatrix.SetToRotation( - horizontalAngleRadians, - mCameraGroundNormal );
    }

    planarNewRight = rotationMatrix.Transform( planarOldRight );
    planarNewRight.Normalize();

    Assert( math::DotProduct( planarNewRight, planarOldRight ) > 0 );

    newPositionToOldTarget = oldTarget - newPosition;

    newUp = math::CrossProduct( newPositionToOldTarget, planarNewRight );
    newUp.Normalize();

    Assert( math::DotProduct( newUp, oldUp ) > 0 );

    rendering::Context::GetCurrentCamera()->SetLookAtVectors(
        newPosition,
        oldTarget,
        newUp );
}

static bool sPlaceSeedBegun = false;

void Engine::BeginPlaceSeed()
{
    if ( !sPlaceSeedBegun )
    {
        sPlaceSeedBegun       = true;
        mFinishedSegmentation = false;

#ifdef PRINT_USER_INTERACTION_TIME
        rendering::TextConsole::PrintToStaticConsole( "userInteractionTime", "" );
#endif

        mGPUSegRenderStrategy->BeginPlaceSeed();
    }
}

void Engine::EndPlaceSeed()
{
    if ( sPlaceSeedBegun )
    {
        mGPUSegRenderStrategy->EndPlaceSeed();

        sPlaceSeedBegun       = false;
        mFinishedSegmentation = false;
#ifdef PRINT_USER_INTERACTION_TIME
        rendering::TextConsole::PrintToStaticConsole( "userInteractionTime", "" );
#endif
    }
}

void Engine::AddSeedPoint( int screenX, int screenY )
{
    if ( sPlaceSeedBegun )
    {
        int virtualScreenX, virtualScreenY;
        int initialViewportWidth, initialViewportHeight;

        rendering::Context::GetInitialViewport( initialViewportWidth, initialViewportHeight );

        rendering::rtgi::GetVirtualScreenCoordinates(
            initialViewportWidth,
            initialViewportHeight,
            screenX,
            screenY,
            virtualScreenX,
            virtualScreenY );

        mGPUSegRenderStrategy->AddSeedPoint( virtualScreenX, virtualScreenY );
        mFinishedSegmentation = false;
#ifdef PRINT_USER_INTERACTION_TIME
        rendering::TextConsole::PrintToStaticConsole( "userInteractionTime", "" );
#endif
    }
}

void Engine::ClearCurrentSegmentation()
{
    if ( mVolumeLoaded )
    {
        mSegmenter->ClearCurrentSegmentation();
        mFinishedSegmentation = false;
#ifdef PRINT_USER_INTERACTION_TIME
        rendering::TextConsole::PrintToStaticConsole( "userInteractionTime", "" );
#endif
    }
}

void Engine::FreezeCurrentSegmentation()
{
    if ( mVolumeLoaded )
    {
        mSegmenter->FreezeCurrentSegmentation();
    }
}

void Engine::InitializeSeed( const math::Vector3& seedCoordinates, unsigned int sphereSize )
{
    if ( mVolumeLoaded )
    {
        mSegmenter->InitializeSeed( seedCoordinates, sphereSize );
        mFinishedSegmentation = false;
#ifdef PRINT_USER_INTERACTION_TIME
        rendering::TextConsole::PrintToStaticConsole( "userInteractionTime", "" );
#endif
    }
}

void Engine::ClearAllSegmentations()
{
    if ( mVolumeLoaded )
    {
        mSegmenter->ClearAllSegmentations();
        mFinishedSegmentation = false;
#ifdef PRINT_USER_INTERACTION_TIME
        rendering::TextConsole::PrintToStaticConsole( "userInteractionTime", "" );
#endif
    }
}

void Engine::FinishedSegmentationSession()
{
    if ( mVolumeLoaded )
    {
        StopSegmentation();
        mSegmenter->FinishedSegmentationSession();
        mFinishedSegmentation = true;

        core::String userInteractionTimeString = core::String( "Total User Interaction Time: %1 s" ).arg( mTotalUserInteractionTimeSeconds, 6 );
#ifdef PRINT_USER_INTERACTION_TIME
        rendering::TextConsole::PrintToStaticConsole( "userInteractionTime", userInteractionTimeString );
#endif
    }
}

void Engine::PlaySegmentation()
{
    if ( mVolumeLoaded )
    {
        mSegmenter->RequestPlaySegmentation();
        mFinishedSegmentation = false;
#ifdef PRINT_USER_INTERACTION_TIME
        rendering::TextConsole::PrintToStaticConsole( "userInteractionTime", "" );
#endif
    }
}

void Engine::StopSegmentation()
{
    if ( mVolumeLoaded )
    {
        mSegmenter->RequestPauseSegmentation();
    }
}

static const int MINIMUM_DISTANCE_TO_ORIGIN = 0.1;

void Engine::MoveCameraAlongViewVector( float distance )
{
    math::Vector3 oldPosition, oldTarget, oldUp;
    rendering::Context::GetCurrentCamera()->GetLookAtVectors( oldPosition, oldTarget, oldUp );

    math::Vector3 viewVector = oldTarget - oldPosition;

    if ( viewVector.Length() > distance + MINIMUM_DISTANCE_TO_ORIGIN )
    {
        viewVector.Normalize();

        math::Vector3 newPosition;
        newPosition = oldPosition + ( distance * viewVector );

        rendering::Context::GetCurrentCamera()->SetLookAtVectors( newPosition, oldTarget, oldUp );
    }
}


void Engine::SetViewport( int height, int width )
{
    rendering::Context::SetCurrentViewport( height, width );
}

static bool sSketchBegun = false;

void Engine::AddSketchPoint( int screenX, int screenY )
{
    if ( sSketchBegun )
    {
        rendering::SketchRenderStrategy* sketchRenderStrategy =
            dynamic_cast< rendering::SketchRenderStrategy* >( rendering::Context::GetCurrentRenderStrategy() );

        if ( sketchRenderStrategy != NULL )
        {
            int virtualScreenX, virtualScreenY;
            int initialViewportWidth, initialViewportHeight;

            rendering::Context::GetInitialViewport( initialViewportWidth, initialViewportHeight );

            rendering::rtgi::GetVirtualScreenCoordinates(
                initialViewportWidth,
                initialViewportHeight,
                screenX,
                screenY,
                virtualScreenX,
                virtualScreenY );

            sketchRenderStrategy->AddSketchPoint( virtualScreenX, virtualScreenY );
        }
    }
}

void Engine::RemoveSketchPoint( int screenX, int screenY )
{
    if ( sSketchBegun )
    {
        rendering::SketchRenderStrategy* sketchRenderStrategy =
            dynamic_cast< rendering::SketchRenderStrategy* >( rendering::Context::GetCurrentRenderStrategy() );

        if ( sketchRenderStrategy != NULL )
        {
            int virtualScreenX, virtualScreenY;
            int initialViewportWidth, initialViewportHeight;

            rendering::Context::GetInitialViewport( initialViewportWidth, initialViewportHeight );

            rendering::rtgi::GetVirtualScreenCoordinates(
                initialViewportWidth,
                initialViewportHeight,
                screenX,
                screenY,
                virtualScreenX,
                virtualScreenY );

            sketchRenderStrategy->RemoveSketchPoint( virtualScreenX, virtualScreenY );
        }
    }
}

void Engine::MoveSketchPoint( int oldScreenX, int oldScreenY, int newScreenX, int newScreenY )
{
    if ( sSketchBegun )
    {
        rendering::SketchRenderStrategy* sketchRenderStrategy =
            dynamic_cast< rendering::SketchRenderStrategy* >( rendering::Context::GetCurrentRenderStrategy() );

        if ( sketchRenderStrategy != NULL )
        {
            int oldVirtualScreenX, oldVirtualScreenY, newVirtualScreenX, newVirtualScreenY;
            int initialViewportWidth, initialViewportHeight;

            rendering::Context::GetInitialViewport( initialViewportWidth, initialViewportHeight );

            rendering::rtgi::GetVirtualScreenCoordinates(
                initialViewportWidth,
                initialViewportHeight,
                oldScreenX,
                oldScreenY,
                oldVirtualScreenX,
                oldVirtualScreenY );

            rendering::rtgi::GetVirtualScreenCoordinates(
                initialViewportWidth,
                initialViewportHeight,
                newScreenX,
                newScreenY,
                newVirtualScreenX,
                newVirtualScreenY );

            sketchRenderStrategy->MoveSketchPoint( oldVirtualScreenX, oldVirtualScreenY, newVirtualScreenX, newVirtualScreenY );
        }
    }
}

void Engine::ClearSketches()
{
    Assert( !sSketchBegun );

    rendering::SketchRenderStrategy* sketchRenderStrategy =
        dynamic_cast< rendering::SketchRenderStrategy* >( rendering::Context::GetCurrentRenderStrategy() );

    if ( sketchRenderStrategy != NULL )
    {
        sketchRenderStrategy->ClearSketches();
    }
}

void Engine::BeginSketch()
{
    if ( !sSketchBegun )
    {
        sSketchBegun = true;

        rendering::SketchRenderStrategy* sketchRenderStrategy =
            dynamic_cast< rendering::SketchRenderStrategy* >( rendering::Context::GetCurrentRenderStrategy() );

        if ( sketchRenderStrategy != NULL )
        {
            sketchRenderStrategy->BeginSketch();
        }
    }
}

void Engine::EndSketch()
{
    if ( sSketchBegun )
    {
        rendering::SketchRenderStrategy* sketchRenderStrategy =
            dynamic_cast< rendering::SketchRenderStrategy* >( rendering::Context::GetCurrentRenderStrategy() );

        if ( sketchRenderStrategy != NULL )
        {
            sketchRenderStrategy->EndSketch();
        }    

        sSketchBegun = false;
    }
}

void Engine::RenderCallback( void* userData )
{
    if ( mScriptLoaded )
    {
        python::ScriptManager::Call( "gpuseg", "RenderCallback" );
    }
}

void Engine::SetAutomaticParameterAdjustEnabled( bool enabled )
{
    mSegmenter->SetAutomaticParameterAdjustEnabled( enabled );
}