#include "Segmenter.hpp"

#include <builtin_types.h>

#include <cudpp.h>

#include "core/Time.hpp"
#include "core/Printf.hpp"

#include "container/List.hpp"

#include "content/Parameter.hpp"

#include "rendering/rtgi/RTGI.hpp"
#include "rendering/rtgi/PixelBuffer.hpp"
#include "rendering/rtgi/FrameBufferObject.hpp"
#include "rendering/rtgi/VertexBuffer.hpp"
#include "rendering/rtgi/IndexBuffer.hpp"
#include "rendering/rtgi/Effect.hpp"

#include "rendering/DebugDraw.hpp"
#include "rendering/TextConsole.hpp"

#include "Config.hpp"
#include "VolumeLoader.hpp"
#include "VolumeFileDesc.hpp"
#include "Engine.hpp"
#include "cuda/Cuda.hpp"
#include "cuda/CudaTextures.hpp"
#include "cuda/CudaKernels.hpp"

#include <cmath>

static content::Parameter< float > sTarget                 ( "Segmenter", "growLevel" );
static content::Parameter< float > sMaxDistanceBeforeShrink( "Segmenter", "growWindow" );
static content::Parameter< float > sCurvatureInfluence     ( "Segmenter", "curvatureInfluence" );
static content::Parameter< float > sTimeStep               ( "Segmenter", "timeStep" );
static content::Parameter< float > sUpdateOpenGLTexture    ( "Segmenter", "updateOpenGLTexture" );
static content::Parameter< float > sNumSubSteps            ( "Segmenter", "numSubSteps" );
static content::Parameter< float > sConvergenceThreshold   ( "Segmenter", "convergenceThreshold" );

Segmenter::Segmenter()
{
    mCoordinatesVolumeDevice          = NULL;
    mValidElementsVolumeDevice        = NULL;
    mTagVolumeDevice                  = NULL;
    mNumValidActiveElementsDevice     = NULL;

    mCompactPlanHandle                = NULL;

    mSourceVolumeTexture              = NULL;
    mCurrentLevelSetVolumeTexture     = NULL;
    mFrozenLevelSetVolumeTexture      = NULL;

    mLevelSetExportPixelBuffer        = NULL;
    mFrameBufferObject                = NULL;

    mLevelSetVolumeDeviceX            = NULL;
    mLevelSetVolumeDeviceY            = NULL;
    mLevelSetVolumeDeviceRead         = NULL;
    mLevelSetVolumeDeviceWrite        = NULL;


    mNumValidActiveElementsHost       = 0;
    mVolumeNumElements                = -1;
    mSourceVolumeNumBytes             = -1;
    mTagVolumeNumBytes                = -1;
    mLevelSetVolumeNumBytes           = -1;

    mNumIterations                    = 0;
    mTargetPrevious                   = 0.0f;
    mMaxDistanceBeforeShrinkPrevious  = 0.0f;
    mCurvatureInfluencePrevious       = 0.0f;
    mTimeStepPrevious                 = 0.0f;

    mEngine = NULL;

    mLevelSetInitialized              = false;
    mCoordinatesVolumeInitialized     = false;
    mComputedSegmentationDetailsOnce  = false;
    mPaused                           = true;
    mCallEngineOnResumeSegmentation   = true;
    mCallEngineOnStopSegmentation     = true;
    mAutomaticParameterAdjustEnabled  = false;

    CudaInitialize();

    mFrameBufferObject = rendering::rtgi::CreateFrameBufferObject();
    mFrameBufferObject->AddRef();

#if defined(PLATFORM_WIN32)
    mTagVolumeEffect = rendering::rtgi::CreateEffect( "effects/TagVolume.cgfx" );
    mTagVolumeEffect->AddRef();
#endif
}

Segmenter::~Segmenter()
{
#if defined(PLATFORM_WIN32)
    mTagVolumeEffect->Release();
    mTagVolumeEffect = NULL;
#endif

    mFrameBufferObject->Release();
    mFrameBufferObject = NULL;

    CudaTerminate();
}

VolumeDesc Segmenter::LoadVolume( const VolumeDesc& volumeDesc )
{
    // Save the volume information before padding
    mVolumeBeforePaddingDesc = volumeDesc;
    mVolumeBeforePaddingDesc.volumeData = NULL;

    VolumeDesc paddedVolumeDesc = LoadVolumeHostPadded( volumeDesc );

    BeforeLoadVolumeDebug( paddedVolumeDesc );

    // pre-compute volume sizes
    mVolumeDimensions       = dim3( paddedVolumeDesc.numVoxelsX, paddedVolumeDesc.numVoxelsY, paddedVolumeDesc.numVoxelsZ );
    mVolumeNumElements      = mVolumeDimensions.x * mVolumeDimensions.y * mVolumeDimensions.z;
    mSourceVolumeNumBytes   = mVolumeNumElements * paddedVolumeDesc.numBytesPerVoxel;
    mLevelSetVolumeNumBytes = mVolumeNumElements * sizeof( CudaLevelSetElement );
    mTagVolumeNumBytes      = mVolumeNumElements * sizeof( CudaTagElement );
    mCompactVolumeNumBytes  = mVolumeNumElements * sizeof( CudaCompactElement );

#ifndef CUDA_30
    // allocate read-only array for volume on the device
    CudaAllocateDeviceArray3D( mVolumeDimensions, &mSourceVolumeArray3DDevice, volumeDesc.numBytesPerVoxel, volumeDesc.isSigned );
#endif

    // allocate read-write scratchpads on the device
#ifdef COMPUTE_PERFORMANCE_METRICS
    CudaAllocateDeviceMemory( mLevelSetVolumeNumBytes, &mLevelSetVolumeDummy1 );
    CudaAllocateDeviceMemory( mLevelSetVolumeNumBytes, &mLevelSetVolumeDummy2 );
    CudaAllocateDeviceMemory( mTagVolumeNumBytes,      &mTagVolumeDummy );
    CudaAllocateDeviceMemory( mCompactVolumeNumBytes,  &mCompactVolumeDummy );
#endif

    CudaAllocateDeviceMemory( mLevelSetVolumeNumBytes, &mLevelSetVolumeDeviceX );
    CudaAllocateDeviceMemory( mLevelSetVolumeNumBytes, &mLevelSetVolumeDeviceY );
    CudaAllocateDeviceMemory( mCompactVolumeNumBytes,  &mCoordinatesVolumeDevice );
    CudaAllocateDeviceMemory( mCompactVolumeNumBytes,  &mValidElementsVolumeDevice );
    CudaAllocateDeviceMemory( mTagVolumeNumBytes,      &mTagVolumeDevice );

    CudaAllocateDeviceMemory( sizeof( size_t ),        &mNumValidActiveElementsDevice );

    // init gpu scratchpad buffers
    CudaMemSet( mLevelSetVolumeDeviceX,         127,        mLevelSetVolumeNumBytes );
    CudaMemSet( mLevelSetVolumeDeviceY,         127,        mLevelSetVolumeNumBytes );
    CudaMemSet( mCoordinatesVolumeDevice,       0xffffffff, mCompactVolumeNumBytes );
    CudaMemSet( mValidElementsVolumeDevice,     0x0,        mCompactVolumeNumBytes );
    CudaMemSet( mTagVolumeDevice,               0x0,        mTagVolumeNumBytes );

    // bind all textures
    CudaBindTextureToBuffer                                           ( CUDA_TEXTURE_TAG_1D,             mTagVolumeDevice );
    CudaBindTextureToBuffer< CudaCompactElement, CudaCompactElement4 >( CUDA_TEXTURE_VALID_ELEMENTS_1D,  mValidElementsVolumeDevice );

#ifndef CUDA_30
    // upload source data
    CudaMemCopyHostToArray3D(
        paddedVolumeDesc.volumeData,
        mSourceVolumeArray3DDevice,
        mVolumeDimensions,
        paddedVolumeDesc.numBytesPerVoxel );
#endif

    // construct plan objects for CUDPP operations
    CUDPPConfiguration configuration;

    configuration.algorithm = CUDPP_COMPACT;
    configuration.datatype  = CUDPP_UINT;
    configuration.options   = CUDPP_OPTION_FORWARD;

    CudppPlan( &mCompactPlanHandle, configuration, mVolumeNumElements, 1, 0 );

    //
    // allocate RTGI textures
    //
    rendering::rtgi::TextureDataDesc textureDataDesc;

    // source data
    textureDataDesc.dimensions                   = rendering::rtgi::TextureDimensions_3D;
    textureDataDesc.pixelFormat                  = GetTexturePixelFormat( volumeDesc );
    textureDataDesc.width                        = mVolumeDimensions.x;
    textureDataDesc.height                       = mVolumeDimensions.y;
    textureDataDesc.depth                        = mVolumeDimensions.z;
    textureDataDesc.data                         = paddedVolumeDesc.volumeData;

    mSourceVolumeTexture = rendering::rtgi::CreateTexture( textureDataDesc );
    mSourceVolumeTexture->AddRef();

    // We want to keep the entire volume description around, so don't delete the volume data
    mVolumeDesc = paddedVolumeDesc;
    //mVolumeDesc = UnloadVolumeHostPadded( paddedVolumeDesc );

    // active elements
    CudaTagElement* activeElementsBufferHost = new CudaTagElement[ mVolumeNumElements ];
    memset( activeElementsBufferHost, 0, mTagVolumeNumBytes );

    textureDataDesc.dimensions  = rendering::rtgi::TextureDimensions_3D;
    textureDataDesc.pixelFormat = rendering::rtgi::TexturePixelFormat_R8_UI_DENORM;
    textureDataDesc.width       = mVolumeDimensions.x;
    textureDataDesc.height      = mVolumeDimensions.y;
    textureDataDesc.depth       = mVolumeDimensions.z;
    textureDataDesc.data        = activeElementsBufferHost;

    mActiveElementsTexture = rendering::rtgi::CreateTexture( textureDataDesc );
    mActiveElementsTexture->AddRef();

    delete[] activeElementsBufferHost;

    // level set data
    CudaLevelSetElement* levelSetBufferHost = new CudaLevelSetElement[ mVolumeNumElements ];
    memset( levelSetBufferHost, 127, mLevelSetVolumeNumBytes );

    textureDataDesc.dimensions  = rendering::rtgi::TextureDimensions_3D;
    textureDataDesc.pixelFormat = rendering::rtgi::TexturePixelFormat_A8_I_NORM;
    textureDataDesc.width       = mVolumeDimensions.x;
    textureDataDesc.height      = mVolumeDimensions.y;
    textureDataDesc.depth       = mVolumeDimensions.z;
    textureDataDesc.data        = levelSetBufferHost;

    mCurrentLevelSetVolumeTexture = rendering::rtgi::CreateTexture( textureDataDesc );
    mCurrentLevelSetVolumeTexture->AddRef();

    mFrozenLevelSetVolumeTexture = rendering::rtgi::CreateTexture( textureDataDesc );
    mFrozenLevelSetVolumeTexture->AddRef();

    delete[] levelSetBufferHost;

    //
    // allocate RTGI buffers
    //

    // vertex buffer
    CudaCompactElement* vertices = new CudaCompactElement[ mVolumeNumElements ];
    memset( vertices, 0, mCompactVolumeNumBytes );

    mActiveElementsCompactedVertexBuffer = rendering::rtgi::CreateVertexBuffer( vertices, mCompactVolumeNumBytes );
    mActiveElementsCompactedVertexBuffer->AddRef();

    delete[] vertices;

    // pixel buffer s
    mLevelSetExportPixelBuffer = rendering::rtgi::CreatePixelBuffer( mTagVolumeNumBytes );
    mLevelSetExportPixelBuffer->AddRef();

    mFrozenLevelSetExportPixelBuffer = rendering::rtgi::CreatePixelBuffer( mTagVolumeNumBytes );
    mFrozenLevelSetExportPixelBuffer->AddRef();

    // register buffers with cuda
    CudaRtgiRegisterBuffer( mActiveElementsCompactedVertexBuffer );
    CudaRtgiRegisterBuffer( mLevelSetExportPixelBuffer );
    CudaRtgiRegisterBuffer( mFrozenLevelSetExportPixelBuffer );

#ifdef CUDA_30
    // register textures with cuda
    CudaRtgiRegisterTexture( mSourceVolumeTexture );
#endif

    CudaTagElement* levelSetExportBuffer;
    CudaTagElement* frozenLevelSetExportBuffer;

    CudaRtgiMapBuffer( &levelSetExportBuffer,       mLevelSetExportPixelBuffer );
    CudaRtgiMapBuffer( &frozenLevelSetExportBuffer, mFrozenLevelSetExportPixelBuffer );    
    CudaMemSet( levelSetExportBuffer,       127, mTagVolumeNumBytes );
    CudaMemSet( frozenLevelSetExportBuffer, 127, mTagVolumeNumBytes );
    CudaRtgiUnmapBuffer( mFrozenLevelSetExportPixelBuffer );
    CudaRtgiUnmapBuffer( mLevelSetExportPixelBuffer );

    mLevelSetVolumeDeviceRead     = mLevelSetVolumeDeviceX;
    mLevelSetVolumeDeviceWrite    = mLevelSetVolumeDeviceY;
    mCoordinatesVolumeInitialized = false;
    mLevelSetInitialized          = false;

#ifdef PRINT_NUM_SEGMENTED_VOXELS
    rendering::TextConsole::PrintToStaticConsole( "numSegmentedVoxels", "" );
#endif

    AfterLoadVolumeDebug();

    return mVolumeDesc;
}

void Segmenter::UnloadVolume()
{
    mLevelSetInitialized          = false;
    mCoordinatesVolumeInitialized = false;
    mLevelSetVolumeDeviceWrite    = NULL;
    mLevelSetVolumeDeviceRead     = NULL;

#ifdef CUDA_30
    CudaRtgiUnregisterTexture( mSourceVolumeTexture );
#endif

    CudaRtgiUnregisterBuffer( mFrozenLevelSetExportPixelBuffer );
    CudaRtgiUnregisterBuffer( mLevelSetExportPixelBuffer );
    CudaRtgiUnregisterBuffer( mActiveElementsCompactedVertexBuffer );

    mFrozenLevelSetExportPixelBuffer->Release();
    mFrozenLevelSetExportPixelBuffer = NULL;

    mLevelSetExportPixelBuffer->Release();
    mLevelSetExportPixelBuffer = NULL;

    mActiveElementsCompactedVertexBuffer->Release();
    mActiveElementsCompactedVertexBuffer = NULL;

    mFrozenLevelSetVolumeTexture->Release();
    mFrozenLevelSetVolumeTexture = NULL;

    mCurrentLevelSetVolumeTexture->Release();
    mCurrentLevelSetVolumeTexture = NULL;

    mActiveElementsTexture->Release();
    mActiveElementsTexture = NULL;

    mSourceVolumeTexture->Release();
    mSourceVolumeTexture = NULL;

    CudppDestroyPlan( mCompactPlanHandle );

    mVolumeDesc = UnloadVolumeHostPadded( mVolumeDesc );

#ifndef CUDA_30
    CudaDeallocateDeviceArray3D( &mSourceVolumeArray3DDevice );
#endif

    CudaDeallocateDeviceMemory( &mLevelSetVolumeDeviceX );
    CudaDeallocateDeviceMemory( &mLevelSetVolumeDeviceY );
    CudaDeallocateDeviceMemory( &mValidElementsVolumeDevice );
    CudaDeallocateDeviceMemory( &mValidElementsVolumeDevice );
    CudaDeallocateDeviceMemory( &mCoordinatesVolumeDevice );
    CudaDeallocateDeviceMemory( &mTagVolumeDevice );
    CudaDeallocateDeviceMemory( &mNumValidActiveElementsDevice );

#ifdef LEFOHN_BENCHMARK
    lefohn::SegmentationSimulator segmentationSimulator( 0 );

    core::Printf( "Iteration Number,\t Number Active Tiles,\t Time Taken in milliseconds \n" );

    int currentIterationNumber = 0;
    foreach( int numActiveTiles, mActiveTileCounter )
    {
        currentIterationNumber++;
        double time = segmentationSimulator.Update( numActiveTiles ) * 1000.0;  // multiply by 1000 to change seconds to milliseconds.
        time += 14.0;   // Add 14ms for downloading of memory allocation message texture.
        core::Printf( core::String( "%1,\t %2,\t %3 \n" ).arg( currentIterationNumber ).arg( numActiveTiles ).arg( time ) );        
    }
#endif
}

// Sets outCollectedValues to be full of the voxel values inside the current level set and
// returns the total value of all voxels collected
float Segmenter::CollectCurrentLevelSetVoxels( container::Array< float > &outCollectedValues )
{
    signed char* hostBuffer = NULL;

    cudaMallocHost( &hostBuffer, mTagVolumeNumBytes );

    Assert( hostBuffer != NULL );

    CudaTagElement* levelSetExport = NULL;

    CudaRtgiMapBuffer( &levelSetExport, mLevelSetExportPixelBuffer );
    CudaMemCopyGlobalToHost( hostBuffer, levelSetExport, mTagVolumeNumBytes );
    CudaRtgiUnmapBuffer( mLevelSetExportPixelBuffer );

    levelSetExport = NULL;

    enum voxelDataTypeEnum { BYTE, UBYTE, SHORT, USHORT, INT, UINT };

    voxelDataTypeEnum voxelDataType;

    if( mVolumeDesc.numBytesPerVoxel == 1 && mVolumeDesc.isSigned )
        voxelDataType = BYTE;
    else if ( mVolumeDesc.numBytesPerVoxel == 1 && !mVolumeDesc.isSigned )
        voxelDataType = UBYTE;
    else if ( mVolumeDesc.numBytesPerVoxel == 2 && mVolumeDesc.isSigned )
        voxelDataType = SHORT;
    else if ( mVolumeDesc.numBytesPerVoxel == 2 && !mVolumeDesc.isSigned )
        voxelDataType = USHORT;
    else if ( mVolumeDesc.numBytesPerVoxel == 4 && mVolumeDesc.isSigned )
        voxelDataType = INT;
    else if ( mVolumeDesc.numBytesPerVoxel == 4 && !mVolumeDesc.isSigned )
        voxelDataType = UINT;

    float currentValue = 0.0;
    float totalValue = 0.0;

    // Collect values
    for( int i = 0; i < mVolumeNumElements; ++i )
    {
        if( hostBuffer[ i ] <= 0 )
        {
            switch( voxelDataType )
            {
                case BYTE:
                {
                    char* volumeData = reinterpret_cast< char* >( mVolumeDesc.volumeData );
                    currentValue = static_cast< float >( volumeData[ i ] );
                    break;
                }
                case UBYTE:
                {
                    unsigned char* volumeData = reinterpret_cast< unsigned char* >( mVolumeDesc.volumeData );
                    currentValue = static_cast< float >( volumeData[ i ] );
                    break;
                }
                case SHORT:
                {
                    short* volumeData = reinterpret_cast< short* >( mVolumeDesc.volumeData );
                    currentValue = static_cast< float >( volumeData[ i ] );
                    break;
                }
                case USHORT:
                {
                    unsigned short* volumeData = reinterpret_cast< unsigned short* >( mVolumeDesc.volumeData );
                    currentValue = static_cast< float >( volumeData[ i ] );
                    break;
                }
                case INT:
                {
                    int* volumeData = reinterpret_cast< int* >( mVolumeDesc.volumeData );
                    currentValue = static_cast< float >( volumeData[ i ] );
                    break;
                }
                case UINT:
                {
                    unsigned int* volumeData = reinterpret_cast< unsigned int* >( mVolumeDesc.volumeData );
                    currentValue = static_cast< float >( volumeData[ i ] );
                    break;
                }
                default:
                    Assert( 0 );
            }

            outCollectedValues.Append( currentValue );
            totalValue += currentValue;
        }
    }

    cudaFreeHost( hostBuffer );

    return totalValue;
}

void Segmenter::CalculateSegmentationParameters()
{
    container::Array< float > collectedLevelSetVoxels;

    float total = CollectCurrentLevelSetVoxels( collectedLevelSetVoxels );

    float numberOfValues = static_cast< float >( collectedLevelSetVoxels.Size() );
    
    float mean = total / numberOfValues;

    // Calculate standard deviation
    float standardDeviation = 0.0;
    foreach( float value, collectedLevelSetVoxels )
    {
        float meanDifference = value - mean;
        standardDeviation += ( meanDifference * meanDifference );
    }

    standardDeviation = sqrt( standardDeviation / ( numberOfValues - 1 ) );

    // growLevel = mean, growWindow = standardDeviation
    sTarget.SetValue( mean );
    sMaxDistanceBeforeShrink.SetValue( standardDeviation );

    if( mEngine )
    {
        mEngine->ParametersChanged();
    }
}

void Segmenter::SetAutomaticParameterAdjustEnabled( bool enable )
{
    mAutomaticParameterAdjustEnabled = enable;
}

bool Segmenter::IsAutomaticParameterAdjustEnabled()
{
    return mAutomaticParameterAdjustEnabled;
}


void Segmenter::InitializeSeed( const math::Vector3& seedCoordinates, unsigned int sphereSize )
{
    //
    // map level set buffer textures into cuda memory space
    //
    CudaTagElement*     levelSetExport          = NULL;
    CudaCompactElement* activeElementsCompacted = NULL;

    CudaRtgiMapBuffer( &levelSetExport, mLevelSetExportPixelBuffer );
    CudaRtgiMapBuffer( &activeElementsCompacted, mActiveElementsCompactedVertexBuffer );

    dim3 seed(
        static_cast< unsigned int >( seedCoordinates[ math::X ] ),
        static_cast< unsigned int >( seedCoordinates[ math::Y ] ),
        static_cast< unsigned int >( seedCoordinates[ math::Z ] ) );

#ifdef REPRODUCIBLE_SEED
    seed.x = 64;
    seed.y = 88;
    seed.z = 98;

    sphereSize = 15;
#endif 

    //
    // clear scratchpad volumes
    //
    CudaMemSet( activeElementsCompacted,    0xffffffff, mCompactVolumeNumBytes );
    CudaMemSet( mValidElementsVolumeDevice, 0x0,        mCompactVolumeNumBytes );
    CudaMemSet( mTagVolumeDevice,           0x0,        mTagVolumeNumBytes );

    //
    // initialize the level set in global gpu memory
    //
    CudaBindTextureToBuffer( CUDA_TEXTURE_LEVEL_SET_1D, mLevelSetVolumeDeviceRead );
    CudaInitializeLevelSetVolume( mLevelSetVolumeDeviceWrite, levelSetExport, mVolumeDimensions, seed, sphereSize, mVolumeDesc.zAnisotropy );
    SwapLevelSetBuffers();

    CudaBindTextureToBuffer( CUDA_TEXTURE_LEVEL_SET_1D, mLevelSetVolumeDeviceRead );
    CudaInitializeLevelSetVolume( mLevelSetVolumeDeviceWrite, levelSetExport, mVolumeDimensions, seed, sphereSize, mVolumeDesc.zAnisotropy );
    SwapLevelSetBuffers();

    //
    // get sparse list of active elements
    //
    CudaBindTextureToBuffer( CUDA_TEXTURE_LEVEL_SET_1D, mLevelSetVolumeDeviceRead );
    CudaInitializeActiveElementsVolume( mValidElementsVolumeDevice, mVolumeDimensions );

    //
    // initialize coordinate buffer
    //
    CudaInitializeCoordinateVolume( mCoordinatesVolumeDevice, mVolumeDimensions );

    //
    // compact the list of active elements
    //
    CudppCompact(
        mCompactPlanHandle,
        mCoordinatesVolumeDevice,
        mValidElementsVolumeDevice,
        activeElementsCompacted,
        mNumValidActiveElementsDevice,
        mVolumeNumElements,
        &mNumValidActiveElementsHost );

    CudaRtgiUnmapBuffer( mActiveElementsCompactedVertexBuffer );
    CudaRtgiUnmapBuffer( mLevelSetExportPixelBuffer );

    activeElementsCompacted = NULL;
    levelSetExport          = NULL;

    UpdateRenderingTextures();

    if( mAutomaticParameterAdjustEnabled )
    {
        CalculateSegmentationParameters();
    }

    mCallEngineOnResumeSegmentation  = true;
    mLevelSetInitialized             = true;
    mCoordinatesVolumeInitialized    = true;
    mComputedSegmentationDetailsOnce = false;
    mNumIterations                   = 0;
    mTargetPrevious                  = sTarget.GetValue();
    mMaxDistanceBeforeShrinkPrevious = sMaxDistanceBeforeShrink.GetValue();
    mCurvatureInfluencePrevious      = sCurvatureInfluence.GetValue();

#ifdef PRINT_NUM_SEGMENTED_VOXELS
    rendering::TextConsole::PrintToStaticConsole( "numSegmentedVoxels", "" );
#endif

#ifdef COMPUTE_PERFORMANCE_METRICS
    mActiveTileCounter.Clear();
    mActiveVoxelCounter.Clear();
    mLevelSetUpdateTimer.Clear();
    mOutputNewActiveVoxelsTimer.Clear();
    mInitializeActiveVoxelsConditionalMemoryWriteTimer.Clear();
    mInitializeActiveVoxelsUnconditionalMemoryWriteTimer.Clear();
    mWriteAndCompactEntireVolumeTimer.Clear();
    mFilterDuplicatesTimer.Clear();
    mCompactTimer.Clear();
    mClearTagVolumeTimer.Clear();
    mClearValidVolumeTimer.Clear();
#endif
}

void Segmenter::ClearCurrentSegmentation()
{
    mCallEngineOnResumeSegmentation    = true;
    mLevelSetInitialized               = false;
    mNumValidActiveElementsHost        = 0;
    mNumIterations                     = 0;
    mComputedSegmentationDetailsOnce   = false;

    //
    // map level set buffer textures into cuda memory space
    //
    CudaLevelSetElement* levelSetExport          = NULL;
    CudaCompactElement*  activeElementsCompacted = NULL;

    CudaRtgiMapBuffer( &levelSetExport,          mLevelSetExportPixelBuffer );
    CudaRtgiMapBuffer( &activeElementsCompacted, mActiveElementsCompactedVertexBuffer );

    CudaMemSet( levelSetExport,          127,        mLevelSetVolumeNumBytes );
    CudaMemSet( activeElementsCompacted, 0xffffffff, mCompactVolumeNumBytes );

    CudaRtgiUnmapBuffer( mActiveElementsCompactedVertexBuffer );
    CudaRtgiUnmapBuffer( mLevelSetExportPixelBuffer );

    activeElementsCompacted = NULL;
    levelSetExport          = NULL;

    //
    // clear the level set volume
    //
    CudaMemSet( mLevelSetVolumeDeviceX,         127,        mLevelSetVolumeNumBytes );
    CudaMemSet( mLevelSetVolumeDeviceY,         127,        mLevelSetVolumeNumBytes );
    CudaMemSet( mValidElementsVolumeDevice,     0x0,        mCompactVolumeNumBytes );
    CudaMemSet( mTagVolumeDevice,               0x0,        mTagVolumeNumBytes );

    UpdateRenderingTextures();
    UpdateRenderingTextures();

#ifdef PRINT_NUM_SEGMENTED_VOXELS
    rendering::TextConsole::PrintToStaticConsole( "numSegmentedVoxels", "" );
#endif

    mEngine->SegmentationFinished();
}

void Segmenter::ClearAllSegmentations()
{
    ClearCurrentSegmentation();

    CudaTagElement* frozenLevelSetExport = NULL;
    CudaRtgiMapBuffer( &frozenLevelSetExport, mFrozenLevelSetExportPixelBuffer );
    CudaMemSet( frozenLevelSetExport, 127, mTagVolumeNumBytes );
    CudaRtgiUnmapBuffer( mFrozenLevelSetExportPixelBuffer );

    frozenLevelSetExport = NULL;

    mFrozenLevelSetVolumeTexture->Update( mFrozenLevelSetExportPixelBuffer );

#ifdef PRINT_NUM_SEGMENTED_VOXELS
    rendering::TextConsole::PrintToStaticConsole( "numSegmentedVoxels", "" );
#endif
}

void Segmenter::FreezeCurrentSegmentation()
{
    CudaTagElement* levelSetExport       = NULL;
    CudaTagElement* frozenLevelSetExport = NULL;

    CudaRtgiMapBuffer( &levelSetExport,       mLevelSetExportPixelBuffer );
    CudaRtgiMapBuffer( &frozenLevelSetExport, mFrozenLevelSetExportPixelBuffer );

    // add current volume to frozen volume
    CudaAddLevelSetVolume( frozenLevelSetExport, levelSetExport, mVolumeDimensions );

    CudaRtgiUnmapBuffer( mFrozenLevelSetExportPixelBuffer );
    CudaRtgiUnmapBuffer( mLevelSetExportPixelBuffer );

    levelSetExport       = NULL;
    frozenLevelSetExport = NULL;

    mFrozenLevelSetVolumeTexture->Update( mFrozenLevelSetExportPixelBuffer );

    //ClearCurrentSegmentation();
}

void Segmenter::FinishedSegmentationSession()
{
    // allocate host buffer
    signed char* hostBuffer = NULL;
    cudaMallocHost( &hostBuffer, mTagVolumeNumBytes );

    Assert( hostBuffer != NULL );

    CudaTagElement* levelSetExport       = NULL;
    CudaTagElement* frozenLevelSetExport = NULL;

    CudaRtgiMapBuffer( &levelSetExport,       mLevelSetExportPixelBuffer );
    CudaRtgiMapBuffer( &frozenLevelSetExport, mFrozenLevelSetExportPixelBuffer );

    // copy level set export into tag volume
    CudaMemCopyGlobalToGlobal( mTagVolumeDevice, levelSetExport, mTagVolumeNumBytes );

    // add current volume to frozen volume
    CudaAddLevelSetVolume( mTagVolumeDevice, frozenLevelSetExport, mVolumeDimensions );

    // copy level set field from device to host
    CudaMemCopyGlobalToHost( hostBuffer, mTagVolumeDevice, mLevelSetVolumeNumBytes );

    // reset the tag volume
    CudaMemSet( mTagVolumeDevice, 0, mTagVolumeNumBytes );

    CudaRtgiUnmapBuffer( mFrozenLevelSetExportPixelBuffer );
    CudaRtgiUnmapBuffer( mLevelSetExportPixelBuffer );

    levelSetExport       = NULL;
    frozenLevelSetExport = NULL;

    int numSegmentedVoxels = 0;

    for ( int i = 0; i < mVolumeNumElements; i++ )
    {
        if ( hostBuffer[ i ] <= 0 )
        {
            numSegmentedVoxels++;
        }
    }

    double volumePerVoxelCubicCentimeters =
        ( mVolumeDesc.voxelWidthMillimeters * mVolumeDesc.voxelHeightMillimeters * mVolumeDesc.voxelDepthMillimeters ) / 1000.0;

    double segmentedVolumeCubicCentimeters =
        volumePerVoxelCubicCentimeters * numSegmentedVoxels;

    core::String numSegmentedVoxelsString =
        core::String( "Total Number of Segmented Voxels: %1 (%2 cc)" ).arg( numSegmentedVoxels ).arg( segmentedVolumeCubicCentimeters );

#ifdef PRINT_NUM_SEGMENTED_VOXELS
    rendering::TextConsole::PrintToStaticConsole( "numSegmentedVoxels", numSegmentedVoxelsString );
#endif

    cudaFreeHost( hostBuffer );
}

void Segmenter::RequestPlaySegmentation()
{
    if ( mPaused )
    {
        mCallEngineOnResumeSegmentation = true;
    }

    mPaused = false;
}

void Segmenter::RequestPauseSegmentation()
{
    if ( mEngine != NULL )
    {
        mEngine->SegmentationFinished();
    }

    mPaused = true;
}

void Segmenter::RequestUpdateSegmentation( double timeDeltaSeconds )
{
    UpdateHostStateBeforeRequestUpdate();

    if ( ShouldRestartSegmentation() )
    {
        UpdateHostStateRestartSegmentation();
    }

    if ( ShouldUpdateSegmentation() )
    {
        UpdateHostStateBeforeUpdateSegmentation();

        for ( int i = 0; i < sNumSubSteps.GetValue(); i++ )
        {
            UpdateHostStateBeforeUpdateSegmentationIteration();

            //
            // map level set buffer textures into cuda memory space
            //
            CudaTagElement*     levelSetExport          = NULL;
            CudaCompactElement* activeElementsCompacted = NULL;
            cudaArray*          sourceArray             = NULL;

            CudaRtgiMapBuffer( &activeElementsCompacted, mActiveElementsCompactedVertexBuffer );
            CudaRtgiMapBuffer( &levelSetExport,          mLevelSetExportPixelBuffer );

#ifdef CUDA_30
            CudaRtgiMapTexture( &sourceArray, mSourceVolumeTexture );
#else
            sourceArray = mSourceVolumeArray3DDevice;
#endif

#ifdef COMPUTE_PERFORMANCE_METRICS
            mActiveVoxelCounter.Append( mNumValidActiveElementsHost );
            core::TimeGetTimeDeltaSeconds();
#endif



            //
            // attach level set texture to whichever level set buffer is the "read" buffer
            //
            CudaBindTextureToArray( GetCudaSourceTexture( mVolumeDesc ), sourceArray );

            CudaBindTextureToBuffer( CUDA_TEXTURE_LEVEL_SET_1D, mLevelSetVolumeDeviceRead );
            CudaBindTextureToBuffer< CudaCompactElement, CudaCompactElement4 >( CUDA_TEXTURE_ACTIVE_ELEMENTS_1D, activeElementsCompacted );
            CudaBindTextureToBuffer< CudaCompactElement, CudaCompactElement4 >( CUDA_TEXTURE_VALID_ELEMENTS_1D,  mValidElementsVolumeDevice );

            //
            // update the level set write volume and time derivative volume.
            //
            CudaUpdateLevelSetVolumeAsync(
                mLevelSetVolumeDeviceWrite,
                levelSetExport,
                mTagVolumeDevice,
                mNumValidActiveElementsHost,
                mVolumeDimensions,
                static_cast< int >( sTarget.GetValue() ),
                static_cast< int >( sMaxDistanceBeforeShrink.GetValue() ),
                sCurvatureInfluence.GetValue(),
                sTimeStep.GetValue(),
                mVolumeDesc.numBytesPerVoxel,
                mVolumeDesc.isSigned );

            //
            // we need the update kernel to finish before proceeding
            //
            CudaSynchronize();



#ifdef COMPUTE_PERFORMANCE_METRICS
            mLevelSetUpdateTimer.Append( core::TimeGetTimeDeltaSeconds() );
            core::TimeGetTimeDeltaSeconds();
#endif



            //
            // swap the read-only and write only buffer pointers and re-bind the level set texture to the read-only level set buffer.
            //
            SwapLevelSetBuffers();
            CudaBindTextureToBuffer( CUDA_TEXTURE_LEVEL_SET_1D, mLevelSetVolumeDeviceRead );

            //
            // if our tweakable parameters have changed, then we need to reinitialize the list of active elements
            //
            if ( ShouldInitializeActiveElements() )
            {
                UpdateHostStateInitializeActiveElements();

                //
                // initialize active elements
                //
                CudaInitializeActiveElementsVolume( mValidElementsVolumeDevice, mVolumeDimensions );

                //
                // initialize coordinate buffer
                //
                if ( ShouldInitializeCoordinates() )
                {
                    UpdateHostStateInitializeCoordinates();
                    CudaInitializeCoordinateVolume( mCoordinatesVolumeDevice, mVolumeDimensions );
                }

                //
                // compact into list of active elements
                //
                CudppCompact(
                    mCompactPlanHandle,
                    mCoordinatesVolumeDevice,
                    mValidElementsVolumeDevice,
                    activeElementsCompacted,
                    mNumValidActiveElementsDevice,
                    mVolumeNumElements,
                    &mNumValidActiveElementsHost );
            }

            //
            // otherwise we let our current active elements spawn new active elements
            //
            else
            {
                if ( ShouldOptimizeForFewActiveVoxels() )
                {
                    UpdateHostStateOptimizeForFewActiveVoxels();

                    unsigned int numElementsTotalAligned = CudaGetWarpAlignedValue( mNumValidActiveElementsHost ) * 7;
                    int          numBytesTotalAligned    = numElementsTotalAligned * sizeof( CudaCompactElement );

                    //
                    // write new active elements to active elements volume
                    //
                    CudaOutputNewActiveElements(
                        mCoordinatesVolumeDevice,
                        mValidElementsVolumeDevice,
                        mNumValidActiveElementsHost,
                        mVolumeDimensions );



#ifdef COMPUTE_PERFORMANCE_METRICS
                    mOutputNewActiveVoxelsTimer.Append( core::TimeGetTimeDeltaSeconds() );
                    core::TimeGetTimeDeltaSeconds();

                    //CudaInitializeActiveElementsVolumeConditionalMemoryWrite( mLevelSetVolumeDummy1, mLevelSetVolumeDummy2, mTagVolumeDummy, mVolumeDimensions );

                    mInitializeActiveVoxelsConditionalMemoryWriteTimer.Append( core::TimeGetTimeDeltaSeconds() );
                    core::TimeGetTimeDeltaSeconds();

                    //CudaInitializeActiveElementsVolumeUnconditionalMemoryWrite( mLevelSetVolumeDummy1, mLevelSetVolumeDummy2, mTagVolumeDummy, mVolumeDimensions );

                    mInitializeActiveVoxelsUnconditionalMemoryWriteTimer.Append( core::TimeGetTimeDeltaSeconds() );
                    core::TimeGetTimeDeltaSeconds();

                    CudaInitializeActiveElementsVolume( mCompactVolumeDummy, mVolumeDimensions );

                    size_t dummyValue;
                    CudppCompact(
                        mCompactPlanHandle,
                        mCompactVolumeDummy,
                        mValidElementsVolumeDevice,
                        mCompactVolumeDummy,
                        mNumValidActiveElementsDevice,
                        mVolumeNumElements,
                        &dummyValue );

                    mWriteAndCompactEntireVolumeTimer.Append( core::TimeGetTimeDeltaSeconds() );
                    core::TimeGetTimeDeltaSeconds();
#endif



                    CudaMemSetCharSparse( mTagVolumeDevice, 0x0, mNumValidActiveElementsHost, mVolumeDimensions );



#ifdef COMPUTE_PERFORMANCE_METRICS
                    mClearTagVolumeTimer.Append( core::TimeGetTimeDeltaSeconds() );
                    core::TimeGetTimeDeltaSeconds();
#endif



                    CudaFilterDuplicates(
                        mCoordinatesVolumeDevice,
                        mValidElementsVolumeDevice,
                        mTagVolumeDevice,
                        mNumValidActiveElementsHost,
                        mVolumeDimensions );



#ifdef COMPUTE_PERFORMANCE_METRICS
                    mFilterDuplicatesTimer.Append( core::TimeGetTimeDeltaSeconds() );
                    core::TimeGetTimeDeltaSeconds();
#endif



                    //
                    // compact into list of active elements
                    //
                    CudppCompact(
                        mCompactPlanHandle,
                        mCoordinatesVolumeDevice,
                        mValidElementsVolumeDevice,
                        activeElementsCompacted,
                        mNumValidActiveElementsDevice,
                        numElementsTotalAligned,
                        &mNumValidActiveElementsHost );

#ifdef COMPUTE_PERFORMANCE_METRICS
                    mCompactTimer.Append( core::TimeGetTimeDeltaSeconds() );
                    core::TimeGetTimeDeltaSeconds();
#endif

                    CudaMemSet( mValidElementsVolumeDevice, 0x0, numBytesTotalAligned );

#ifdef COMPUTE_PERFORMANCE_METRICS
                    mClearValidVolumeTimer.Append( core::TimeGetTimeDeltaSeconds() );
                    core::TimeGetTimeDeltaSeconds();
#endif

                }
                else
                {
                    UpdateHostStateOptimizeForManyActiveVoxels();

                    //
                    // initialize coordinate buffer
                    //
                    if ( ShouldInitializeCoordinates() )
                    {
                        UpdateHostStateInitializeCoordinates();
                        CudaInitializeCoordinateVolume( mCoordinatesVolumeDevice, mVolumeDimensions );
                    }

                    //
                    // write new active elements to active elements volume
                    //
                    CudaUpdateActiveElementsVolume(
                        mValidElementsVolumeDevice,
                        mNumValidActiveElementsHost,
                        mVolumeDimensions );

                    //
                    // compact into list of active elements
                    //
                    CudppCompact(
                        mCompactPlanHandle,
                        mCoordinatesVolumeDevice,
                        mValidElementsVolumeDevice,
                        activeElementsCompacted,
                        mNumValidActiveElementsDevice,
                        mVolumeNumElements,
                        &mNumValidActiveElementsHost );

                    CudaMemSetIntSparse( mValidElementsVolumeDevice, 0x0, mNumValidActiveElementsHost, mVolumeDimensions );
                }
            }

            CudaUnbindTexture( GetCudaSourceTexture( mVolumeDesc ) );

#ifdef CUDA_30
            CudaRtgiUnmapTexture( mSourceVolumeTexture );
#endif

            UpdateHostStateAfterUpdateSegmentationIteration( levelSetExport );

            CudaRtgiUnmapBuffer( mActiveElementsCompactedVertexBuffer );
            CudaRtgiUnmapBuffer( mLevelSetExportPixelBuffer );

            sourceArray             = NULL;
            activeElementsCompacted = NULL;
            levelSetExport          = NULL;
        }

        if ( ShouldUpdateRenderingTextures() )
        {
            UpdateRenderingTextures();
        }

        UpdateHostStateAfterUpdateSegmentation();
    }
    else
    {
        UpdateHostStateDoNotUpdateSegmentation();
    }

    UpdateHostStateAfterRequestUpdate( timeDeltaSeconds );
}

bool Segmenter::IsSegmentationFinished()
{
    return mNumValidActiveElementsHost <= sConvergenceThreshold.GetValue() && mNumIterations > MINIMUM_SEGMENTATION_ITERATIONS;
}

bool Segmenter::IsSegmentationInitialized()
{
    return mLevelSetInitialized;
}

bool Segmenter::IsSegmentationInProgress()
{
    return ShouldUpdateSegmentation();
}

rendering::rtgi::Texture* Segmenter::GetSourceVolumeTexture()
{
    return mSourceVolumeTexture;
}

rendering::rtgi::Texture* Segmenter::GetCurrentLevelSetVolumeTexture()
{
    return mCurrentLevelSetVolumeTexture;
}

rendering::rtgi::Texture* Segmenter::GetFrozenLevelSetVolumeTexture()
{
    return mFrozenLevelSetVolumeTexture;
}

rendering::rtgi::Texture* Segmenter::GetActiveElementsVolumeTexture()
{
    return mActiveElementsTexture;
}

VolumeDesc Segmenter::GetVolumeDesc()
{
    return mVolumeDesc;
}

size_t Segmenter::GetActiveElementCount()
{
    return mNumValidActiveElementsHost;
}

void Segmenter::SetEngine( Engine* engine )
{
    AssignRef( mEngine, engine );
}

rendering::rtgi::TexturePixelFormat Segmenter::GetTexturePixelFormat( const VolumeDesc& volumeDesc )
{
    if ( volumeDesc.numBytesPerVoxel == 1 && !volumeDesc.isSigned ) return rendering::rtgi::TexturePixelFormat_R8_UI_DENORM;
    if ( volumeDesc.numBytesPerVoxel == 1 &&  volumeDesc.isSigned ) return rendering::rtgi::TexturePixelFormat_R8_I_DENORM;
    if ( volumeDesc.numBytesPerVoxel == 2 && !volumeDesc.isSigned ) return rendering::rtgi::TexturePixelFormat_R16_UI_DENORM;
    if ( volumeDesc.numBytesPerVoxel == 2 &&  volumeDesc.isSigned ) return rendering::rtgi::TexturePixelFormat_R16_I_DENORM;
    if ( volumeDesc.numBytesPerVoxel == 4 && !volumeDesc.isSigned ) return rendering::rtgi::TexturePixelFormat_R32_UI_DENORM;
    if ( volumeDesc.numBytesPerVoxel == 4 &&  volumeDesc.isSigned ) return rendering::rtgi::TexturePixelFormat_R32_I_DENORM;

    Assert( 0 );

    return rendering::rtgi::TexturePixelFormat_Invalid;
}

const char* Segmenter::GetCudaSourceTexture( const VolumeDesc& volumeDesc )
{
    if ( volumeDesc.numBytesPerVoxel == 1 && !volumeDesc.isSigned ) return CUDA_TEXTURE_SOURCE_3D_UI8;
    if ( volumeDesc.numBytesPerVoxel == 1 &&  volumeDesc.isSigned ) return CUDA_TEXTURE_SOURCE_3D_I8;
    if ( volumeDesc.numBytesPerVoxel == 2 && !volumeDesc.isSigned ) return CUDA_TEXTURE_SOURCE_3D_UI16;
    if ( volumeDesc.numBytesPerVoxel == 2 &&  volumeDesc.isSigned ) return CUDA_TEXTURE_SOURCE_3D_I16;
    if ( volumeDesc.numBytesPerVoxel == 4 && !volumeDesc.isSigned ) return CUDA_TEXTURE_SOURCE_3D_UI32;
    if ( volumeDesc.numBytesPerVoxel == 4 &&  volumeDesc.isSigned ) return CUDA_TEXTURE_SOURCE_3D_I32;

    Assert( 0 );

    return "";
}

VolumeDesc Segmenter::LoadVolumeHostPadded( const VolumeDesc& volumeDesc )
{
    //
    // compute num voxels to pad in each dimension
    //
    unsigned int numPadVoxelsX = ( VOLUME_ALIGNMENT_X - ( volumeDesc.numVoxelsX % VOLUME_ALIGNMENT_X ) ) % VOLUME_ALIGNMENT_X;
    unsigned int numPadVoxelsY = ( VOLUME_ALIGNMENT_Y - ( volumeDesc.numVoxelsY % VOLUME_ALIGNMENT_Y ) ) % VOLUME_ALIGNMENT_Y;
    unsigned int numPadVoxelsZ = ( VOLUME_ALIGNMENT_Z - ( volumeDesc.numVoxelsZ % VOLUME_ALIGNMENT_Z ) ) % VOLUME_ALIGNMENT_Z;

    unsigned int numVoxelsAfterPadX = volumeDesc.numVoxelsX + numPadVoxelsX;
    unsigned int numVoxelsAfterPadY = volumeDesc.numVoxelsY + numPadVoxelsY;
    unsigned int numVoxelsAfterPadZ = volumeDesc.numVoxelsZ + numPadVoxelsZ;

    Assert( numVoxelsAfterPadX <= 1024 && numVoxelsAfterPadY <= 1024 && numVoxelsAfterPadZ <= 1024 );

    unsigned int numPadBytesX = numPadVoxelsX * volumeDesc.numBytesPerVoxel;
    unsigned int numPadBytesY = numPadVoxelsY * numVoxelsAfterPadX * volumeDesc.numBytesPerVoxel;
    unsigned int numPadBytesZ = numPadVoxelsZ * numVoxelsAfterPadY * numVoxelsAfterPadX * volumeDesc.numBytesPerVoxel;

    unsigned int numBytesBeforePadX = volumeDesc.numVoxelsX * volumeDesc.numBytesPerVoxel;

    //
    // allocate space for padded volume
    //
    unsigned int   volumeDataPaddedNumVoxels = numVoxelsAfterPadX * numVoxelsAfterPadY * numVoxelsAfterPadZ;
    unsigned int   volumeDataPaddedNumBytes  = volumeDataPaddedNumVoxels * volumeDesc.numBytesPerVoxel;
    unsigned char* volumeDataPadded          = new unsigned char[ volumeDataPaddedNumBytes ];

    memset( volumeDataPadded, 0, volumeDataPaddedNumBytes );

    //
    // copy and pad
    //
    unsigned int paddedIndex = 0;
    unsigned int rawIndex    = 0;
    int          padValue    = volumeDesc.isSigned ? -127 : 0;

    for ( unsigned int z = 0; z < volumeDesc.numVoxelsZ; z++ )
    {
        for ( unsigned int y = 0; y < volumeDesc.numVoxelsY; y++ )
        {
            // copy each scan line
            memcpy( volumeDataPadded + paddedIndex, ((unsigned char*)volumeDesc.volumeData) + rawIndex, numBytesBeforePadX );

            paddedIndex += numBytesBeforePadX;
            rawIndex    += numBytesBeforePadX;

            // then pad each scan line
            memset( volumeDataPadded + paddedIndex, padValue, numPadBytesX );

            paddedIndex += numPadBytesX;
        }

        // pad the space at the end of each slice
        memset( volumeDataPadded + paddedIndex, padValue, numPadBytesY );

        paddedIndex += numPadBytesY;
    }

    // pad the space at the end of the volume
    memset( volumeDataPadded + paddedIndex, padValue, numPadBytesZ );

    paddedIndex += numPadBytesZ;

    Assert( paddedIndex == volumeDataPaddedNumBytes );

    VolumeDesc paddedVolumeDesc = volumeDesc;
    paddedVolumeDesc.numVoxelsX = numVoxelsAfterPadX;
    paddedVolumeDesc.numVoxelsY = numVoxelsAfterPadY;
    paddedVolumeDesc.numVoxelsZ = numVoxelsAfterPadZ;
    paddedVolumeDesc.volumeData = volumeDataPadded;

    return paddedVolumeDesc;
}

VolumeDesc Segmenter::UnloadVolumeHostPadded( const VolumeDesc& volumeDesc )
{
    delete[] volumeDesc.volumeData;

    VolumeDesc unloadedVolumeDesc = volumeDesc;
    unloadedVolumeDesc.volumeData = NULL;

    return unloadedVolumeDesc;
}

VolumeDesc Segmenter::GetSaveableSegmentation()
{
    // Grab the combine level set volume (frozen and active).
    signed char* hostBuffer = NULL;
    cudaMallocHost( &hostBuffer, mTagVolumeNumBytes );

    Assert( hostBuffer != NULL );

    CudaTagElement* levelSetExport       = NULL;
    CudaTagElement* frozenLevelSetExport = NULL;

    CudaRtgiMapBuffer( &levelSetExport,       mLevelSetExportPixelBuffer );
    CudaRtgiMapBuffer( &frozenLevelSetExport, mFrozenLevelSetExportPixelBuffer );

    // copy level set export into tag volume
    CudaMemCopyGlobalToGlobal( mTagVolumeDevice, levelSetExport, mTagVolumeNumBytes );

    // add current volume to frozen volume
    CudaAddLevelSetVolume( mTagVolumeDevice, frozenLevelSetExport, mVolumeDimensions );

    // copy level set field from device to host
    CudaMemCopyGlobalToHost( hostBuffer, mTagVolumeDevice, mLevelSetVolumeNumBytes );

    // reset the tag volume
    CudaMemSet( mTagVolumeDevice, 0, mTagVolumeNumBytes );

    CudaRtgiUnmapBuffer( mFrozenLevelSetExportPixelBuffer );
    CudaRtgiUnmapBuffer( mLevelSetExportPixelBuffer );

    levelSetExport       = NULL;
    frozenLevelSetExport = NULL;

    int minValue = 0;
    int maxValue = 0;

    for ( int i = 0; i < mVolumeNumElements; i++ )
    {
        if ( hostBuffer[ i ] <= 0 )
        {
            hostBuffer[ i ] = 1;
            maxValue        = 1;
        }
        else
        {
            hostBuffer[ i ] = 0;
        }
    }

    VolumeDesc paddedSegmentation;

    // Values based on the data we get from above
    paddedSegmentation.minValue         = minValue;
    paddedSegmentation.maxValue         = maxValue;
    paddedSegmentation.volumeData       = hostBuffer;
    paddedSegmentation.numBytesPerVoxel = 1;
    paddedSegmentation.isSigned         = true;
    
    // Values based on the loaded source volume
    paddedSegmentation.numVoxelsX  = mVolumeDesc.numVoxelsX;
    paddedSegmentation.numVoxelsY  = mVolumeDesc.numVoxelsY;
    paddedSegmentation.numVoxelsZ  = mVolumeDesc.numVoxelsZ;
    paddedSegmentation.upDirection = mVolumeDesc.upDirection;
    paddedSegmentation.zAnisotropy = mVolumeDesc.zAnisotropy;
    paddedSegmentation.filePaths   = mVolumeDesc.filePaths;

    VolumeDesc unpaddedSegmentation = UnpadVolume( paddedSegmentation );

    cudaFreeHost( hostBuffer );

    return unpaddedSegmentation;
}

VolumeDesc Segmenter::UnpadVolume( const VolumeDesc& volumeDesc )
{
    unsigned int unpaddedVolumeNumVoxels = mVolumeBeforePaddingDesc.numVoxelsX * mVolumeBeforePaddingDesc.numVoxelsY * mVolumeBeforePaddingDesc.numVoxelsZ;
    unsigned int unpaddedVolumeNumBytes = unpaddedVolumeNumVoxels * volumeDesc.numBytesPerVoxel;

    unsigned int numBytesBeforePadX = mVolumeBeforePaddingDesc.numVoxelsX * volumeDesc.numBytesPerVoxel;

    unsigned int paddedDifferenceY = volumeDesc.numVoxelsY - mVolumeBeforePaddingDesc.numVoxelsY;

    unsigned int numBytesPaddedX = volumeDesc.numVoxelsX * volumeDesc.numBytesPerVoxel;
    unsigned int numBytesPaddedY = paddedDifferenceY * volumeDesc.numVoxelsX * volumeDesc.numBytesPerVoxel;

    unsigned char* unpaddedVolumeData = new unsigned char[ unpaddedVolumeNumBytes ];

    memset( unpaddedVolumeData, 0, unpaddedVolumeNumBytes );

    unsigned int unpaddedIndex = 0;
    unsigned int paddedIndex = 0;

    for ( unsigned int z = 0; z < mVolumeBeforePaddingDesc.numVoxelsZ; z++ )
    {
        for ( unsigned int y = 0; y < mVolumeBeforePaddingDesc.numVoxelsY; y++ )
        {
            memcpy( unpaddedVolumeData + unpaddedIndex, reinterpret_cast< unsigned char* >( volumeDesc.volumeData ) + paddedIndex, numBytesBeforePadX );
            unpaddedIndex += numBytesBeforePadX;
            // Skip anything padded at the end of a scanline
            paddedIndex += numBytesPaddedX;
        }
        // Skip anything padded at the bottom of each slice
        paddedIndex += numBytesPaddedY;
    }

    // Don't even bother with the padding at the end of the volume.

    // Copy over all the unpadded volume attributes and return
    VolumeDesc unpaddedVolumeDesc;
    unpaddedVolumeDesc.volumeData       = unpaddedVolumeData;
    unpaddedVolumeDesc.isSigned         = volumeDesc.isSigned;
    unpaddedVolumeDesc.maxValue         = volumeDesc.maxValue;
    unpaddedVolumeDesc.minValue         = volumeDesc.minValue;
    unpaddedVolumeDesc.numBytesPerVoxel = volumeDesc.numBytesPerVoxel;
    unpaddedVolumeDesc.upDirection      = volumeDesc.upDirection;
    unpaddedVolumeDesc.zAnisotropy      = volumeDesc.zAnisotropy;
    unpaddedVolumeDesc.numVoxelsX       = mVolumeBeforePaddingDesc.numVoxelsX;
    unpaddedVolumeDesc.numVoxelsY       = mVolumeBeforePaddingDesc.numVoxelsY;
    unpaddedVolumeDesc.numVoxelsZ       = mVolumeBeforePaddingDesc.numVoxelsZ;
    unpaddedVolumeDesc.filePaths        = mVolumeBeforePaddingDesc.filePaths;

    return unpaddedVolumeDesc;
}

void Segmenter::SwapLevelSetBuffers()
{
    if ( mLevelSetVolumeDeviceRead == mLevelSetVolumeDeviceX )
    {
        mLevelSetVolumeDeviceRead  = mLevelSetVolumeDeviceY;
        mLevelSetVolumeDeviceWrite = mLevelSetVolumeDeviceX;
    }
    else
    {
        mLevelSetVolumeDeviceRead  = mLevelSetVolumeDeviceX;
        mLevelSetVolumeDeviceWrite = mLevelSetVolumeDeviceY;
    }
}

bool Segmenter::ShouldRestartSegmentation()
{
    return ShouldInitializeActiveElements();
}

bool Segmenter::ShouldUpdateSegmentation()
{
    return ( ( IsSegmentationInitialized() && !IsSegmentationFinished() ) || ( ShouldInitializeActiveElements() ) ) && !mPaused;
}

bool Segmenter::ShouldOptimizeForFewActiveVoxels()
{
    unsigned int numElementsTotalAligned = CudaGetWarpAlignedValue( mNumValidActiveElementsHost ) * 7;
    int          numBytesTotalAligned    = numElementsTotalAligned * sizeof( CudaCompactElement );

    return numBytesTotalAligned < mCompactVolumeNumBytes;
}

bool Segmenter::ShouldInitializeActiveElements()
{
    return mTargetPrevious != sTarget.GetValue() || mMaxDistanceBeforeShrinkPrevious != sMaxDistanceBeforeShrink.GetValue() || mCurvatureInfluencePrevious != sCurvatureInfluence.GetValue();
}

bool Segmenter::ShouldInitializeCoordinates()
{
    return !mCoordinatesVolumeInitialized;
}

bool Segmenter::ShouldUpdateRenderingTextures()
{
    return true; //sUpdateOpenGLTexture.GetValue() == 1.0f;
}

void Segmenter::UpdateHostStateBeforeRequestUpdate()
{
}

void Segmenter::UpdateHostStateAfterRequestUpdate( double timeDeltaSeconds )
{
    double framesPerSecond          = 1.0f / timeDeltaSeconds;
    double frameTimeMilliseconds    = timeDeltaSeconds * 1000.0f;
    double simulationStepsPerSecond;

    if ( ShouldUpdateSegmentation() )
    {
        simulationStepsPerSecond = sNumSubSteps.GetValue() / timeDeltaSeconds;
    }
    else
    {
        simulationStepsPerSecond = 0;
    }

    core::String frameTimeString, simulationTimeString, activeVoxelString;

    simulationTimeString = core::String( "Level Set Iterations Per Second: %1" ).arg( simulationStepsPerSecond );
    activeVoxelString    = core::String( "Active Voxels: %1" ).arg( mNumValidActiveElementsHost );

#ifdef PRINT_NUM_ACTIVE_VOXELS
    rendering::TextConsole::PrintToStaticConsole( "activeVoxels", activeVoxelString );
#endif
#ifdef PRINT_SIMULATION_TIME
    rendering::TextConsole::PrintToStaticConsole( "simulationTime", simulationTimeString );
#endif
}


void Segmenter::UpdateHostStateRestartSegmentation()
{
    mNumIterations = 0;
}

void Segmenter::UpdateHostStateBeforeUpdateSegmentation()
{
    if ( ShouldInitializeActiveElements() )
    {
        mNumIterations                  = 0;
        mCallEngineOnResumeSegmentation = true;
    }

    if ( mCallEngineOnResumeSegmentation && mEngine != NULL )
    {
        mEngine->SegmentationStarted();

        mCallEngineOnStopSegmentation   = true;
        mCallEngineOnResumeSegmentation = false;
    }
}

void Segmenter::UpdateHostStateAfterUpdateSegmentation()
{
#ifdef PRINT_USER_INTERACTION_TIME
    rendering::TextConsole::PrintToStaticConsole( "userInteractionTime", "" );
#endif
}

void Segmenter::UpdateHostStateDoNotUpdateSegmentation()
{
    if ( mEngine != NULL && mCallEngineOnStopSegmentation )
    {
        mEngine->SegmentationFinished();

        mCallEngineOnStopSegmentation = false;



#ifdef COMPUTE_PERFORMANCE_METRICS
        PrintSegmentationDetails();
#endif



    }
}

void Segmenter::UpdateHostStateBeforeUpdateSegmentationIteration()
{
}

void Segmenter::UpdateHostStateAfterUpdateSegmentationIteration( const CudaTagElement* levelSetExportBuffer )
{
    mNumIterations++;

    if ( ( ( mNumValidActiveElementsHost == 0 ) || ( mNumIterations == 1800 ) ) && !mComputedSegmentationDetailsOnce )
    {
        mComputedSegmentationDetailsOnce = true;

        PrintSegmentationDetails();
        ComputeSegmentationAccuracy( levelSetExportBuffer );
    }

#ifdef LEFOHN_BENCHMARK
    //
    // Copy the level set data to the host
    //
    CudaMemCopyGlobalToHost( sLevelSetScratchpadHost,
        mLevelSetVolumeDeviceRead, mLevelSetVolumeNumBytes );

    //
    // Compute the number of active tiles
    //
    int numActiveTiles = lefohn::ComputeNumberOfActiveTiles(
        sLevelSetScratchpadHost );

    //
    // Store this number in an array for subsequent use by the lefohn system.
    //
    mActiveTileCounter.Append( numActiveTiles );

    core::Printf( core::String( "%1 iterations,\t %2 active tiles,\t %3 active voxels \n").arg( mNumIterations ).arg( numActiveTiles ).arg( mNumValidActiveElementsHost ) );

    // Hack to stop at a breakpoint.
    if( mNumIterations == 2243 )
    {
        int iNeedABreakPointNOW = 50000;
    }
#endif
}

void Segmenter::UpdateHostStateOptimizeForFewActiveVoxels()
{
    mCoordinatesVolumeInitialized = false;
}

void Segmenter::UpdateHostStateOptimizeForManyActiveVoxels()
{
}

void Segmenter::UpdateHostStateInitializeActiveElements()
{
    mTargetPrevious                  = sTarget.GetValue();
    mMaxDistanceBeforeShrinkPrevious = sMaxDistanceBeforeShrink.GetValue();
    mCurvatureInfluencePrevious      = sCurvatureInfluence.GetValue();
    mNumIterations                   = 0;
}

void Segmenter::UpdateHostStateInitializeCoordinates()
{
    mCoordinatesVolumeInitialized = true;
}

void Segmenter::UpdateRenderingTextures()
{
    if ( sUpdateOpenGLTexture.GetValue() > 0.0f )
    {
#if defined(PLATFORM_WIN32)
        //
        // active elements...
        //
        ClearTexture( mActiveElementsTexture );
        TagTextureSparse( mActiveElementsTexture, 127.0f );
    
#elif defined(PLATFORM_OSX)
        // Cannot currently do this on OSX
#endif
    }

    //
    // level set...
    //
    mCurrentLevelSetVolumeTexture->Update( mLevelSetExportPixelBuffer );
}

void Segmenter::TagTextureSparse( rendering::rtgi::Texture* texture, float tagValue )
{
    rendering::rtgi::VertexDataSourceDesc vertexDataSourceDesc;

    vertexDataSourceDesc.numCoordinatesPerSemantic = 1;
    vertexDataSourceDesc.numVertices               = mNumValidActiveElementsHost;
    vertexDataSourceDesc.offset                    = 0;
    vertexDataSourceDesc.stride                    = sizeof( float );
    vertexDataSourceDesc.vertexBuffer              = mActiveElementsCompactedVertexBuffer;
    vertexDataSourceDesc.vertexBufferDataType      = rendering::rtgi::VertexBufferDataType_Float;

    container::Map< core::String, rendering::rtgi::VertexDataSourceDesc > vertexDataSources;

    vertexDataSources.Insert( "POSITION", vertexDataSourceDesc );

    math::Matrix44 orthographicProjectionMatrix;
    orthographicProjectionMatrix.SetTo2DOrthographic( 0, mVolumeDimensions.x, 0, mVolumeDimensions.y );

    mFrameBufferObject->Bind( texture );

    mTagVolumeEffect->BindVertexDataSources( vertexDataSources );
    mTagVolumeEffect->BeginSetEffectParameters();
    mTagVolumeEffect->SetEffectParameter( "tagValue",                     tagValue );
    mTagVolumeEffect->SetEffectParameter( "orthographicProjectionMatrix", orthographicProjectionMatrix );
    mTagVolumeEffect->EndSetEffectParameters();

    while ( mTagVolumeEffect->BindPass() )
    {
        rendering::rtgi::SetPointSize( 1 );
        rendering::rtgi::VertexBuffer::Render( rendering::rtgi::PrimitiveRenderMode_Points, mNumValidActiveElementsHost );

        mTagVolumeEffect->UnbindPass();
    }

    mTagVolumeEffect->UnbindVertexDataSources();

    mFrameBufferObject->Unbind();
}

void Segmenter::ClearTexture( rendering::rtgi::Texture* texture )
{
    mFrameBufferObject->Bind( texture );

    rendering::rtgi::ClearColorBuffer( rendering::rtgi::ColorRGBA( 0, 0, 0, 0 ) );

    mFrameBufferObject->Unbind();
}

void Segmenter::BeforeLoadVolumeDebug( const VolumeDesc& volumeDesc )
{
    //
    // debug output
    //
    unsigned int freeMemory, totalMemory;
    unsigned int numMegabytes =
        volumeDesc.numVoxelsX * volumeDesc.numVoxelsY * volumeDesc.numVoxelsZ * volumeDesc.numBytesPerVoxel / ( 1024 * 1024 );

    CudaGetMemoryInfo( &freeMemory, &totalMemory );
    core::Printf( "\n" );
    core::Printf( core::String( "data set size in bytes = %1 * %2 * %3 * %4 Bytes = %5 MBytes\n" )
        .arg( volumeDesc.numVoxelsX )
        .arg( volumeDesc.numVoxelsY )
        .arg( volumeDesc.numVoxelsZ )
        .arg( volumeDesc.numBytesPerVoxel )
        .arg( numMegabytes ) );

    core::Printf( "\n" );
    core::Printf( "before initialization of CUDA buffers...\n");
    core::Printf( core::String( "    current memory available = %1 MBytes\n" ).arg( freeMemory  / ( 1024 * 1024 ) ) );
    core::Printf( core::String( "    total memory available   = %1 MBytes\n" ).arg( totalMemory / ( 1024 * 1024 ) ) );
    core::Printf( "\n" );
}

void Segmenter::AfterLoadVolumeDebug()
{
    unsigned int freeMemory, totalMemory;

    CudaGetMemoryInfo( &freeMemory, &totalMemory );

    core::Printf( "after initialization of CUDA buffers...\n");
    core::Printf( core::String( "    current memory available = %1 MBytes\n" ).arg(  freeMemory / ( 1024 * 1024 ) ) );
    core::Printf( core::String( "    total memory available   = %1 MBytes\n" ).arg( totalMemory / ( 1024 * 1024 ) ) );
    core::Printf( "\n" );
}

void Segmenter::PrintSegmentationDetails()
{
#ifdef COMPUTE_PERFORMANCE_METRICS

    core::Printf( "begin print num active voxels\n" );

    foreach( int i, mActiveVoxelCounter )
    {
        core::Printf( core::String( "%1\n" ).arg( i ) );
    }

    core::Printf( "end print num active voxels\n" );



    core::Printf( "begin print level set update times\n" );

    foreach( double d, mLevelSetUpdateTimer )
    {
        core::Printf( core::String( "%1\n" ).arg( d ) );
    }

    core::Printf( "end print level set update times\n" );



    core::Printf( "begin print output active voxel times\n" );

    foreach( double d, mOutputNewActiveVoxelsTimer )
    {
        core::Printf( core::String( "%1\n" ).arg( d ) );
    }

    core::Printf( "end print output active voxel times\n" );



    core::Printf( "begin print filter duplicate times\n" );

    foreach( double d, mFilterDuplicatesTimer )
    {
        core::Printf( core::String( "%1\n" ).arg( d ) );
    }

    core::Printf( "end print filter duplicate times\n" );



    core::Printf( "begin print compact times\n" );

    foreach( double d, mCompactTimer )
    {
        core::Printf( core::String( "%1\n" ).arg( d ) );
    }

    core::Printf( "end print compact times\n" );



    core::Printf( "begin print clear tag volume times\n" );

    foreach( double d, mClearTagVolumeTimer )
    {
        core::Printf( core::String( "%1\n" ).arg( d ) );
    }

    core::Printf( "end print clear tag volume times\n" );



    core::Printf( "begin print clear valid volume times\n" );

    foreach( double d, mClearValidVolumeTimer )
    {
        core::Printf( core::String( "%1\n" ).arg( d ) );
    }

    core::Printf( "end print clear valid volume times\n" );



    core::Printf( "begin print initialize active voxels (conditional memory write) times\n" );

    foreach( double d, mInitializeActiveVoxelsConditionalMemoryWriteTimer )
    {
        core::Printf( core::String( "%1\n" ).arg( d ) );
    }

    core::Printf( "end print initialize active voxels (conditional memory write) times\n" );



    core::Printf( "begin print initialize active voxels (unconditional memory write) times\n" );

    foreach( double d, mInitializeActiveVoxelsUnconditionalMemoryWriteTimer )
    {
        core::Printf( core::String( "%1\n" ).arg( d ) );
    }

    core::Printf( "end print initialize active voxels (unconditional memory write) times\n" );



    core::Printf( "begin print compact entire volume times\n" );

    foreach( double d, mWriteAndCompactEntireVolumeTimer )
    {
        core::Printf( core::String( "%1\n" ).arg( d ) );
    }

    core::Printf( "end print compact entire volume times\n" );

#endif
}

void Segmenter::ComputeSegmentationAccuracy( const CudaTagElement* levelSetExportBuffer )
{
#ifdef COMPUTE_ACCURACY_METRICS
    Assert( levelSetExportBuffer != NULL );

    // allocate host buffer
    CudaTagElement* hostBufferPadded = NULL;
    cudaMallocHost( &hostBufferPadded, mLevelSetVolumeNumBytes );

    Assert( hostBufferPadded != NULL );

    // copy level set field from device to host
    CudaMemCopyGlobalToHost( hostBufferPadded, levelSetExportBuffer, mLevelSetVolumeNumBytes );

    // load the ground truth
    VolumeFileDesc groundTruthVolumeFileDesc;

    groundTruthVolumeFileDesc.fileName         = "data/brainweb/phantom_1.0mm_normal_crisp.raw";
    groundTruthVolumeFileDesc.isSigned         = false;
    groundTruthVolumeFileDesc.numBytesPerVoxel = 1;
    groundTruthVolumeFileDesc.numVoxelsX       = 181;
    groundTruthVolumeFileDesc.numVoxelsY       = 217;
    groundTruthVolumeFileDesc.numVoxelsZ       = 181;


    VolumeDesc groundTruthVolumeDesc = VolumeLoader::LoadVolume( groundTruthVolumeFileDesc );

    // test each voxel against ground truth
    int truePositives  = 0;
    int trueNegatives  = 0;
    int falsePositives = 0;
    int falseNegatives = 0;

    int GREY_MATTER  = 2;
    int WHITE_MATTER = 3;

    // now test voxel by voxel
    unsigned int paddedIndex   = 0;
    unsigned int unpaddedIndex = 0;

    for ( unsigned int z = 0; z < groundTruthVolumeDesc.numVoxelsZ; z++ )
    {
        for ( unsigned int y = 0; y < groundTruthVolumeDesc.numVoxelsY; y++ )
        {
            // test each scan line
            for ( unsigned int x = 0; x < groundTruthVolumeDesc.numVoxelsX; x++ )
            {
                char* groundTruthVolumeData = reinterpret_cast< char* >( groundTruthVolumeDesc.volumeData ) + unpaddedIndex;
                char* levelSetData          = reinterpret_cast< char* >( hostBufferPadded + paddedIndex );
                char  levelSetDataValue     = *levelSetData;
                char  groundTruthDataValue  = *groundTruthVolumeData;

                bool insideLevelSetSurface = levelSetDataValue <= 0;

#ifdef GROUND_TRUTH_WHITE_MATTER
                bool insideGroundTruthROI  = groundTruthDataValue == WHITE_MATTER;
#elif GROUND_TRUTH_WHITE_AND_GREY_MATTER
                bool insideGroundTruthROI  = groundTruthDataValue == WHITE_MATTER || groundTruthDataValue == GREY_MATTER;
#endif

                if ( insideLevelSetSurface && insideGroundTruthROI )
                {
                    truePositives++;
                }

                if ( !insideLevelSetSurface && !insideGroundTruthROI )
                {
                    trueNegatives++;
                }

                if ( insideLevelSetSurface && !insideGroundTruthROI )
                {
                    falsePositives++;
                }

                if ( !insideLevelSetSurface && insideGroundTruthROI )
                {
                    falseNegatives++;
                }

                paddedIndex++;
                unpaddedIndex++;
            }

            paddedIndex += mVolumeDesc.numVoxelsX - groundTruthVolumeDesc.numVoxelsX;
        }

        paddedIndex += ( mVolumeDesc.numVoxelsY - groundTruthVolumeDesc.numVoxelsY ) * mVolumeDesc.numVoxelsX;
    }

    core::Printf( "finished segmentation\n" );
    core::Printf( core::String( "%1\n" ).arg( truePositives ) );
    core::Printf( core::String( "%1\n" ).arg( trueNegatives ) );
    core::Printf( core::String( "%1\n" ).arg( falsePositives ) );
    core::Printf( core::String( "%1\n" ).arg( falseNegatives ) );

    core::Printf( "\n" );

    core::Printf( core::String( "%1\n" ).arg( static_cast< double >( truePositives ) / static_cast< double >( truePositives + falseNegatives + falsePositives ) ) );

    core::Printf( "\n" );

    core::Printf( core::String( "%1\n" ).arg( static_cast< double >( truePositives + trueNegatives ) / static_cast< double >( truePositives + + trueNegatives + falseNegatives + falsePositives ) ) );

    core::Printf( "\n" );
    core::Printf( "\n" );

    cudaFreeHost( hostBufferPadded );
#endif
}

