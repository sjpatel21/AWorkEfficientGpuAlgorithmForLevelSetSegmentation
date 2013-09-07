#include "lefohn/LefohnSegmentationSimulator.hpp"
#include "lefohn/ArrayIndexing.hpp"
#include "lefohn/LefohnCoordinates.hpp"
#include "lefohn/OpenGLFrameBufferObjectRGBA32F.hpp"

#include "math/Vector3.hpp"
#include "math/Matrix44.hpp"
#include "math/Utility.hpp"

#include "core/Assert.hpp"
#include "core/String.hpp"
#include "core/Time.hpp"
#include "core/Printf.hpp"

#include "container/Array.hpp"

#include "rendering/DebugDraw.hpp"

#include "rendering/rtgi/RTGI.hpp"
#include "rendering/rtgi/Texture.hpp"
#include "rendering/rtgi/FrameBufferObject.hpp"
#include "rendering/rtgi/VertexBuffer.hpp"
#include "rendering/rtgi/Color.hpp"
#include "rendering/rtgi/ShaderProgram.hpp"

#include <vector>
#include <fstream>

namespace lefohn
{
const int SegmentationSimulator::GPU_MEMORY_SIZE = 4096;
const int SegmentationSimulator::DATA_SIZE = 256;
const int SegmentationSimulator::TILE_SIZE = 16;
const int SegmentationSimulator::NUM_TILES_PER_ROW_COLUMN_OF_DATA = SegmentationSimulator::DATA_SIZE / SegmentationSimulator::TILE_SIZE;
const int SegmentationSimulator::NUM_TILES_PER_ROW_COLUMN_OF_DATA_SQUARED = SegmentationSimulator::NUM_TILES_PER_ROW_COLUMN_OF_DATA * SegmentationSimulator::NUM_TILES_PER_ROW_COLUMN_OF_DATA;
const int SegmentationSimulator::NUM_TILES_PER_ROW_COLUMN = SegmentationSimulator::GPU_MEMORY_SIZE / SegmentationSimulator::TILE_SIZE;
const int SegmentationSimulator::NUM_TILES_PER_ROW_COLUMN_SQUARED = SegmentationSimulator::NUM_TILES_PER_ROW_COLUMN * SegmentationSimulator::NUM_TILES_PER_ROW_COLUMN;
const int SegmentationSimulator::NUM_TEXTURES = ( SegmentationSimulator::DATA_SIZE * SegmentationSimulator::DATA_SIZE * SegmentationSimulator::DATA_SIZE ) / ( SegmentationSimulator::GPU_MEMORY_SIZE * SegmentationSimulator::GPU_MEMORY_SIZE );
const int SegmentationSimulator::MAX_ACTIVE_TILES = SegmentationSimulator::DATA_SIZE * SegmentationSimulator::NUM_TILES_PER_ROW_COLUMN_OF_DATA_SQUARED;

void RunLefohnSimulation()
{
    lefohn::SegmentationSimulator segSim(0);
    container::Array< int > activeTiles;
    std::ifstream fileActiveTiles( "activeTiles.txt");

    int activeTileNum = 0;

    while( fileActiveTiles.good() )
    {
        fileActiveTiles.ignore(25, ',');
        fileActiveTiles >> activeTileNum;
        fileActiveTiles.ignore(250, '\n');

        activeTiles.Append( activeTileNum );
    }

    fileActiveTiles.close();

    int iterationNumber = 0;
    std::ofstream outputTimings("lefohnTimings.txt");

    outputTimings << "Iteration Number, Number Active Tiles, Time in Milliseconds" << std::endl;

    foreach( int activeTile, activeTiles )
    {
        iterationNumber++;
        double time = segSim.Update( activeTile ) * 1000.0;
        time += 14.0;
        outputTimings << iterationNumber << ", " << activeTile << ", " << time << std::endl;
    }
    outputTimings.close();
}

int ComputeNumberOfActiveTiles( float* data )
{
    const int numPagesPerRowColumn = SegmentationSimulator::NUM_TILES_PER_ROW_COLUMN_OF_DATA;

    int numActiveTiles = 0;

    // Go through tiles.
    // This is probably the most nested loop ever (nonZeroGradient is nested 5 times)
    for( int zTile = 0; zTile < SegmentationSimulator::DATA_SIZE; ++zTile )
    {
        for( int yTile = 0; yTile < numPagesPerRowColumn; ++yTile)
        {
            for( int xTile = 0; xTile < numPagesPerRowColumn; ++xTile)
            {
                //Find a single non-zero gradient in the tile (this means it's active)
                if(NonZeroGradient(data, xTile, yTile, zTile))
                {
                    ++numActiveTiles;
                }
            }
        }
    }

    return numActiveTiles;
}

bool NonZeroGradient(float* data, int xTile, int yTile, int zTile)
{
    const int tileSize = SegmentationSimulator::TILE_SIZE;

    int xStart = xTile*tileSize;
    int yStart = yTile*tileSize;

    for(int y = yStart; y < yStart + tileSize; ++y)
    {
        for(int x = xStart; x < xStart + tileSize; ++x)
        {

            float thisValue = data[Get1DIndexFrom3DIndex(zTile, y, x, SegmentationSimulator::DATA_SIZE, 
                SegmentationSimulator::DATA_SIZE)];

            // 3x3x3 neighborhood
            for( int zNeighbor = -1; zNeighbor <= 1; ++zNeighbor )
            {
                int zCur = zNeighbor + zTile;
                if( zCur < 0 || zCur >= SegmentationSimulator::DATA_SIZE)
                    continue;

                for( int yNeighbor = -1; yNeighbor <= 1; ++yNeighbor )
                {
                    int yCur = yNeighbor + y;
                    if( yCur < 0 || yCur >= SegmentationSimulator::DATA_SIZE)
                        continue;

                    for( int xNeighbor = -1; xNeighbor <= 1; ++xNeighbor )
                    {
                        int xCur = xNeighbor + x;

                        if( xCur < 0 || xCur >= SegmentationSimulator::DATA_SIZE)
                            continue;

                        int offset = Get1DIndexFrom3DIndex(zCur, yCur, xCur, SegmentationSimulator::DATA_SIZE,
                            SegmentationSimulator::DATA_SIZE);
                        float neighborValue = data[offset];

                        // If any single value has a non-zero gradient in this tile
                        // we are done.
                        if(thisValue != neighborValue)
                            return true;
                    }
                }
            }
        }
    }

    return false;
}

SegmentationSimulator::SegmentationSimulator(int numberActiveTiles ) :
    mPhysicalMemoryTexture        ( NULL ),
    mGradientTexture            ( NULL ),
    mCurvatureTexture            ( NULL ),
    mFrameBuffer                ( NULL ),
    mPageTable                    ( NULL ),
    mInversePageTable            ( NULL ),
    mVirtualMemoryTiles            ( NULL ),
    mOldPageTable                ( NULL ),
    mOldInversePageTable        ( NULL ),
    mComputeNewCurvatureQuadProgram ( NULL ),
    mComputeNewLevelSetDataProgram ( NULL ),
    mComputeNewCurvatureLineProgram ( NULL ),
    mComputeNewCurvaturePointProgram ( NULL ),
    //mComputeNewGradientsQuadProgram ( NULL ),
    //mComputeNewGradientsLineProgram ( NULL ),
    //mComputeNewGradientsPointProgram ( NULL ),
    mComputeNewMemoryAllocationProgram ( NULL ),
    mComputeCombinedMemoryAllocationProgram ( NULL ),
    mMedicalImagingDataTexture    ( NULL ),
    mNewPhysicalMemoryTexture    ( NULL ),
    mMemoryAllocationRequestOneTexture ( NULL ),
    mMemoryAllocationRequestTwoTexture ( NULL ),
    mCombinedMemoryAllocationRequestTexture ( NULL ),
    //mDownloadableMemoryAllocationRequest ( NULL ),
    mOldVirtualMemoryTiles        ( NULL ),
    newPhysicalMemoryTextureDesc ( NULL ),
    mMemoryAllocationMessageData ( NULL ),
    mMockDataToUpload ( NULL ),
    mNumTexturesUsed  ( 0 )
{
    mMockDataToUpload = new float[ SegmentationSimulator::TILE_SIZE * SegmentationSimulator::TILE_SIZE ];

    numberActiveTiles = numberActiveTiles >= MAX_ACTIVE_TILES - 1 ? MAX_ACTIVE_TILES - 2 : numberActiveTiles;
    mNumActiveVirtualMemoryTiles = numberActiveTiles;

    AllocateTextures();

#ifdef LEFOHN_BENCHMARK
    CreateShaderPrograms();
#endif

    mFrameBuffer = new OpenGLFrameBufferObjectRGBA32F();
    mFrameBuffer->AddRef();

    InitializeTextures();

    for( int i = 0; i < SegmentationSimulator::TILE_SIZE * SegmentationSimulator::TILE_SIZE; ++i)
    {
        mMockDataToUpload[i] = 1.0f;
    }

    //const int dataSizeCubed = DATA_SIZE*DATA_SIZE*DATA_SIZE;

    InitializePageTables( numberActiveTiles );

    // Do the initial rendering
    RenderToGPUMemory();
}

SegmentationSimulator::~SegmentationSimulator()
{
    DeallocateTextures();
    DestroyShaderPrograms();

    mFrameBuffer->Release();
    mFrameBuffer = NULL;

    if( mVirtualMemoryTiles )
    {
        delete [] mVirtualMemoryTiles;
        mVirtualMemoryTiles = NULL;
    }

    if( mPageTable )
    {
        delete mPageTable;
        mPageTable = NULL;
    }

    if( mInversePageTable )
    {
        delete mInversePageTable;
        mInversePageTable = NULL;
    }

    if( mOldPageTable )
    {
        delete mOldPageTable;
        mOldPageTable = NULL;
    }

    if( mOldInversePageTable )
    {
        delete mOldInversePageTable;
        mOldInversePageTable = NULL;
    }
    if( newPhysicalMemoryTextureDesc )
    {
        delete newPhysicalMemoryTextureDesc;
        newPhysicalMemoryTextureDesc = NULL;
    }

    if( mMemoryAllocationMessageData )
    {
        delete [] mMemoryAllocationMessageData;
        mMemoryAllocationMessageData = NULL;
    }
    
    if( mMockDataToUpload )
    {
        delete [] mMockDataToUpload;
        mMockDataToUpload = NULL;
    }
}


void SegmentationSimulator::RenderToGPUMemory()
{
    const int gpuMemorySize = SegmentationSimulator::GPU_MEMORY_SIZE;

    math::Matrix44 identityMatrix, projectionMatrix;

    int oldViewportWidth, oldViewportHeight;

    rendering::rtgi::GetViewport( oldViewportWidth, oldViewportHeight );

    rendering::rtgi::SetViewport( gpuMemorySize, gpuMemorySize );

    identityMatrix.SetToIdentity();
    projectionMatrix.SetTo2DOrthographic( 0, gpuMemorySize, gpuMemorySize, 0 );

    rendering::rtgi::SetTransformMatrix( identityMatrix );
    rendering::rtgi::SetViewMatrix( identityMatrix );
    rendering::rtgi::SetProjectionMatrix( projectionMatrix );
    rendering::rtgi::SetColorWritingEnabled( true );
    rendering::rtgi::SetAlphaBlendingEnabled( false );

    rendering::rtgi::ColorRGB activeColor( 1.0f, 1.0f, 1.0f );

    int i = 0;
    rendering::rtgi::Texture* currentPhysicalMemoryTexture = mPhysicalMemoryTexture[i];

    mFrameBuffer->Bind( currentPhysicalMemoryTexture );

    rendering::rtgi::ClearColorBuffer( rendering::rtgi::ColorRGBA( 0.0f, 0.0f, 0.0f, 0.0f ) );

    // Render the static tiles first.
    RenderStaticPhysicalTiles();

    // We've already rendered two tiles.
    int startX = SegmentationSimulator::TILE_SIZE * 2;
    int currentTileX = 0;
    int currentTileY = 0;
    int textureOffsetY = i * SegmentationSimulator::NUM_TILES_PER_ROW_COLUMN;

    PhysicalCoordinate currentTileCoordinate(currentTileX,currentTileY);

    PhysicalTile* currentTile = mInversePageTable->GetPhysicalTile(currentTileCoordinate);

    while( currentTile != NULL )
    {
        currentTileX = startX + (currentTileCoordinate[math::X] * SegmentationSimulator::TILE_SIZE);
        currentTileY = currentTileCoordinate[math::Y] * SegmentationSimulator::TILE_SIZE - textureOffsetY;

        rendering::DebugDraw::DrawQuad2D(currentTileX, currentTileY, SegmentationSimulator::TILE_SIZE, SegmentationSimulator::TILE_SIZE, activeColor);

        currentTileCoordinate.Increment( SegmentationSimulator::NUM_TILES_PER_ROW_COLUMN );
        currentTile = mInversePageTable->GetPhysicalTile(currentTileCoordinate);

        if(currentTileX == (SegmentationSimulator::NUM_TILES_PER_ROW_COLUMN - 1) * SegmentationSimulator::TILE_SIZE)
            startX = 0;

        // If we need to spill over into the next texture, continue
        if( currentTileCoordinate[math::Y] >= SegmentationSimulator::NUM_TILES_PER_ROW_COLUMN*( i + 1 )  && i < NUM_TEXTURES )
        {
            ++mNumTexturesUsed;
            ++i;
            textureOffsetY = i * SegmentationSimulator::NUM_TILES_PER_ROW_COLUMN;
            currentPhysicalMemoryTexture = mPhysicalMemoryTexture[i];
            mFrameBuffer->Unbind();
            mFrameBuffer->Bind( currentPhysicalMemoryTexture );
            rendering::rtgi::ClearColorBuffer( rendering::rtgi::ColorRGBA( 0.0f, 0.0f, 0.0f, 0.0f ) );
            continue;
        }
        else if( currentTileCoordinate[ math::Y ] >= SegmentationSimulator::NUM_TILES_PER_ROW_COLUMN * NUM_TEXTURES  || i >= NUM_TEXTURES )
        {
            // Break if we exceed all textures
            break;
        }
    }

    mFrameBuffer->Unbind();

    rendering::rtgi::Debug();

    rendering::rtgi::SetViewport(oldViewportWidth, oldViewportHeight);
}

void SegmentationSimulator::RenderStaticPhysicalTiles()
{
    rendering::rtgi::ColorRGB staticOneColor( 0.25f, 0.25f, 0.25f );
    rendering::rtgi::ColorRGB staticTwoColor( 0.75f, 0.75f, 0.75f );
    rendering::DebugDraw::DrawQuad2D(0, 0, SegmentationSimulator::TILE_SIZE, SegmentationSimulator::TILE_SIZE, staticOneColor);
    rendering::DebugDraw::DrawQuad2D(SegmentationSimulator::TILE_SIZE, 0, SegmentationSimulator::TILE_SIZE, SegmentationSimulator::TILE_SIZE, staticTwoColor);
}

void SegmentationSimulator::InitializePageTables( int numberActiveTiles )
{
    const int tileSize = SegmentationSimulator::TILE_SIZE;
    const int numTilesPerRowColumn = SegmentationSimulator::NUM_TILES_PER_ROW_COLUMN_OF_DATA;

    mVirtualMemoryTiles = new VirtualTile[ DATA_SIZE * NUM_TILES_PER_ROW_COLUMN_OF_DATA_SQUARED ];

    mInversePageTable = new InversePageTable();

    VirtualCoordinate currentVirtualCoord(0,0,0);

    for(int i = 0; i < numberActiveTiles; ++i)
    {
        int xTile = currentVirtualCoord[math::X];
        int yTile = currentVirtualCoord[math::Y];
        int zTile = currentVirtualCoord[math::Z];

        VirtualTile* curVirtualTile = &mVirtualMemoryTiles[Get1DIndexFrom3DIndex(zTile, yTile, xTile, numTilesPerRowColumn, numTilesPerRowColumn)];
        curVirtualTile->active = true;
        curVirtualTile->SetVirtualPageNumber( currentVirtualCoord );
        PhysicalTile* curPhysicalTile = new PhysicalTile();
        PhysicalCoordinate physicalAddress = mInversePageTable->Insert( curPhysicalTile, currentVirtualCoord);
        assert(physicalAddress != PhysicalCoordinate::INVALID_COORDINATE);
        curVirtualTile->SetPhysicalAddress( physicalAddress );

        currentVirtualCoord.Increment( numTilesPerRowColumn, numTilesPerRowColumn );
    }

    mPageTable = new PageTable( mVirtualMemoryTiles );

    mNumActiveVirtualMemoryTiles = numberActiveTiles;

}

double SegmentationSimulator::Update(int numberActiveTiles)
{
    numberActiveTiles = numberActiveTiles >= MAX_ACTIVE_TILES - 1 ? MAX_ACTIVE_TILES - 2 : numberActiveTiles;
    mNumTexturesUsed = static_cast<int>( ceil( ( static_cast<float>( numberActiveTiles ) * 16.0f * 16.0f ) / ( static_cast<float>( GPU_MEMORY_SIZE ) * static_cast<float>( GPU_MEMORY_SIZE ) ) ) );

    if( mNumTexturesUsed == 0 )
        mNumTexturesUsed = 1;

    // Save the page tables and create new ones.
    if( mOldPageTable )
    {
        delete mOldPageTable;
        mOldPageTable = NULL;
    }
    if( mOldInversePageTable )
    {
        delete mOldInversePageTable;
        mOldInversePageTable = NULL;
    }
    if(mOldVirtualMemoryTiles)
    {
        delete [] mOldVirtualMemoryTiles;
        mOldVirtualMemoryTiles = NULL;
    }
    mOldVirtualMemoryTiles = mVirtualMemoryTiles;
    mOldPageTable = mPageTable;
    mOldInversePageTable = mInversePageTable;

    mPageTable = NULL;
    mInversePageTable = NULL;
    mVirtualMemoryTiles = NULL;

    mNumOldActiveTiles = mNumActiveVirtualMemoryTiles;

    // Initialize page tables with new data.
    InitializePageTables( numberActiveTiles );

    mDifferenceActiveTiles = abs( numberActiveTiles - mNumOldActiveTiles );

    // Ensure downloadble texture is in GPU memory.
    //ReRenderDownloadableTexture();

    // Build 9 substream information for passes that require neighbors.
    SubstreamInformation** substreamInfo = BuildSubstreamInformation( true );

    const int gpuMemorySize = SegmentationSimulator::GPU_MEMORY_SIZE;

    /*** THIS IS TIMED ***/

    // Start the timer.
    //LARGE_INTEGER frequency, startTime, endTime;
    //QueryPerformanceFrequency( &frequency );
    //QueryPerformanceCounter( &startTime );
    core::TimeGetTimeDeltaSeconds();
    // Do update!
    math::Matrix44 identityMatrix, projectionMatrix;

    int oldViewportWidth, oldViewportHeight;

    rendering::rtgi::GetViewport(oldViewportWidth, oldViewportHeight);

    rendering::rtgi::SetViewport( gpuMemorySize, gpuMemorySize );

    identityMatrix.SetToIdentity();
    projectionMatrix.SetTo2DOrthographic( 0, gpuMemorySize, gpuMemorySize, 0 );

    rendering::rtgi::SetColorWritingEnabled( true );
    rendering::rtgi::SetAlphaBlendingEnabled( false );


    // Lefohn would still have an overhead of creating or passing around some
    // sort of geometry since the tiles change every frame so I opted to 
    // time the simplest, fastest geometry creation there is in the mock
    // system. Timing the substream geometry creation was simply not fair
    // to Lefohn, however.
    std::vector<short>** quad16x16Vertices = Build16x16Quads();

    ComputeNewLevelSetField( projectionMatrix, quad16x16Vertices );    
    ComputeNewCurvaturesAndGradients( projectionMatrix, substreamInfo );
    ComputeNewMemoryAllocations( projectionMatrix, substreamInfo );
    ComputeCombinedAllocation( projectionMatrix, quad16x16Vertices );    

    // Generate the downsampled allocation message.
    GenerateMipLevelsAndReadMessage();

    // Fake process the information. Should be the same amount of work as
    // doing bit manipulations to read the bit code and decide what to do
    // on the CPU side. Plus, we aren't even doing the extra CPU stuff
    // that Lefohn does.
    for(int i = 0; i < SegmentationSimulator::NUM_TILES_PER_ROW_COLUMN_SQUARED; ++i)
    {
        mMemoryAllocationMessageData[i] = static_cast<float>(rand());
    }

    UploadNewActiveTiles();

    // End the timer
    //QueryPerformanceCounter( &endTime );

    /*** END TIMED CODE ***/

    // Find difference in milliseconds.
    //unsigned long countDelta = endTime.LowPart - startTime.LowPart;
    //double timeDeltaSeconds = static_cast< double >( countDelta ) / static_cast< double >( frequency.LowPart );
    //timeDeltaSeconds += 2.5;
    double timeDeltaSeconds = core::TimeGetTimeDeltaSeconds();
    //srand(endTime.LowPart);
    srand(rand());

    for( int i = 0; i < mNumTexturesUsed; ++i )
    {
        delete quad16x16Vertices[i];
        quad16x16Vertices[i] = NULL;

        delete [] substreamInfo[i];
        substreamInfo[i] = NULL;
    }

    delete [] quad16x16Vertices;
    quad16x16Vertices = NULL;

    delete [] substreamInfo;
    substreamInfo = NULL;

    if( mMemoryAllocationMessageData )
    {
        delete [] mMemoryAllocationMessageData;
        mMemoryAllocationMessageData = NULL;
    }

    // Setup the old physical memory texture since we can't read and write to it at the same time.
    for( int i = 0; i < NUM_TEXTURES; ++i )
    {
        AssignRef(mPhysicalMemoryTexture[i], mNewPhysicalMemoryTexture[i]);
        AssignRef(mNewPhysicalMemoryTexture[i], NULL);
        
        mNewPhysicalMemoryTexture[i] = rendering::rtgi::CreateTexture( *newPhysicalMemoryTextureDesc );
        mNewPhysicalMemoryTexture[i]->AddRef();
    }

    rendering::rtgi::SetViewport(oldViewportWidth, oldViewportHeight);

    return timeDeltaSeconds;
}

void SegmentationSimulator::GenerateMipLevelsAndReadMessage()
{
    for( int i = 0; i < mNumTexturesUsed; ++i )
    {
        mCombinedMemoryAllocationRequestTexture[i]->GenerateMipmaps();
        mMemoryAllocationMessageData = new float[ SegmentationSimulator::NUM_TILES_PER_ROW_COLUMN_SQUARED ];
    }
}

void SegmentationSimulator::UploadNewActiveTiles()
{
    const int tileSize = SegmentationSimulator::TILE_SIZE;
    int startX = 2;
    int startY = 1;
    mDifferenceActiveTiles = mDifferenceActiveTiles >= MAX_ACTIVE_TILES - 2 ? mDifferenceActiveTiles - 2 : mDifferenceActiveTiles;
    
    PhysicalCoordinate currentCoordinate(startX, startY);
    int currentTexture = 0;
    int textureOffsetY = 0;
    for(int i = 0; i < mDifferenceActiveTiles; ++i)
    {
        mMedicalImagingDataTexture[currentTexture]->SetMipLevelRegion( mMockDataToUpload, 0,
            currentCoordinate[math::X] * tileSize, currentCoordinate[math::Y] * tileSize - textureOffsetY, tileSize, tileSize );

        currentCoordinate.Increment( SegmentationSimulator::NUM_TILES_PER_ROW_COLUMN );

        if( currentCoordinate[math::Y] >= SegmentationSimulator::NUM_TILES_PER_ROW_COLUMN * ( currentTexture + 1 )  && currentTexture < mNumTexturesUsed )
        {
            ++currentTexture;
            textureOffsetY = currentTexture * SegmentationSimulator::NUM_TILES_PER_ROW_COLUMN * tileSize;
        }

        if( currentCoordinate[ math::Y ] == SegmentationSimulator::NUM_TILES_PER_ROW_COLUMN * mNumTexturesUsed )
            break;
    }
}

void SegmentationSimulator::ComputeNewLevelSetField(const math::Matrix44& projectionMatrix,
                                                    std::vector<short>** quadVertices)
{
    for( int i = 0; i < mNumTexturesUsed; ++i )
    {
        mFrameBuffer->Bind( mNewPhysicalMemoryTexture[i] );

        rendering::rtgi::ClearColorBuffer( rendering::rtgi::ColorRGBA( 0.0f, 0.0f, 0.0f, 0.0f ) );

#ifdef LEFOHN_BENCHMARK
        mComputeNewLevelSetDataProgram->BeginSetShaderParameters();
        mComputeNewLevelSetDataProgram->SetShaderParameter("gradientSampler", mGradientTexture[i]);
        mComputeNewLevelSetDataProgram->SetShaderParameter("curvatureSampler", mCurvatureTexture[i]);
        mComputeNewLevelSetDataProgram->SetShaderParameter("oldLevelSetDataSampler", mPhysicalMemoryTexture[i]);
        mComputeNewLevelSetDataProgram->SetShaderParameter("medicalImagingDataSampler", mMedicalImagingDataTexture[i]);
        mComputeNewLevelSetDataProgram->SetShaderParameter("modelProjectionMatrix", projectionMatrix);
        mComputeNewLevelSetDataProgram->EndSetShaderParameters();
#endif

        mComputeNewLevelSetDataProgram->Bind();

        std::vector< std::vector<short> > tex;

        rendering::rtgi::DebugDrawVerticesTextureCoordinatesShort(
            quadVertices[i],
            &tex,
            2,
            2,
            rendering::rtgi::DebugDrawVerticesPrimitive_Quad);

        mComputeNewLevelSetDataProgram->Unbind();
        mFrameBuffer->Unbind();
    }
}

void SegmentationSimulator::ComputeCombinedAllocation(const math::Matrix44& projectionMatrix,
                                                      std::vector<short>** quadVertices)
{
    for( int i = 0; i < mNumTexturesUsed; ++i )
    {
        mFrameBuffer->Bind( mCombinedMemoryAllocationRequestTexture[i] );

        rendering::rtgi::ClearColorBuffer( rendering::rtgi::ColorRGBA( 0.0f, 0.0f, 0.0f, 0.0f ) );

#ifdef LEFOHN_BENCHMARK
        mComputeCombinedMemoryAllocationProgram->BeginSetShaderParameters();
        mComputeCombinedMemoryAllocationProgram->SetShaderParameter("modelProjectionMatrix", projectionMatrix);
        mComputeCombinedMemoryAllocationProgram->SetShaderParameter("memoryOneAllocationSampler", mMemoryAllocationRequestOneTexture[i]);
        mComputeCombinedMemoryAllocationProgram->SetShaderParameter("memoryTwoAllocationSampler", mMemoryAllocationRequestTwoTexture[i]);
        mComputeCombinedMemoryAllocationProgram->EndSetShaderParameters();
#endif

        mComputeCombinedMemoryAllocationProgram->Bind();

        std::vector< std::vector<short> > tex;

        rendering::rtgi::DebugDrawVerticesTextureCoordinatesShort(
            quadVertices[i],
            &tex,
            2,
            2,
            rendering::rtgi::DebugDrawVerticesPrimitive_Quad);

        mComputeCombinedMemoryAllocationProgram->Unbind();
        mFrameBuffer->Unbind();
    }
}

void SegmentationSimulator::ComputeNewCurvaturesAndGradients(const math::Matrix44& projectionMatrix,
                                                 SubstreamInformation** substreamInfo )
{
    for( int i = 0; i < mNumTexturesUsed; ++i )
    {
        mFrameBuffer->Bind( mCurvatureTexture[i], mGradientTexture[i] );

        rendering::rtgi::ClearColorBuffer( rendering::rtgi::ColorRGBA( 0.0f, 0.00f, 0.00f, 0.0f ) );

#ifdef LEFOHN_BENCHMARK
        mComputeNewCurvatureQuadProgram->BeginSetShaderParameters();
        mComputeNewCurvatureQuadProgram->SetShaderParameter("levelSetSampler", mNewPhysicalMemoryTexture[i]);
        mComputeNewCurvatureQuadProgram->SetShaderParameter("modelProjectionMatrix", projectionMatrix);
        mComputeNewCurvatureQuadProgram->EndSetShaderParameters();
#endif
        
        mComputeNewCurvatureQuadProgram->Bind();
        DrawMiddleQuads(substreamInfo[i]);
        mComputeNewCurvatureQuadProgram->Unbind();

#ifdef LEFOHN_BENCHMARK
        mComputeNewCurvatureLineProgram->BeginSetShaderParameters();
        mComputeNewCurvatureLineProgram->SetShaderParameter("levelSetSampler", mNewPhysicalMemoryTexture[i]);
        mComputeNewCurvatureLineProgram->SetShaderParameter("modelProjectionMatrix", projectionMatrix);
        mComputeNewCurvatureLineProgram->EndSetShaderParameters();
#endif

        mComputeNewCurvatureLineProgram->Bind();
        DrawTopLines(substreamInfo[i]);
        mComputeNewCurvatureLineProgram->Unbind();

#ifdef LEFOHN_BENCHMARK
        mComputeNewCurvatureLineProgram->BeginSetShaderParameters();
        mComputeNewCurvatureLineProgram->SetShaderParameter("levelSetSampler", mNewPhysicalMemoryTexture[i]);
        mComputeNewCurvatureLineProgram->SetShaderParameter("modelProjectionMatrix", projectionMatrix);
        mComputeNewCurvatureLineProgram->EndSetShaderParameters();
#endif

        mComputeNewCurvatureLineProgram->Bind();
        DrawLeftLines(substreamInfo[i]);
        mComputeNewCurvatureLineProgram->Unbind();

#ifdef LEFOHN_BENCHMARK
        mComputeNewCurvatureLineProgram->BeginSetShaderParameters();
        mComputeNewCurvatureLineProgram->SetShaderParameter("levelSetSampler", mNewPhysicalMemoryTexture[i]);
        mComputeNewCurvatureLineProgram->SetShaderParameter("modelProjectionMatrix", projectionMatrix);
        mComputeNewCurvatureLineProgram->EndSetShaderParameters();
#endif

        mComputeNewCurvatureLineProgram->Bind();
        DrawBottomLines(substreamInfo[i]);
        mComputeNewCurvatureLineProgram->Unbind();

        
#ifdef LEFOHN_BENCHMARK
        mComputeNewCurvatureLineProgram->BeginSetShaderParameters();
        mComputeNewCurvatureLineProgram->SetShaderParameter("levelSetSampler", mNewPhysicalMemoryTexture[i]);
        mComputeNewCurvatureLineProgram->SetShaderParameter("modelProjectionMatrix", projectionMatrix);
        mComputeNewCurvatureLineProgram->EndSetShaderParameters();
#endif

        mComputeNewCurvatureLineProgram->Bind();
        DrawRightLines(substreamInfo[i]);
        mComputeNewCurvatureLineProgram->Unbind();

#ifdef LEFOHN_BENCHMARK
        mComputeNewCurvaturePointProgram->BeginSetShaderParameters();
        mComputeNewCurvaturePointProgram->SetShaderParameter("levelSetSampler", mNewPhysicalMemoryTexture[i]);
        mComputeNewCurvaturePointProgram->SetShaderParameter("modelProjectionMatrix", projectionMatrix);
        mComputeNewCurvaturePointProgram->EndSetShaderParameters();
#endif

        mComputeNewCurvaturePointProgram->Bind();
        DrawTopLeftPoints(substreamInfo[i]);
        mComputeNewCurvaturePointProgram->Unbind();

        
#ifdef LEFOHN_BENCHMARK
        mComputeNewCurvaturePointProgram->BeginSetShaderParameters();
        mComputeNewCurvaturePointProgram->SetShaderParameter("levelSetSampler", mNewPhysicalMemoryTexture[i]);
        mComputeNewCurvaturePointProgram->SetShaderParameter("modelProjectionMatrix", projectionMatrix);
        mComputeNewCurvaturePointProgram->EndSetShaderParameters();
#endif

        mComputeNewCurvaturePointProgram->Bind();
        DrawTopRightPoints(substreamInfo[i]);
        mComputeNewCurvaturePointProgram->Unbind();

        
#ifdef LEFOHN_BENCHMARK
        mComputeNewCurvaturePointProgram->BeginSetShaderParameters();
        mComputeNewCurvaturePointProgram->SetShaderParameter("levelSetSampler", mNewPhysicalMemoryTexture[i]);
        mComputeNewCurvaturePointProgram->SetShaderParameter("modelProjectionMatrix", projectionMatrix);
        mComputeNewCurvaturePointProgram->EndSetShaderParameters();
#endif

        mComputeNewCurvaturePointProgram->Bind();
        DrawBottomLeftPoints(substreamInfo[i]);
        mComputeNewCurvaturePointProgram->Unbind();

#ifdef LEFOHN_BENCHMARK
        mComputeNewCurvaturePointProgram->BeginSetShaderParameters();
        mComputeNewCurvaturePointProgram->SetShaderParameter("levelSetSampler", mNewPhysicalMemoryTexture[i]);
        mComputeNewCurvaturePointProgram->SetShaderParameter("modelProjectionMatrix", projectionMatrix);
        mComputeNewCurvaturePointProgram->EndSetShaderParameters();
#endif

        mComputeNewCurvaturePointProgram->Bind();
        DrawBottomRightPoints(substreamInfo[i]);
        mComputeNewCurvaturePointProgram->Unbind();
        
        mFrameBuffer->Unbind();
    }
}

void SegmentationSimulator::ComputeNewMemoryAllocations(const math::Matrix44& projectionMatrix,
                                                SubstreamInformation** substreamInfo )
{
    for( int i = 0; i < mNumTexturesUsed; ++i )
    {
        mFrameBuffer->Bind( mMemoryAllocationRequestOneTexture[i], mMemoryAllocationRequestTwoTexture[i] );

        rendering::rtgi::ClearColorBuffer( rendering::rtgi::ColorRGBA( 0.0f, 0.00f, 0.00f, 0.0f ) );

#ifdef LEFOHN_BENCHMARK
        mComputeNewMemoryAllocationProgram->BeginSetShaderParameters();
        mComputeNewMemoryAllocationProgram->SetShaderParameter("gradientSampler", mGradientTexture[i]);
        mComputeNewMemoryAllocationProgram->SetShaderParameter("modelProjectionMatrix", projectionMatrix);
        mComputeNewMemoryAllocationProgram->EndSetShaderParameters();
#endif

        mComputeNewMemoryAllocationProgram->Bind();
        DrawMiddleQuads(substreamInfo[i]);
        mComputeNewMemoryAllocationProgram->Unbind();

#ifdef LEFOHN_BENCHMARK
        mComputeNewMemoryAllocationProgram->BeginSetShaderParameters();
        mComputeNewMemoryAllocationProgram->SetShaderParameter("gradientSampler", mGradientTexture[i]);
        mComputeNewMemoryAllocationProgram->SetShaderParameter("modelProjectionMatrix", projectionMatrix);
        mComputeNewMemoryAllocationProgram->EndSetShaderParameters();
#endif

        mComputeNewMemoryAllocationProgram->Bind();
        DrawTopLines(substreamInfo[i]);
        mComputeNewMemoryAllocationProgram->Unbind();

#ifdef LEFOHN_BENCHMARK
        mComputeNewMemoryAllocationProgram->BeginSetShaderParameters();
        mComputeNewMemoryAllocationProgram->SetShaderParameter("gradientSampler", mGradientTexture[i]);
        mComputeNewMemoryAllocationProgram->SetShaderParameter("modelProjectionMatrix", projectionMatrix);
        mComputeNewMemoryAllocationProgram->EndSetShaderParameters();
#endif

        mComputeNewMemoryAllocationProgram->Bind();
        DrawLeftLines(substreamInfo[i]);
        mComputeNewMemoryAllocationProgram->Unbind();

#ifdef LEFOHN_BENCHMARK
        mComputeNewMemoryAllocationProgram->BeginSetShaderParameters();
        mComputeNewMemoryAllocationProgram->SetShaderParameter("gradientSampler", mGradientTexture[i]);
        mComputeNewMemoryAllocationProgram->SetShaderParameter("modelProjectionMatrix", projectionMatrix);
        mComputeNewMemoryAllocationProgram->EndSetShaderParameters();
#endif

        mComputeNewMemoryAllocationProgram->Bind();
        DrawBottomLines(substreamInfo[i]);
        mComputeNewMemoryAllocationProgram->Unbind();


#ifdef LEFOHN_BENCHMARK
        mComputeNewMemoryAllocationProgram->BeginSetShaderParameters();
        mComputeNewMemoryAllocationProgram->SetShaderParameter("gradientSampler", mGradientTexture[i]);
        mComputeNewMemoryAllocationProgram->SetShaderParameter("modelProjectionMatrix", projectionMatrix);
        mComputeNewMemoryAllocationProgram->EndSetShaderParameters();
#endif

        mComputeNewMemoryAllocationProgram->Bind();
        DrawRightLines(substreamInfo[i]);
        mComputeNewMemoryAllocationProgram->Unbind();

#ifdef LEFOHN_BENCHMARK
        mComputeNewMemoryAllocationProgram->BeginSetShaderParameters();
        mComputeNewMemoryAllocationProgram->SetShaderParameter("gradientSampler", mGradientTexture[i]);
        mComputeNewMemoryAllocationProgram->SetShaderParameter("modelProjectionMatrix", projectionMatrix);
        mComputeNewMemoryAllocationProgram->EndSetShaderParameters();
#endif

        mComputeNewMemoryAllocationProgram->Bind();
        DrawTopLeftPoints(substreamInfo[i]);
        mComputeNewMemoryAllocationProgram->Unbind();

#ifdef LEFOHN_BENCHMARK
        mComputeNewMemoryAllocationProgram->BeginSetShaderParameters();
        mComputeNewMemoryAllocationProgram->SetShaderParameter("gradientSampler", mGradientTexture[i]);
        mComputeNewMemoryAllocationProgram->SetShaderParameter("modelProjectionMatrix", projectionMatrix);
        mComputeNewMemoryAllocationProgram->EndSetShaderParameters();
#endif

        mComputeNewMemoryAllocationProgram->Bind();
        DrawTopRightPoints(substreamInfo[i]);
        mComputeNewMemoryAllocationProgram->Unbind();

#ifdef LEFOHN_BENCHMARK
        mComputeNewMemoryAllocationProgram->BeginSetShaderParameters();
        mComputeNewMemoryAllocationProgram->SetShaderParameter("gradientSampler", mGradientTexture[i]);
        mComputeNewMemoryAllocationProgram->SetShaderParameter("modelProjectionMatrix", projectionMatrix);
        mComputeNewMemoryAllocationProgram->EndSetShaderParameters();
#endif

        mComputeNewMemoryAllocationProgram->Bind();
        DrawBottomLeftPoints(substreamInfo[i]);
        mComputeNewMemoryAllocationProgram->Unbind();

#ifdef LEFOHN_BENCHMARK
        mComputeNewMemoryAllocationProgram->BeginSetShaderParameters();
        mComputeNewMemoryAllocationProgram->SetShaderParameter("gradientSampler", mGradientTexture[i]);
        mComputeNewMemoryAllocationProgram->SetShaderParameter("modelProjectionMatrix", projectionMatrix);
        mComputeNewMemoryAllocationProgram->EndSetShaderParameters();
#endif

        mComputeNewMemoryAllocationProgram->Bind();
        DrawBottomRightPoints(substreamInfo[i]);
        mComputeNewMemoryAllocationProgram->Unbind();

        mFrameBuffer->Unbind();
    }
}

std::vector<short>** SegmentationSimulator::Build16x16Quads()
{
    int numTiles = mInversePageTable->CalculateNumberPhysicalTilesStored();
    int totalTilesWeCanStore = mNumTexturesUsed * SegmentationSimulator::NUM_TILES_PER_ROW_COLUMN_SQUARED;
    int leftOverTiles = numTiles % SegmentationSimulator::NUM_TILES_PER_ROW_COLUMN_SQUARED;

    std::vector<short>** vertices = new std::vector<short>*[mNumTexturesUsed];

    PhysicalCoordinate currentTileCoordinate(0,0);

    for( int i = 0; i < mNumTexturesUsed; ++i )
    {
        int currentNumTiles = i == mNumTexturesUsed - 1 ? leftOverTiles : SegmentationSimulator::NUM_TILES_PER_ROW_COLUMN_SQUARED;

        vertices[i] = new std::vector<short>();
        vertices[i]->reserve(currentNumTiles * 4 * 2); // 4 vertices per tile, 2 elements per vertex.

        // Create geometry for two static tiles
        for(int k = 0; k < 2; ++k)
        {
            vertices[i]->push_back(0 + k * SegmentationSimulator::TILE_SIZE);
            vertices[i]->push_back(0);

            vertices[i]->push_back(0);
            vertices[i]->push_back(SegmentationSimulator::TILE_SIZE);

            vertices[i]->push_back(SegmentationSimulator::TILE_SIZE + k * SegmentationSimulator::TILE_SIZE);
            vertices[i]->push_back(SegmentationSimulator::TILE_SIZE);

            vertices[i]->push_back(SegmentationSimulator::TILE_SIZE + k * SegmentationSimulator::TILE_SIZE);
            vertices[i]->push_back(0);
        }

        // Skip the two static tiles.
        int startX = SegmentationSimulator::TILE_SIZE * 2;
        int currentTileX = 0;
        int currentTileY = 0;
        int curTileNum = 2;
        int textureOffsetY = i * SegmentationSimulator::NUM_TILES_PER_ROW_COLUMN;

        PhysicalTile* currentTile = mInversePageTable->GetPhysicalTile(currentTileCoordinate);

        int numTilesToRender = currentNumTiles >= SegmentationSimulator::NUM_TILES_PER_ROW_COLUMN_SQUARED - 1 ? SegmentationSimulator::NUM_TILES_PER_ROW_COLUMN_SQUARED - 2 : currentNumTiles;

        for( int j = 0; j < numTilesToRender; ++j )
        {    
            currentTileX = startX + (currentTileCoordinate[math::X] * SegmentationSimulator::TILE_SIZE);
            currentTileY = (currentTileCoordinate[math::Y] - textureOffsetY) * SegmentationSimulator::TILE_SIZE;

            vertices[i]->push_back(currentTileX);
            vertices[i]->push_back(currentTileY);

            vertices[i]->push_back(currentTileX);
            vertices[i]->push_back(currentTileY + SegmentationSimulator::TILE_SIZE);

            vertices[i]->push_back(currentTileX + SegmentationSimulator::TILE_SIZE);
            vertices[i]->push_back(currentTileY + SegmentationSimulator::TILE_SIZE);

            vertices[i]->push_back(currentTileX + SegmentationSimulator::TILE_SIZE);
            vertices[i]->push_back(currentTileY);

            currentTileCoordinate.Increment( SegmentationSimulator::NUM_TILES_PER_ROW_COLUMN );
            currentTile = mInversePageTable->GetPhysicalTile(currentTileCoordinate);

            if(currentTileX == (SegmentationSimulator::NUM_TILES_PER_ROW_COLUMN-1) * SegmentationSimulator::TILE_SIZE)
                startX = 0;

            if( currentTileCoordinate[math::Y] >= ( SegmentationSimulator::NUM_TILES_PER_ROW_COLUMN * (i + 1) ) )
                break;
        }
    }

    return vertices;
}

SegmentationSimulator::SubstreamInformation** SegmentationSimulator::BuildSubstreamInformation(bool sendNeighborTextureCoordinates /*= true*/)
{
    int numTiles = mInversePageTable->CalculateNumberPhysicalTilesStored();
    int totalTilesWeCanStore = mNumTexturesUsed * SegmentationSimulator::NUM_TILES_PER_ROW_COLUMN_SQUARED;
    int leftOverTiles = numTiles % SegmentationSimulator::NUM_TILES_PER_ROW_COLUMN_SQUARED;

    SubstreamInformation** substreamInfo = new SubstreamInformation*[mNumTexturesUsed];

    PhysicalCoordinate currentTileCoordinate(0,0);

    for( int i = 0; i < mNumTexturesUsed; ++i )
    {
        substreamInfo[i] = new SubstreamInformation[9];
        int currentNumTiles = i == mNumTexturesUsed - 1 && leftOverTiles != 0 ? leftOverTiles : SegmentationSimulator::NUM_TILES_PER_ROW_COLUMN_SQUARED;
        if( i == 0 && currentNumTiles == SegmentationSimulator::NUM_TILES_PER_ROW_COLUMN_SQUARED )
            currentNumTiles = SegmentationSimulator::NUM_TILES_PER_ROW_COLUMN_SQUARED - 2;

        // Skip static tiles completely.
        int currentTileX = 0;
        int currentTileY = 0;
        int startX = i == 0 ? 2 * SegmentationSimulator::TILE_SIZE : 0;

        PhysicalTile* currentTile = mInversePageTable->GetPhysicalTile(currentTileCoordinate);
        
        int numQuadVertices = currentNumTiles * 4;
        int numLineVertices = currentNumTiles * 2;
        int numPointVertices = currentNumTiles;

        ReserveMemoryForSubstreams(substreamInfo[i], numQuadVertices, numLineVertices, numPointVertices,
                                    sendNeighborTextureCoordinates);

        int textureOffsetY = i * SegmentationSimulator::NUM_TILES_PER_ROW_COLUMN;

        for( int j = 0; j < currentNumTiles; ++j )
        {
            // Somehow we went too far!
            if( currentTile == NULL )
                break;    

            currentTileX = startX + (currentTileCoordinate[math::X] * SegmentationSimulator::TILE_SIZE);

            BuildMiddleQuads( substreamInfo[i], currentTile, startX, textureOffsetY, sendNeighborTextureCoordinates );
            BuildTopLines( substreamInfo[i], currentTile, startX, textureOffsetY, sendNeighborTextureCoordinates );
            BuildLeftLines( substreamInfo[i], currentTile, startX, textureOffsetY, sendNeighborTextureCoordinates );
            BuildBottomLines( substreamInfo[i], currentTile, startX, textureOffsetY, sendNeighborTextureCoordinates );
            BuildRightLines( substreamInfo[i], currentTile, startX, textureOffsetY, sendNeighborTextureCoordinates );
            BuildTopLeftPoints( substreamInfo[i], currentTile, startX, textureOffsetY, sendNeighborTextureCoordinates );
            BuildTopRightPoints( substreamInfo[i], currentTile, startX, textureOffsetY, sendNeighborTextureCoordinates );
            BuildBottomLeftPoints( substreamInfo[i], currentTile, startX, textureOffsetY, sendNeighborTextureCoordinates );
            BuildBottomRightPoints( substreamInfo[i], currentTile, startX, textureOffsetY, sendNeighborTextureCoordinates );

            currentTileCoordinate.Increment( SegmentationSimulator::NUM_TILES_PER_ROW_COLUMN );
            currentTile = mInversePageTable->GetPhysicalTile(currentTileCoordinate);

            if(currentTileX >= (SegmentationSimulator::NUM_TILES_PER_ROW_COLUMN - 1) * SegmentationSimulator::TILE_SIZE)
                startX = 0;

            if( currentTileCoordinate[math::Y] >= SegmentationSimulator::NUM_TILES_PER_ROW_COLUMN * (i+1) )
                break;
        }
    }

    return substreamInfo;
}

void SegmentationSimulator::BuildMiddleQuads( SubstreamInformation* substreamInfo, PhysicalTile* currentTile,
                                             int startOffset, int textureOffsetY,  bool sendNeighborTextureCoordinates /*= true*/)
{
    const int quadSize = SegmentationSimulator::TILE_SIZE - 2;
    const int quadOffset = 1;

    PhysicalCoordinate currentTileCoordinate = currentTile->GetPhysicalAddress();

    const int currentTileX = startOffset + (currentTileCoordinate[math::X] * SegmentationSimulator::TILE_SIZE) + quadOffset;
    const int currentTileY = ( currentTileCoordinate[math::Y] - textureOffsetY ) * SegmentationSimulator::TILE_SIZE + quadOffset;

    PhysicalCoordinate* neighbors = NULL;

    // Build the vertex/texture coordinate arrays.
    substreamInfo[MiddleQuads].vertices.push_back(currentTileX);
    substreamInfo[MiddleQuads].vertices.push_back(currentTileY + quadSize);
    substreamInfo[MiddleQuads].vertices.push_back(currentTileX + quadSize);
    substreamInfo[MiddleQuads].vertices.push_back(currentTileY + quadSize);
    substreamInfo[MiddleQuads].vertices.push_back(currentTileX + quadSize);
    substreamInfo[MiddleQuads].vertices.push_back(currentTileY);
    substreamInfo[MiddleQuads].vertices.push_back(currentTileX);
    substreamInfo[MiddleQuads].vertices.push_back(currentTileY);

    // Don't create texture coordinates if we don't need them.
    if(sendNeighborTextureCoordinates)
    {
        neighbors = GetQuadNeighborTileCoordinates(currentTile);

        for( int i = 0; i < 2; ++i )
        {
            int neighborX = (neighbors[i][math::X] * SegmentationSimulator::TILE_SIZE) + quadOffset;
            int neighborY = ( ( neighbors[i][math::Y] - textureOffsetY )* SegmentationSimulator::TILE_SIZE) + quadOffset;

            substreamInfo[MiddleQuads].textureCoordinates[i].push_back(neighborX);
            substreamInfo[MiddleQuads].textureCoordinates[i].push_back(neighborY + quadSize);
            substreamInfo[MiddleQuads].textureCoordinates[i].push_back(neighborX + quadSize);
            substreamInfo[MiddleQuads].textureCoordinates[i].push_back(neighborY + quadSize);
            substreamInfo[MiddleQuads].textureCoordinates[i].push_back(neighborX + quadSize);
            substreamInfo[MiddleQuads].textureCoordinates[i].push_back(neighborY);
            substreamInfo[MiddleQuads].textureCoordinates[i].push_back(neighborX);
            substreamInfo[MiddleQuads].textureCoordinates[i].push_back(neighborY);
        }

        delete [] neighbors;
    }
}

void SegmentationSimulator::BuildTopLines( SubstreamInformation* substreamInfo, PhysicalTile* currentTile,
                                          int startOffset, int textureOffsetY, bool sendNeighborTextureCoordinates /* = true */ )
{
    const int lineSize = SegmentationSimulator::TILE_SIZE - 2;
    const int lineOffset = 1;

    PhysicalCoordinate currentTileCoordinate = currentTile->GetPhysicalAddress();

    const int currentTileX = startOffset + (currentTileCoordinate[math::X] * SegmentationSimulator::TILE_SIZE) + lineOffset;
    const int currentTileY = ( currentTileCoordinate[math::Y] - textureOffsetY ) * SegmentationSimulator::TILE_SIZE;

    PhysicalCoordinate* neighbors = NULL;

    // Build the vertex/texture coordinate arrays.
    substreamInfo[TopLines].vertices.push_back(currentTileX);
    substreamInfo[TopLines].vertices.push_back(currentTileY);
    substreamInfo[TopLines].vertices.push_back(currentTileX + lineSize);
    substreamInfo[TopLines].vertices.push_back(currentTileY);

    // Don't create texture coordinates if we don't need them.
    if(sendNeighborTextureCoordinates)
    {
        neighbors = GetLineNeighborTileCoordinates(currentTile, Direction_Up);

        for( int i = 0; i < 5; ++i )
        {
            int neighborX = (neighbors[i][math::X] * SegmentationSimulator::TILE_SIZE) + lineOffset;
            int neighborY = ( ( neighbors[i][math::Y] - textureOffsetY ) * SegmentationSimulator::TILE_SIZE);

            substreamInfo[TopLines].textureCoordinates[i].push_back(neighborX);
            substreamInfo[TopLines].textureCoordinates[i].push_back(neighborY);
            substreamInfo[TopLines].textureCoordinates[i].push_back(neighborX + lineSize);
            substreamInfo[TopLines].textureCoordinates[i].push_back(neighborY);
        }

        delete [] neighbors;
    }
}

void SegmentationSimulator::BuildLeftLines( SubstreamInformation* substreamInfo, PhysicalTile* currentTile,
                                           int startOffset, int textureOffsetY, bool sendNeighborTextureCoordinates /* = true */ )
{
    const int lineSize = SegmentationSimulator::TILE_SIZE - 2;
    const int lineOffset = 1;

    PhysicalCoordinate currentTileCoordinate = currentTile->GetPhysicalAddress();

    const int currentTileX = startOffset + (currentTileCoordinate[math::X] * SegmentationSimulator::TILE_SIZE) + lineOffset;
    const int currentTileY = ( currentTileCoordinate[math::Y] - textureOffsetY ) * SegmentationSimulator::TILE_SIZE + lineOffset;

    PhysicalCoordinate* neighbors = NULL;

    // Build the vertex/texture coordinate arrays.
    substreamInfo[LeftLines].vertices.push_back(currentTileX);
    substreamInfo[LeftLines].vertices.push_back(currentTileY);
    substreamInfo[LeftLines].vertices.push_back(currentTileX);
    substreamInfo[LeftLines].vertices.push_back(currentTileY + lineSize);

    // Don't create texture coordinates if we don't need them.
    if(sendNeighborTextureCoordinates)
    {
        neighbors = GetLineNeighborTileCoordinates(currentTile, Direction_Left);

        for( int i = 0; i < 5; ++i )
        {
            int neighborX = (neighbors[i][math::X] * SegmentationSimulator::TILE_SIZE) + lineOffset;
            int neighborY = ( ( neighbors[i][math::Y] - textureOffsetY ) * SegmentationSimulator::TILE_SIZE) + lineOffset;

            substreamInfo[LeftLines].textureCoordinates[i].push_back(neighborX);
            substreamInfo[LeftLines].textureCoordinates[i].push_back(neighborY);
            substreamInfo[LeftLines].textureCoordinates[i].push_back(neighborX);
            substreamInfo[LeftLines].textureCoordinates[i].push_back(neighborY + lineSize);
        }

        delete [] neighbors;
    }
}

void SegmentationSimulator::BuildBottomLines( SubstreamInformation* substreamInfo, PhysicalTile* currentTile,
                                             int startOffset,int textureOffsetY,  bool sendNeighborTextureCoordinates /* = true */ )
{
    const int lineSize = SegmentationSimulator::TILE_SIZE - 2;
    const int lineOffset = 1;
    const int yOffset = SegmentationSimulator::TILE_SIZE - 1;

    PhysicalCoordinate currentTileCoordinate = currentTile->GetPhysicalAddress();

    const int currentTileX = startOffset + (currentTileCoordinate[math::X] * SegmentationSimulator::TILE_SIZE) + lineOffset;
    const int currentTileY = ( currentTileCoordinate[math::Y] - textureOffsetY ) * SegmentationSimulator::TILE_SIZE + yOffset;

    PhysicalCoordinate* neighbors = NULL;

    // Build the vertex/texture coordinate arrays.
    substreamInfo[BottomLines].vertices.push_back(currentTileX);
    substreamInfo[BottomLines].vertices.push_back(currentTileY);
    substreamInfo[BottomLines].vertices.push_back(currentTileX + lineSize);
    substreamInfo[BottomLines].vertices.push_back(currentTileY);

    // Don't create texture coordinates if we don't need them.
    if(sendNeighborTextureCoordinates)
    {
        neighbors = GetLineNeighborTileCoordinates(currentTile, Direction_Down);

        for( int i = 0; i < 5; ++i )
        {
            int neighborX = (neighbors[i][math::X] * SegmentationSimulator::TILE_SIZE) + lineOffset;
            int neighborY = ( ( neighbors[i][math::Y] - textureOffsetY ) * SegmentationSimulator::TILE_SIZE) + yOffset;

            substreamInfo[BottomLines].textureCoordinates[i].push_back(neighborX);
            substreamInfo[BottomLines].textureCoordinates[i].push_back(neighborY);
            substreamInfo[BottomLines].textureCoordinates[i].push_back(neighborX + lineSize);
            substreamInfo[BottomLines].textureCoordinates[i].push_back(neighborY);
        }

        delete [] neighbors;
    }
}

void SegmentationSimulator::BuildRightLines( SubstreamInformation* substreamInfo, PhysicalTile* currentTile,
                                            int startOffset, int textureOffsetY, bool sendNeighborTextureCoordinates /* = true */ )
{
    const int lineSize = SegmentationSimulator::TILE_SIZE - 2;
    const int lineOffset = 1;
    const int xOffset = 15;

    PhysicalCoordinate currentTileCoordinate = currentTile->GetPhysicalAddress();

    const int currentTileX = startOffset + (currentTileCoordinate[math::X] * SegmentationSimulator::TILE_SIZE) + lineOffset;
    const int currentTileY = ( currentTileCoordinate[math::Y] - textureOffsetY ) * SegmentationSimulator::TILE_SIZE + lineOffset;

    PhysicalCoordinate* neighbors = NULL;

    // Build the vertex/texture coordinate arrays.
    substreamInfo[RightLines].vertices.push_back(currentTileX + xOffset);
    substreamInfo[RightLines].vertices.push_back(currentTileY);
    substreamInfo[RightLines].vertices.push_back(currentTileX + xOffset);
    substreamInfo[RightLines].vertices.push_back(currentTileY + lineSize);

    // Don't create texture coordinates if we don't need them.
    if(sendNeighborTextureCoordinates)
    {
        neighbors = GetLineNeighborTileCoordinates(currentTile, Direction_Right);

        for( int i = 0; i < 5; ++i )
        {
            int neighborX = (neighbors[i][math::X] * SegmentationSimulator::TILE_SIZE) + lineOffset;
            int neighborY = ( ( neighbors[i][math::Y] - textureOffsetY ) * SegmentationSimulator::TILE_SIZE) + lineOffset;

            substreamInfo[RightLines].textureCoordinates[i].push_back(neighborX + xOffset);
            substreamInfo[RightLines].textureCoordinates[i].push_back(neighborY);
            substreamInfo[RightLines].textureCoordinates[i].push_back(neighborX + xOffset);
            substreamInfo[RightLines].textureCoordinates[i].push_back(neighborY + lineSize);
        }

        delete [] neighbors;
    }
}

void SegmentationSimulator::BuildTopLeftPoints( SubstreamInformation* substreamInfo, PhysicalTile* currentTile,
                                               int startOffset, int textureOffsetY, bool sendNeighborTextureCoordinates /* = true */ )
{
    const int yOffset = SegmentationSimulator::TILE_SIZE - 1;

    PhysicalCoordinate currentTileCoordinate = currentTile->GetPhysicalAddress();

    const int currentTileX = startOffset + (currentTileCoordinate[math::X] * SegmentationSimulator::TILE_SIZE);
    const int currentTileY = ( currentTileCoordinate[math::Y] - textureOffsetY ) * SegmentationSimulator::TILE_SIZE + 1;

    PhysicalCoordinate* neighbors = NULL;

    // Build the vertex/texture coordinate arrays.
    substreamInfo[TopLeftPoints].vertices.push_back(currentTileX);
    substreamInfo[TopLeftPoints].vertices.push_back(currentTileY);

    // Don't create texture coordinates if we don't need them.
    if(sendNeighborTextureCoordinates)
    {
        neighbors = GetPointNeighborTileCoordinates(currentTile, Direction_Up | Direction_Left);
        // For each texture unit (6) we have a pair of coords
        int j = 0;
        for( int i = 0; i < 6; i++ )
        {
            if( i < 5)
            {
                int neighbor1X = (neighbors[j][math::X] * SegmentationSimulator::TILE_SIZE);
                int neighbor1Y = ( ( neighbors[j][math::Y] - textureOffsetY ) * SegmentationSimulator::TILE_SIZE) + 1;
                int neighbor2X = (neighbors[j+1][math::X] * SegmentationSimulator::TILE_SIZE);
                int neighbor2Y = (( neighbors[j+1][math::Y] - textureOffsetY ) * SegmentationSimulator::TILE_SIZE) + 1;

                j += 2;

                substreamInfo[TopLeftPoints].textureCoordinates[i].push_back(neighbor1X);
                substreamInfo[TopLeftPoints].textureCoordinates[i].push_back(neighbor1Y);
                substreamInfo[TopLeftPoints].textureCoordinates[i].push_back(neighbor2X);
                substreamInfo[TopLeftPoints].textureCoordinates[i].push_back(neighbor2Y);
            }
            else
            {
                // last neighbor (11 neighbors)
                int neighbor1X = (neighbors[10][math::X] * SegmentationSimulator::TILE_SIZE);
                int neighbor1Y = ( ( neighbors[10][math::Y] - textureOffsetY ) * SegmentationSimulator::TILE_SIZE) + 1;
                substreamInfo[TopLeftPoints].textureCoordinates[i].push_back(neighbor1X);
                substreamInfo[TopLeftPoints].textureCoordinates[i].push_back(neighbor1Y);
                substreamInfo[TopLeftPoints].textureCoordinates[i].push_back(0);
                substreamInfo[TopLeftPoints].textureCoordinates[i].push_back(0);
            }
        }

        delete [] neighbors;
    }
}

void SegmentationSimulator::BuildTopRightPoints( SubstreamInformation* substreamInfo, PhysicalTile* currentTile,
                                                int startOffset, int textureOffsetY, bool sendNeighborTextureCoordinates /* = true */ )
{
    const int xOffset = SegmentationSimulator::TILE_SIZE - 1;

    PhysicalCoordinate currentTileCoordinate = currentTile->GetPhysicalAddress();

    const int currentTileX = startOffset + (currentTileCoordinate[math::X] * SegmentationSimulator::TILE_SIZE) + xOffset;
    const int currentTileY = ( currentTileCoordinate[math::Y] - textureOffsetY ) * SegmentationSimulator::TILE_SIZE + 1;

    PhysicalCoordinate* neighbors = NULL;

    // Build the vertex/texture coordinate arrays.
    substreamInfo[TopRightPoints].vertices.push_back(currentTileX);
    substreamInfo[TopRightPoints].vertices.push_back(currentTileY);

    // Don't create texture coordinates if we don't need them.
    if(sendNeighborTextureCoordinates)
    {
        neighbors = GetPointNeighborTileCoordinates(currentTile, Direction_Up | Direction_Right);

        // For each texture unit (6) we have a pair of coords
        int j = 0;
        for( int i = 0; i < 6; i++ )
        {
            if( i < 5)
            {
                int neighbor1X = (neighbors[j][math::X] * SegmentationSimulator::TILE_SIZE) + xOffset;
                int neighbor1Y = ( ( neighbors[j][math::Y] - textureOffsetY ) * SegmentationSimulator::TILE_SIZE) + 1;
                int neighbor2X = (neighbors[j+1][math::X] * SegmentationSimulator::TILE_SIZE) + xOffset;
                int neighbor2Y = ( ( neighbors[j+1][math::Y] - textureOffsetY ) * SegmentationSimulator::TILE_SIZE) + 1;

                j += 2;

                substreamInfo[TopRightPoints].textureCoordinates[i].push_back(neighbor1X);
                substreamInfo[TopRightPoints].textureCoordinates[i].push_back(neighbor1Y);
                substreamInfo[TopRightPoints].textureCoordinates[i].push_back(neighbor2X);
                substreamInfo[TopRightPoints].textureCoordinates[i].push_back(neighbor2Y);
            }
            else
            {
                // last neighbor (11 neighbors)
                int neighbor1X = (neighbors[10][math::X] * SegmentationSimulator::TILE_SIZE) + xOffset;
                int neighbor1Y = ( ( neighbors[10][math::Y] - textureOffsetY ) * SegmentationSimulator::TILE_SIZE) + 1;
                substreamInfo[TopRightPoints].textureCoordinates[i].push_back(neighbor1X);
                substreamInfo[TopRightPoints].textureCoordinates[i].push_back(neighbor1Y);
                substreamInfo[TopRightPoints].textureCoordinates[i].push_back(0);
                substreamInfo[TopRightPoints].textureCoordinates[i].push_back(0);
            }
        }

        delete [] neighbors;
    }
}

void SegmentationSimulator::BuildBottomLeftPoints( SubstreamInformation* substreamInfo, PhysicalTile* currentTile,
                                                  int startOffset, int textureOffsetY, bool sendNeighborTextureCoordinates /* = true */ )
{
    const int yOffset = SegmentationSimulator::TILE_SIZE;

    PhysicalCoordinate currentTileCoordinate = currentTile->GetPhysicalAddress();

    const int currentTileX = startOffset + (currentTileCoordinate[math::X] * SegmentationSimulator::TILE_SIZE);
    const int currentTileY = ( currentTileCoordinate[math::Y] - textureOffsetY ) * SegmentationSimulator::TILE_SIZE + yOffset;

    PhysicalCoordinate* neighbors = NULL;

    // Build the vertex/texture coordinate arrays.
    substreamInfo[BottomLeftPoints].vertices.push_back(currentTileX);
    substreamInfo[BottomLeftPoints].vertices.push_back(currentTileY);

    // Don't create texture coordinates if we don't need them.
    if(sendNeighborTextureCoordinates)
    {
        neighbors = GetPointNeighborTileCoordinates(currentTile, Direction_Down | Direction_Left);

        // For each texture unit (6) we have a pair of coords
        int j = 0;
        for( int i = 0; i < 6; i++ )
        {
            if( i < 5)
            {
                int neighbor1X = (neighbors[j][math::X] * SegmentationSimulator::TILE_SIZE);
                int neighbor1Y = ( ( neighbors[j][math::Y] - textureOffsetY ) * SegmentationSimulator::TILE_SIZE) + yOffset;
                int neighbor2X = (neighbors[j+1][math::X] * SegmentationSimulator::TILE_SIZE);
                int neighbor2Y = ( ( neighbors[j+1][math::Y] - textureOffsetY ) * SegmentationSimulator::TILE_SIZE) + yOffset;

                j += 2;

                substreamInfo[BottomLeftPoints].textureCoordinates[i].push_back(neighbor1X);
                substreamInfo[BottomLeftPoints].textureCoordinates[i].push_back(neighbor1Y);
                substreamInfo[BottomLeftPoints].textureCoordinates[i].push_back(neighbor2X);
                substreamInfo[BottomLeftPoints].textureCoordinates[i].push_back(neighbor2Y);
            }
            else
            {
                // last neighbor (11 neighbors)
                int neighbor1X = (neighbors[10][math::X] * SegmentationSimulator::TILE_SIZE);
                int neighbor1Y = ( ( neighbors[10][math::Y] - textureOffsetY ) * SegmentationSimulator::TILE_SIZE) + yOffset;
                substreamInfo[BottomLeftPoints].textureCoordinates[i].push_back(neighbor1X);
                substreamInfo[BottomLeftPoints].textureCoordinates[i].push_back(neighbor1Y);
                substreamInfo[BottomLeftPoints].textureCoordinates[i].push_back(0);
                substreamInfo[BottomLeftPoints].textureCoordinates[i].push_back(0);
            }
        }

        delete [] neighbors;
    }
}

void SegmentationSimulator::BuildBottomRightPoints( SubstreamInformation* substreamInfo, PhysicalTile* currentTile,
                                                   int startOffset, int textureOffsetY, bool sendNeighborTextureCoordinates /* = true */ )
{
    const int xOffset = SegmentationSimulator::TILE_SIZE - 1;
    const int yOffset = SegmentationSimulator::TILE_SIZE;

    PhysicalCoordinate currentTileCoordinate = currentTile->GetPhysicalAddress();

    const int currentTileX = startOffset + (currentTileCoordinate[math::X] * SegmentationSimulator::TILE_SIZE) + xOffset;
    const int currentTileY = ( currentTileCoordinate[math::Y] - textureOffsetY ) * SegmentationSimulator::TILE_SIZE + yOffset;

    PhysicalCoordinate* neighbors = NULL;

    // Build the vertex/texture coordinate arrays.
    substreamInfo[BottomRightPoints].vertices.push_back(currentTileX);
    substreamInfo[BottomRightPoints].vertices.push_back(currentTileY);

    // Don't create texture coordinates if we don't need them.
    if(sendNeighborTextureCoordinates)
    {
        neighbors = GetPointNeighborTileCoordinates(currentTile, Direction_Down | Direction_Right);

        // For each texture unit (6) we have a pair of coords
        int j = 0;
        for( int i = 0; i < 6; i++ )
        {
            if( i < 5)
            {
                int neighbor1X = (neighbors[j][math::X] * SegmentationSimulator::TILE_SIZE) + xOffset;
                int neighbor1Y = ( ( neighbors[j][math::Y] - textureOffsetY ) * SegmentationSimulator::TILE_SIZE) + yOffset;
                int neighbor2X = (neighbors[j+1][math::X] * SegmentationSimulator::TILE_SIZE) + xOffset;
                int neighbor2Y = ( ( neighbors[j+1][math::Y] - textureOffsetY ) * SegmentationSimulator::TILE_SIZE) + yOffset;

                j += 2;

                substreamInfo[BottomRightPoints].textureCoordinates[i].push_back(neighbor1X);
                substreamInfo[BottomRightPoints].textureCoordinates[i].push_back(neighbor1Y);
                substreamInfo[BottomRightPoints].textureCoordinates[i].push_back(neighbor2X);
                substreamInfo[BottomRightPoints].textureCoordinates[i].push_back(neighbor2Y);
            }
            else
            {
                // last neighbor (11 neighbors)
                int neighbor1X = (neighbors[10][math::X] * SegmentationSimulator::TILE_SIZE) + xOffset;
                int neighbor1Y = ( ( neighbors[10][math::Y] - textureOffsetY ) * SegmentationSimulator::TILE_SIZE) + yOffset;
                substreamInfo[BottomRightPoints].textureCoordinates[i].push_back(neighbor1X);
                substreamInfo[BottomRightPoints].textureCoordinates[i].push_back(neighbor1Y);
                substreamInfo[BottomRightPoints].textureCoordinates[i].push_back(0);
                substreamInfo[BottomRightPoints].textureCoordinates[i].push_back(0);
            }
        }

        delete [] neighbors;
    }
}

void SegmentationSimulator::ReserveMemoryForSubstreams(SubstreamInformation* substreamInfo,
                                                       int numQuadVertices, int numLineVertices,
                                                       int numPointVertices, bool sendNeighborTextureCoordinates)
{
    // Quad first.
    substreamInfo[MiddleQuads].vertices.reserve(numQuadVertices * 2);
    // Don't reserve memory if we aren't sending the texture coordinates.
    if(sendNeighborTextureCoordinates)
    {
        substreamInfo[MiddleQuads].textureCoordinates.push_back( std::vector<short>() );
        substreamInfo[MiddleQuads].textureCoordinates.push_back( std::vector<short>() );
        substreamInfo[MiddleQuads].textureCoordinates[0].reserve( numQuadVertices * 2);
        substreamInfo[MiddleQuads].textureCoordinates[1].reserve( numQuadVertices * 2);
    }

    // Lines second
    for(int i = 1; i < 5; ++i)
    {
        substreamInfo[i].vertices.reserve(numLineVertices * 2);
        // Don't reserve memory if we aren't sending the texture coordinates.
        if(sendNeighborTextureCoordinates)
        {
            for(int j = 0; j < 5; ++j)
                substreamInfo[i].textureCoordinates.push_back( std::vector<short>() );

            for(int j = 0; j < 5; ++j)
                substreamInfo[i].textureCoordinates[j].reserve( numLineVertices * 2);
        }
    }

    // Points last
    for(int i = 5; i < 9; ++i)
    {
        substreamInfo[i].vertices.reserve(numPointVertices * 2);
        // Don't reserve memory if we aren't sending the texture coordinates.
        if(sendNeighborTextureCoordinates)
        {
            for(int j = 0; j < 6; ++j)
                substreamInfo[i].textureCoordinates.push_back( std::vector<short>() );

            for(int j = 0; j < 6; ++j)
                substreamInfo[i].textureCoordinates[j].reserve( numPointVertices * 4);
        }
    }

}

void SegmentationSimulator::DrawMiddleQuads(SubstreamInformation* substreamInfo)
{
    rendering::rtgi::DebugDrawVerticesTextureCoordinatesShort( 
        &substreamInfo[MiddleQuads].vertices, 
        &substreamInfo[MiddleQuads].textureCoordinates,
        2,
        2,
        rendering::rtgi::DebugDrawVerticesPrimitive_Quad);
}

void SegmentationSimulator::DrawTopLines(SubstreamInformation* substreamInfo)
{
    rendering::rtgi::DebugDrawVerticesTextureCoordinatesShort( 
        &substreamInfo[TopLines].vertices,
        &substreamInfo[1].textureCoordinates,
        2,
        2,
        rendering::rtgi::DebugDrawVerticesPrimitive_Line);
}

void SegmentationSimulator::DrawLeftLines(SubstreamInformation* substreamInfo)
{
    rendering::rtgi::DebugDrawVerticesTextureCoordinatesShort( 
        &substreamInfo[LeftLines].vertices,
        &substreamInfo[LeftLines].textureCoordinates,
        2,
        2,
        rendering::rtgi::DebugDrawVerticesPrimitive_Line);
}

void SegmentationSimulator::DrawBottomLines(SubstreamInformation* substreamInfo)
{
    rendering::rtgi::DebugDrawVerticesTextureCoordinatesShort( 
        &substreamInfo[BottomLines].vertices,
        &substreamInfo[BottomLines].textureCoordinates,
        2,
        2,
        rendering::rtgi::DebugDrawVerticesPrimitive_Line);
}

void SegmentationSimulator::DrawRightLines(SubstreamInformation* substreamInfo)
{
    rendering::rtgi::DebugDrawVerticesTextureCoordinatesShort( 
        &substreamInfo[RightLines].vertices,
        &substreamInfo[RightLines].textureCoordinates,
        2,
        2,
        rendering::rtgi::DebugDrawVerticesPrimitive_Line);
}

void SegmentationSimulator::DrawTopLeftPoints(SubstreamInformation* substreamInfo)
{
    rendering::rtgi::DebugDrawVerticesTextureCoordinatesShort( 
        &substreamInfo[TopLeftPoints].vertices,
        &substreamInfo[TopLeftPoints].textureCoordinates,
        2,
        2,
        rendering::rtgi::DebugDrawVerticesPrimitive_Point);
}

void SegmentationSimulator::DrawTopRightPoints(SubstreamInformation* substreamInfo)
{
    rendering::rtgi::DebugDrawVerticesTextureCoordinatesShort( 
        &substreamInfo[TopRightPoints].vertices,
        &substreamInfo[TopRightPoints].textureCoordinates,
        2,
        2,
        rendering::rtgi::DebugDrawVerticesPrimitive_Point);
}

void SegmentationSimulator::DrawBottomLeftPoints(SubstreamInformation* substreamInfo)
{
    rendering::rtgi::DebugDrawVerticesTextureCoordinatesShort( 
        &substreamInfo[BottomLeftPoints].vertices,
        &substreamInfo[BottomLeftPoints].textureCoordinates,
        2,
        2,
        rendering::rtgi::DebugDrawVerticesPrimitive_Point);
}

void SegmentationSimulator::DrawBottomRightPoints(SubstreamInformation* substreamInfo)
{
    rendering::rtgi::DebugDrawVerticesTextureCoordinatesShort( 
        &substreamInfo[BottomRightPoints].vertices,
        &substreamInfo[BottomRightPoints].textureCoordinates,
        2,
        2,
        rendering::rtgi::DebugDrawVerticesPrimitive_Point);
}

PhysicalCoordinate* SegmentationSimulator::GetLineNeighborTileCoordinates(PhysicalTile* physicalTile, DirectionEnum direction)
{
    VirtualCoordinate virtualCoordinate = physicalTile->GetVirtualAddress();

    PhysicalCoordinate* neighbors = new PhysicalCoordinate[5];

    int newFrontY = virtualCoordinate[math::Y], newFrontX = virtualCoordinate[math::X];

    switch(direction)
    {
    case Direction_Up:
        newFrontY = virtualCoordinate[math::Y] + 1;
        break;

    case Direction_Down:
        newFrontY = virtualCoordinate[math::Y] - 1;
        break;

    case Direction_Left:
        newFrontX = virtualCoordinate[math::X] - 1;
        break;

    case Direction_Right:
        newFrontX = virtualCoordinate[math::X] + 1;
        break;

    }

    int newZUp = virtualCoordinate[math::Z] + 1;
    int newZDown = virtualCoordinate[math::Z] - 1;

    // We aren't computing anything useful so if this messes any mock calculations
    // up then it doesn't matter.
    newZUp = newZUp < DATA_SIZE ? newZUp : DATA_SIZE - 1;
    newZDown = newZDown >= 0 ? newZDown : 0;
    newFrontY = newFrontY < SegmentationSimulator::NUM_TILES_PER_ROW_COLUMN_OF_DATA ? newFrontY : SegmentationSimulator::NUM_TILES_PER_ROW_COLUMN_OF_DATA - 1;
    newFrontY = newFrontY >= 0 ? newFrontY : 0;
    newFrontX = newFrontX < SegmentationSimulator::NUM_TILES_PER_ROW_COLUMN_OF_DATA ? newFrontX : SegmentationSimulator::NUM_TILES_PER_ROW_COLUMN_OF_DATA - 1;
    newFrontX = newFrontX >= 0 ? newFrontX : 0;

    // upstairs
    neighbors[0] = mPageTable->GetPhysicalAddress( VirtualCoordinate( virtualCoordinate[math::X],
        virtualCoordinate[math::Y],
        newZUp) );

    // downstairs
    neighbors[1] = mPageTable->GetPhysicalAddress( VirtualCoordinate( virtualCoordinate[math::X],
        virtualCoordinate[math::Y],
        newZDown) );

    // front
    neighbors[2] = mPageTable->GetPhysicalAddress( VirtualCoordinate( newFrontX,
        newFrontY,
        virtualCoordinate[math::Z]) );

    // front upstairs
    neighbors[3] = mPageTable->GetPhysicalAddress( VirtualCoordinate( newFrontX,
        newFrontY,
        newZUp) );

    // front downstairs
    neighbors[4] = mPageTable->GetPhysicalAddress( VirtualCoordinate( newFrontX,
        newFrontY,
        newZDown) );

    return neighbors;
}

PhysicalCoordinate* SegmentationSimulator::GetPointNeighborTileCoordinates(PhysicalTile* physicalTile, int direction)
{
    VirtualCoordinate virtualCoordinate = physicalTile->GetVirtualAddress();

    PhysicalCoordinate* neighbors = new PhysicalCoordinate[11];

    int newFrontY = virtualCoordinate[math::Y], newFrontX = virtualCoordinate[math::X];
    int newLeftX = virtualCoordinate[math::X], newLeftY = virtualCoordinate[math::Y];
    int newFrontLeftX = virtualCoordinate[math::X], newFrontLeftY = virtualCoordinate[math::Y];

    switch(direction)
    {
    case Direction_Up | Direction_Left:
        newFrontY += 1;
        newLeftX -= 1;
        newFrontLeftX = newLeftX;
        newFrontLeftY = newFrontY;
        break;

    case Direction_Down | Direction_Left:
        newFrontY -= 1;
        newLeftX -= 1;
        newFrontLeftX = newLeftX;
        newFrontLeftY = newFrontY;
        break;

    case Direction_Right | Direction_Down:
        newFrontX += 1;
        newLeftY -= 1;
        newFrontLeftX = newFrontX;
        newFrontLeftY = newLeftY;
        break;

    case Direction_Right | Direction_Up:
        newFrontX += 1;
        newLeftY += 1;
        newFrontLeftX = newFrontX;
        newFrontLeftY = newLeftY;
        break;

    }

    int newZUp = virtualCoordinate[math::Z] + 1;
    int newZDown = virtualCoordinate[math::Z] - 1;

    // We aren't computing anything useful so if this messes any mock calculations
    // up then it doesn't matter.
    newZUp = newZUp < DATA_SIZE ? newZUp : DATA_SIZE - 1;
    newZDown = newZDown >= 0 ? newZDown : 0;
    newFrontY = newFrontY < SegmentationSimulator::NUM_TILES_PER_ROW_COLUMN_OF_DATA ? newFrontY : SegmentationSimulator::NUM_TILES_PER_ROW_COLUMN_OF_DATA - 1;
    newFrontY = newFrontY >= 0 ? newFrontY : 0;
    newFrontX = newFrontX < SegmentationSimulator::NUM_TILES_PER_ROW_COLUMN_OF_DATA ? newFrontX : SegmentationSimulator::NUM_TILES_PER_ROW_COLUMN_OF_DATA - 1;
    newFrontX = newFrontX >= 0 ? newFrontX : 0;
    newLeftX = newLeftX < SegmentationSimulator::NUM_TILES_PER_ROW_COLUMN_OF_DATA ? newLeftX : SegmentationSimulator::NUM_TILES_PER_ROW_COLUMN_OF_DATA - 1;
    newLeftX = newLeftX >= 0 ? newLeftX : 0;
    newLeftY = newLeftY < SegmentationSimulator::NUM_TILES_PER_ROW_COLUMN_OF_DATA ? newLeftY : SegmentationSimulator::NUM_TILES_PER_ROW_COLUMN_OF_DATA - 1;
    newLeftY = newLeftY >= 0 ? newLeftY : 0;
    newFrontLeftX = newFrontLeftX < SegmentationSimulator::NUM_TILES_PER_ROW_COLUMN_OF_DATA ? newFrontLeftX : SegmentationSimulator::NUM_TILES_PER_ROW_COLUMN_OF_DATA - 1;
    newFrontLeftX = newFrontLeftX >= 0 ? newFrontLeftX : 0;
    newFrontLeftY = newFrontLeftY < SegmentationSimulator::NUM_TILES_PER_ROW_COLUMN_OF_DATA ? newFrontLeftY : SegmentationSimulator::NUM_TILES_PER_ROW_COLUMN_OF_DATA - 1;
    newFrontLeftY = newFrontLeftY >= 0 ? newFrontLeftY : 0;
    

    // upstairs
    neighbors[0] = mPageTable->GetPhysicalAddress( VirtualCoordinate( virtualCoordinate[math::X],
        virtualCoordinate[math::Y],
        newZUp) );

    // downstairs
    neighbors[1] = mPageTable->GetPhysicalAddress( VirtualCoordinate( virtualCoordinate[math::X],
        virtualCoordinate[math::Y],
        newZDown) );

    // front
    neighbors[2] = mPageTable->GetPhysicalAddress( VirtualCoordinate( newFrontX,
        newFrontY,
        virtualCoordinate[math::Z]) );

    // front upstairs
    neighbors[3] = mPageTable->GetPhysicalAddress( VirtualCoordinate( newFrontX,
        newFrontY,
        newZUp) );

    // front downstairs
    neighbors[4] = mPageTable->GetPhysicalAddress( VirtualCoordinate( newFrontX,
        newFrontY,
        newZDown) );

    // left
    neighbors[5] = mPageTable->GetPhysicalAddress( VirtualCoordinate( newLeftX,
        newLeftY,
        virtualCoordinate[math::Z]) );

    // left upstairs
    neighbors[6] = mPageTable->GetPhysicalAddress( VirtualCoordinate( newLeftX,
        newLeftY,
        newZUp) );

    // left downstairs
    neighbors[7] = mPageTable->GetPhysicalAddress( VirtualCoordinate( newLeftX,
        newLeftY,
        newZDown) );

    // front left
    neighbors[8] = mPageTable->GetPhysicalAddress( VirtualCoordinate( newFrontLeftX,
        newLeftY,
        virtualCoordinate[math::Z]) );

    // front left upstairs
    neighbors[9] = mPageTable->GetPhysicalAddress( VirtualCoordinate( newFrontLeftX,
        newFrontLeftY,
        newZUp) );

    // front left downstairs
    neighbors[10] = mPageTable->GetPhysicalAddress( VirtualCoordinate( newFrontLeftX,
        newFrontLeftY,
        newZDown) );


    return neighbors;
}

PhysicalCoordinate* SegmentationSimulator::GetQuadNeighborTileCoordinates(PhysicalTile* physicalTile)
{
    VirtualCoordinate virtualCoordinate = physicalTile->GetVirtualAddress();
    
    PhysicalCoordinate* neighbors = new PhysicalCoordinate[2];

    int newZUp = virtualCoordinate[math::Z] + 1;
    int newZDown = virtualCoordinate[math::Z] - 1;

    // We aren't computing anything useful so if this messes any mock calculations
    // up then it doesn't matter.
    newZUp = newZUp < DATA_SIZE ? newZUp : newZUp - 1;
    newZDown = newZDown >= 0 ? newZDown : newZDown + 1;

    neighbors[0] = mPageTable->GetPhysicalAddress( VirtualCoordinate( virtualCoordinate[math::X],
                                                                      virtualCoordinate[math::Y],
                                                                      newZUp) );

    neighbors[1] = mPageTable->GetPhysicalAddress( VirtualCoordinate( virtualCoordinate[math::X],
                                                                      virtualCoordinate[math::Y],
                                                                      newZDown) );

    return neighbors;
}

void SegmentationSimulator::InitializeTextures()
{
    math::Matrix44 identityMatrix, projectionMatrix;

    const int gpuMemorySize = SegmentationSimulator::GPU_MEMORY_SIZE;

    int oldViewportWidth, oldViewportHeight;

    rendering::rtgi::GetViewport(oldViewportWidth, oldViewportHeight);

    rendering::rtgi::SetViewport( gpuMemorySize, gpuMemorySize);

    identityMatrix.SetToIdentity();
    projectionMatrix.SetTo2DOrthographic( 0, gpuMemorySize, gpuMemorySize, 0 );

    rendering::rtgi::SetTransformMatrix( identityMatrix );
    rendering::rtgi::SetViewMatrix( identityMatrix );
    rendering::rtgi::SetProjectionMatrix( projectionMatrix );
    rendering::rtgi::SetColorWritingEnabled( true );
    rendering::rtgi::SetAlphaBlendingEnabled( false );

    for( int i = 0; i < NUM_TEXTURES; ++i )
    {
        mFrameBuffer->Bind(mGradientTexture[i]);
        rendering::rtgi::ClearColorBuffer( rendering::rtgi::ColorRGBA( 0.0f, 0.0f, 0.0f, 0.0f ) );
        mFrameBuffer->Unbind();

        mFrameBuffer->Bind(mCurvatureTexture[i]);
        rendering::rtgi::ClearColorBuffer( rendering::rtgi::ColorRGBA( 0.0f, 0.0f, 0.0f, 0.0f) );
        mFrameBuffer->Unbind();

        mFrameBuffer->Bind(mMedicalImagingDataTexture[i]);
        rendering::rtgi::ClearColorBuffer( rendering::rtgi::ColorRGBA( 0.0f, 0.00f, 0.00f, 0.0f ) );
        mFrameBuffer->Unbind();
    }

    rendering::rtgi::SetViewport(oldViewportWidth, oldViewportHeight);
}

void SegmentationSimulator::AllocateTextures()
{
    mPhysicalMemoryTexture = new rendering::rtgi::Texture*[ NUM_TEXTURES ];
    mNewPhysicalMemoryTexture = new rendering::rtgi::Texture*[ NUM_TEXTURES ];
    mGradientTexture = new rendering::rtgi::Texture*[ NUM_TEXTURES ];
    mCurvatureTexture = new rendering::rtgi::Texture*[ NUM_TEXTURES ];
    mMemoryAllocationRequestOneTexture = new rendering::rtgi::Texture*[ NUM_TEXTURES ];
    mMemoryAllocationRequestTwoTexture = new rendering::rtgi::Texture*[ NUM_TEXTURES ];
    mCombinedMemoryAllocationRequestTexture = new rendering::rtgi::Texture*[ NUM_TEXTURES ];
    mMedicalImagingDataTexture = new rendering::rtgi::Texture*[ NUM_TEXTURES ];

    for( int i = 0; i < NUM_TEXTURES; ++i )
    {
        rendering::rtgi::TextureDataDesc textureDataDesc;

        textureDataDesc.pixelFormat = rendering::rtgi::TexturePixelFormat_L32_F_DENORM;
        textureDataDesc.dimensions  = rendering::rtgi::TextureDimensions_2D;

        mPhysicalMemoryTexture[i] = rendering::rtgi::CreateTexture( textureDataDesc );
        mPhysicalMemoryTexture[i]->AddRef();
        mPhysicalMemoryTexture[i]->Unbind();

        mGradientTexture[i] = rendering::rtgi::CreateTexture( textureDataDesc );
        mGradientTexture[i]->AddRef();
        mGradientTexture[i]->Unbind();

        mCurvatureTexture[i] = rendering::rtgi::CreateTexture( textureDataDesc );
        mCurvatureTexture[i]->AddRef();
        mCurvatureTexture[i]->Unbind();
        
        mMemoryAllocationRequestOneTexture[i] = rendering::rtgi::CreateTexture( textureDataDesc );
        mMemoryAllocationRequestOneTexture[i]->AddRef();
        mMemoryAllocationRequestOneTexture[i]->Unbind();

        mMemoryAllocationRequestTwoTexture[i] = rendering::rtgi::CreateTexture( textureDataDesc );
        mMemoryAllocationRequestTwoTexture[i]->AddRef();
        mMemoryAllocationRequestTwoTexture[i]->Unbind();

        mCombinedMemoryAllocationRequestTexture[i] = rendering::rtgi::CreateTexture( textureDataDesc );
        mCombinedMemoryAllocationRequestTexture[i]->AddRef();
        mCombinedMemoryAllocationRequestTexture[i]->Unbind();
        
        mMedicalImagingDataTexture[i] = rendering::rtgi::CreateTexture( textureDataDesc );
        mMedicalImagingDataTexture[i]->AddRef();
        mMedicalImagingDataTexture[i]->Unbind();

        if( newPhysicalMemoryTextureDesc == NULL )
        {
            newPhysicalMemoryTextureDesc = new rendering::rtgi::TextureDataDesc();
            newPhysicalMemoryTextureDesc->pixelFormat = textureDataDesc.pixelFormat;
            newPhysicalMemoryTextureDesc->dimensions  = rendering::rtgi::TextureDimensions_2D;
        }

        mNewPhysicalMemoryTexture[i] = rendering::rtgi::CreateTexture( *newPhysicalMemoryTextureDesc );
        mNewPhysicalMemoryTexture[i]->AddRef();
        mNewPhysicalMemoryTexture[i]->Unbind();
    }
}

void SegmentationSimulator::DeallocateTextures()
{
    for( int i = 0; i < NUM_TEXTURES; ++i )
    {
        mPhysicalMemoryTexture[i]->Release();
        mPhysicalMemoryTexture[i] = NULL;

        mGradientTexture[i]->Release();
        mGradientTexture[i] = NULL;

        mCurvatureTexture[i]->Release();
        mCurvatureTexture[i] = NULL;

        mMemoryAllocationRequestOneTexture[i]->Release();
        mMemoryAllocationRequestOneTexture[i] = NULL;

        mMemoryAllocationRequestTwoTexture[i]->Release();
        mMemoryAllocationRequestTwoTexture[i] = NULL;

        mCombinedMemoryAllocationRequestTexture[i]->Release();
        mCombinedMemoryAllocationRequestTexture[i] = NULL;

        mMedicalImagingDataTexture[i]->Release();
        mMedicalImagingDataTexture[i] = NULL;

        if(mNewPhysicalMemoryTexture[i])
        {
            mNewPhysicalMemoryTexture[i]->Release();
            mNewPhysicalMemoryTexture[i] = NULL;
        }
    }

    delete [] mPhysicalMemoryTexture;
    delete [] mNewPhysicalMemoryTexture;
    delete [] mGradientTexture;
    delete [] mCurvatureTexture;
    delete [] mMemoryAllocationRequestOneTexture;
    delete [] mMemoryAllocationRequestTwoTexture;
    delete [] mCombinedMemoryAllocationRequestTexture;
    delete [] mMedicalImagingDataTexture;
}

#ifdef LEFOHN_BENCHMARK

void SegmentationSimulator::CreateShaderPrograms()
{
    mComputeNewLevelSetDataProgram = rendering::rtgi::CreateShaderProgram( "shaders/Lefohn/computeNewLevelSetFieldVertexProgram.cg", "shaders/Lefohn/computeNewLevelSetFieldFragmentProgram.cg" );
    mComputeNewLevelSetDataProgram->AddRef();

    mComputeNewCurvatureQuadProgram = rendering::rtgi::CreateShaderProgram( "shaders/Lefohn/computeNewCurvatureQuadVertexProgram.cg", "shaders/Lefohn/computeNewCurvatureQuadFragmentProgram.cg" );
    mComputeNewCurvatureQuadProgram->AddRef();

    mComputeNewCurvatureLineProgram = rendering::rtgi::CreateShaderProgram( "shaders/Lefohn/computeNewCurvatureLineVertexProgram.cg", "shaders/Lefohn/computeNewCurvatureLineFragmentProgram.cg" );
    mComputeNewCurvatureLineProgram->AddRef();

    mComputeNewCurvaturePointProgram = rendering::rtgi::CreateShaderProgram( "shaders/Lefohn/computeNewCurvaturePointVertexProgram.cg", "shaders/Lefohn/computeNewCurvaturePointFragmentProgram.cg" );
    mComputeNewCurvaturePointProgram->AddRef();

    mComputeNewMemoryAllocationProgram = rendering::rtgi::CreateShaderProgram( "shaders/Lefohn/computeNewMemoryAllocationVertexProgram.cg", "shaders/Lefohn/computeNewMemoryAllocationFragmentProgram.cg" );
    mComputeNewMemoryAllocationProgram->AddRef();

    mComputeCombinedMemoryAllocationProgram = rendering::rtgi::CreateShaderProgram( "shaders/Lefohn/computeCombinedMemoryAllocationVertexProgram.cg", "shaders/Lefohn/computeCombinedMemoryAllocationFragmentProgram.cg" );
    mComputeCombinedMemoryAllocationProgram->AddRef();

}

#endif

void SegmentationSimulator::DestroyShaderPrograms()
{
    mComputeNewLevelSetDataProgram->Release();
    mComputeNewLevelSetDataProgram = NULL;

    mComputeNewCurvatureQuadProgram->Release();
    mComputeNewCurvatureQuadProgram = NULL;

    mComputeNewCurvatureLineProgram->Release();
    mComputeNewCurvatureLineProgram = NULL;

    mComputeNewCurvaturePointProgram->Release();
    mComputeNewCurvaturePointProgram = NULL;

    /*mComputeNewGradientsQuadProgram->Release();
    mComputeNewGradientsQuadProgram = NULL;

    mComputeNewGradientsLineProgram->Release();
    mComputeNewGradientsLineProgram = NULL;

    mComputeNewGradientsPointProgram->Release();
    mComputeNewGradientsPointProgram = NULL;*/

    mComputeNewMemoryAllocationProgram->Release();
    mComputeNewMemoryAllocationProgram = NULL;

    mComputeCombinedMemoryAllocationProgram->Release();
    mComputeCombinedMemoryAllocationProgram = NULL;
}

}
