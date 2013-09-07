#ifndef LEFOHN_SEGMENTATION_SIMULATOR_HPP
#define LEFOHN_SEGMENTATION_SIMULATOR_HPP

namespace rendering
{

namespace rtgi
{
    class Texture;
    class FrameBufferObject;
    class ShaderProgram;
    struct TextureDataDesc;
}

}

namespace math
{
    class Matrix44;
}

#include "LefohnPageTable.hpp"

#include <vector>

namespace lefohn
{
    class OpenGLFrameBufferObjectRGBA32F;

    int ComputeNumberOfActiveTiles( float* data );
    bool NonZeroGradient(float* data, int xTile, int yTile, int zTile);

    void RunLefohnSimulation();

class SegmentationSimulator
{
public:

    SegmentationSimulator( int numberActiveTiles );
    ~SegmentationSimulator();

    // Does the mock update.
    // Returns the time it takes to update via shaders and downsampling only.
    // Does not time the steps to change the level set data via the incoming
    // number or any cleanup.
    double Update(int numberActiveTiles);

    static const int GPU_MEMORY_SIZE;
    static const int DATA_SIZE;
    static const int TILE_SIZE;
    static const int NUM_TILES_PER_ROW_COLUMN_OF_DATA;
    static const int NUM_TILES_PER_ROW_COLUMN_OF_DATA_SQUARED;
    static const int NUM_TILES_PER_ROW_COLUMN;
    static const int NUM_TILES_PER_ROW_COLUMN_SQUARED;
    static const int NUM_TEXTURES;
    static const int MAX_ACTIVE_TILES;

private:
    // So that tests can have access to member variables.
    friend class VolumeTestCase;

    struct SubstreamInformation
    {
        std::vector<short> vertices;
        std::vector< std::vector<short> > textureCoordinates;
    };

    enum SubstreamInformationPosition
    {
        MiddleQuads,
        TopLines,
        LeftLines,
        BottomLines,
        RightLines,
        TopLeftPoints,
        TopRightPoints,
        BottomLeftPoints,
        BottomRightPoints
    };

    // Builds the geometry for the two passes that require 16x16 quads.
    std::vector<short>** Build16x16Quads();

    // Builds the geometry for the nine substreams.
    // Note that two substream passes don't require neighbor coordinates.
    // This will return an array of substream information objects. Each element should be rendered
    // sequentially to different textures in the render texture arrays (physical memory)
    SubstreamInformation ** BuildSubstreamInformation(bool sendNeighborTextureCoordinates = true);
    void ReserveMemoryForSubstreams(SubstreamInformation* substreamInfo,
                                    int numQuadVertices, int numLineVertices,
                                    int numPointVertices, bool sendNeighborTextureCoordinates);

    void BuildMiddleQuads( SubstreamInformation* substreamInfo, PhysicalTile* currentTile,
        int startOffset, int textureOffsetY, bool sendNeighborTextureCoordinates = true );
    void BuildTopLines( SubstreamInformation* substreamInfo, PhysicalTile* currentTile,
        int startOffset, int textureOffsetY, bool sendNeighborTextureCoordinates = true );
    void BuildLeftLines( SubstreamInformation* substreamInfo, PhysicalTile* currentTile,
        int startOffset, int textureOffsetY, bool sendNeighborTextureCoordinates = true );
    void BuildBottomLines( SubstreamInformation* substreamInfo, PhysicalTile* currentTile,
        int startOffset, int textureOffsetY, bool sendNeighborTextureCoordinates = true );
    void BuildRightLines( SubstreamInformation* substreamInfo, PhysicalTile* currentTile,
        int startOffset, int textureOffsetY, bool sendNeighborTextureCoordinates = true );
    void BuildTopLeftPoints( SubstreamInformation* substreamInfo, PhysicalTile* currentTile,
        int startOffset, int textureOffsetY, bool sendNeighborTextureCoordinates = true );
    void BuildTopRightPoints( SubstreamInformation* substreamInfo, PhysicalTile* currentTile,
        int startOffset, int textureOffsetY, bool sendNeighborTextureCoordinates = true );
    void BuildBottomLeftPoints( SubstreamInformation* substreamInfo, PhysicalTile* currentTile,
        int startOffset, int textureOffsetY, bool sendNeighborTextureCoordinates = true );
    void BuildBottomRightPoints( SubstreamInformation* substreamInfo, PhysicalTile* currentTile,
        int startOffset, int textureOffsetY, bool sendNeighborTextureCoordinates = true );


    void AllocateTextures();
    void DeallocateTextures();
    void InitializeTextures();

#ifdef LEFOHN_BENCHMARK
    void CreateShaderPrograms();
#endif

    void DestroyShaderPrograms();

    void InitializePageTables( int numberActiveTiles );

    // Render the active physical tiles to the physical memory texture.
    void RenderToGPUMemory();
    // Used by renderToGPUMemory() to render the static physical tiles.
    void RenderStaticPhysicalTiles();


    // GPU portion of Lefohn.
    void ComputeNewCurvaturesAndGradients(const math::Matrix44& projectionMatrix, SubstreamInformation** substreamInfo);
    void ComputeNewMemoryAllocations(const math::Matrix44& projectionMatrix, SubstreamInformation** substreamInfo);
    void ComputeNewLevelSetField(const math::Matrix44& projectionMatrix, std::vector<short>** quadVertices);    
    void ComputeCombinedAllocation(const math::Matrix44& projectionMatrix, std::vector<short>** quadVertices);
    void UploadNewActiveTiles();
    
    // For ComputeNewCurvatures, ComputeNewGradients, ComputeMemoryAllocationOne, and
    // ComputeMemoryAllocationTwo
    void DrawMiddleQuads(SubstreamInformation* substreamInfo);
    void DrawTopLines(SubstreamInformation* substreamInfo);
    void DrawLeftLines(SubstreamInformation* substreamInfo);
    void DrawBottomLines(SubstreamInformation* substreamInfo);
    void DrawRightLines(SubstreamInformation* substreamInfo);
    void DrawTopLeftPoints(SubstreamInformation* substreamInfo);
    void DrawTopRightPoints(SubstreamInformation* substreamInfo);
    void DrawBottomLeftPoints(SubstreamInformation* substreamInfo);
    void DrawBottomRightPoints(SubstreamInformation* substreamInfo);

    // Generates the miplevels of the memory allocation message and reads it into CPU memory.
    // Sets mMemoryAllocationMessageData to point to the data (needs to be cleaned up every frame
    // because the system doesn't clean it up for us).
    void GenerateMipLevelsAndReadMessage();

    // Ensure downloadable texture is in GPU memory.
    //void ReRenderDownloadableTexture();

    // Specifies which direction to get a neighbor in for the following functions
    // that get more than one kind of neighbor (just for lines and points).
    enum DirectionEnum
    {
        Direction_Up = 0x0001,
        Direction_Right = 0x0010,
        Direction_Down = 0x0100,
        Direction_Left = 0x1000
    };

    // *** None of the Get*NeighborTileCoordinates cleans up after themselves ***
    // You must DELETE the returned array.

    // Returns upstairs and downstairs neighbor, respectively.
    PhysicalCoordinate* GetQuadNeighborTileCoordinates(PhysicalTile* physicalTile);
    // Returns upstairs, downstairs, front, front upstairs, front downstairs neighbors, respectively
    PhysicalCoordinate* GetLineNeighborTileCoordinates(PhysicalTile* physicalTile, DirectionEnum direction);
    // Returns upstairs, downstairs, front, front upstairs, front downstairs, left, left upstairs,
    // left downstairs, front left, front left upstairs, and front left downstairs neighbors, respectively
    PhysicalCoordinate* GetPointNeighborTileCoordinates(PhysicalTile* physicalTile, int direction);

    
    PageTable*                mPageTable;
    InversePageTable*        mInversePageTable;
    VirtualTile*            mVirtualMemoryTiles;
    VirtualTile*            mOldVirtualMemoryTiles;
    PageTable*                mOldPageTable;
    InversePageTable*        mOldInversePageTable;
    int                        mNumActiveVirtualMemoryTiles;
    int                        mDifferenceActiveTiles;
    int                        mNumOldActiveTiles;
    int                            mNumTexturesUsed;
    

    // Single channel floating point textures arrays
    rendering::rtgi::Texture**        mPhysicalMemoryTexture;
    rendering::rtgi::Texture**        mNewPhysicalMemoryTexture;
    rendering::rtgi::Texture**        mGradientTexture;
    rendering::rtgi::Texture**        mCurvatureTexture;
    rendering::rtgi::Texture**        mMemoryAllocationRequestOneTexture;
    rendering::rtgi::Texture**        mMemoryAllocationRequestTwoTexture;
    rendering::rtgi::Texture**        mCombinedMemoryAllocationRequestTexture;
    rendering::rtgi::Texture**        mMedicalImagingDataTexture;
    
    rendering::rtgi::ShaderProgram*    mComputeNewLevelSetDataProgram;
    rendering::rtgi::ShaderProgram*    mComputeNewCurvatureQuadProgram;
    rendering::rtgi::ShaderProgram*    mComputeNewCurvatureLineProgram;
    rendering::rtgi::ShaderProgram*    mComputeNewCurvaturePointProgram;

    rendering::rtgi::ShaderProgram*    mComputeNewMemoryAllocationProgram;
    rendering::rtgi::ShaderProgram*    mComputeCombinedMemoryAllocationProgram;

    rendering::rtgi::TextureDataDesc* newPhysicalMemoryTextureDesc;


    OpenGLFrameBufferObjectRGBA32F* mFrameBuffer;

    float*        mMemoryAllocationMessageData;

    float* mMockDataToUpload;


};

}

#endif
