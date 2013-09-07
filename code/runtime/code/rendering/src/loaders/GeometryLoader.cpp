#include "rendering/loaders/GeometryLoader.hpp"

#include <omp.h>

#include <FCollada.h> 
#include <FCDocument/FCDocument.h>
#include <FCDocument/FCDLibrary.h>
#include <FCDocument/FCDGeometry.h>
#include <FCDocument/FCDGeometryMesh.h>
#include <FCDocument/FCDGeometryPolygons.h>
#include <FCDocument/FCDGeometryPolygonsInput.h>
#include <FCDocument/FCDGeometrySource.h>
#include <FUtils/FUDaeEnum.h>

#include "core/Assert.hpp"

#include "container/Map.hpp"
#include "container/HashMap.hpp"
#include "container/Array.hpp"
#include "container/List.hpp"

#include "content/Inventory.hpp"

#include "rendering/rtgi/RTGI.hpp"
#include "rendering/rtgi/IndexBuffer.hpp"
#include "rendering/rtgi/VertexBuffer.hpp"

#include "rendering/Material.hpp"
#include "rendering/MaterialStrip.hpp"
#include "rendering/Renderable.hpp"

namespace rendering
{

#pragma pack( push, 16 )
struct Vertex
{
    float objectSpacePositionX,  objectSpacePositionY,  objectSpacePositionZ;
    float objectSpaceNormalX,    objectSpaceNormalY,    objectSpaceNormalZ;

    float textureSpaceTangentX,  textureSpaceTangentY,  textureSpaceTangentZ;
    float textureSpaceBinormalX, textureSpaceBinormalY, textureSpaceBinormalZ;

    float diffuseMapTextureCoordinateU, diffuseMapTextureCoordinateV;
};
#pragma pack( pop )

enum VertexLayoutNumCoordinates
{
    VertexLayoutNumCoordinates_ObjectSpacePosition         = 3,
    VertexLayoutNumCoordinates_ObjectSpaceNormal           = 3,

    VertexLayoutNumCoordinates_TextureSpaceTangent         = 3,
    VertexLayoutNumCoordinates_TextureSpaceBinormal        = 3,

    VertexLayoutNumCoordinates_DiffuseMapTextureCoordinate = 2
};

enum VertexLayoutByteOffset
{
    VertexLayoutByteOffset_ObjectSpacePosition         = 0  *  sizeof( float ),
    VertexLayoutByteOffset_ObjectSpaceNormal           = 3  *  sizeof( float ),

    VertexLayoutByteOffset_TextureSpaceTangent         = 6  *  sizeof( float ),
    VertexLayoutByteOffset_TextureSpaceBinormal        = 9  *  sizeof( float ),

    VertexLayoutByteOffset_DiffuseMapTextureCoordinate = 12 *  sizeof( float ),
};

static const float        VERTEX_BUFFER_DUMMY_VALUE       = 98765.4321f;

static const core::String SEMANTIC_POSITION               = "POSITION";
static const core::String SEMANTIC_NORMAL                 = "NORMAL";
static const core::String SEMANTIC_TEXTURE_SPACE_TANGENT  = "TANGENT";
static const core::String SEMANTIC_TEXTURE_SPACE_BINORMAL = "BINORMAL";
static const core::String SEMANTIC_TEXCOORD_DIFFUSE_MAP   = "TEXCOORD0";

struct GeometrySourceDesc
{
    FCDGeometrySource* geometrySource;
    int                numParamsPerDataValue;
    int                offsetInVertex;
};

struct RTGIBufferDesc
{
    unsigned short*    indexBufferRaw;
    unsigned int       numIndices;
    Vertex*            vertexBufferRawFormatted;
    unsigned int       numVertices;
    core::String       geometryName;
    core::String       materialSemantic;
    core::String       materialStripName;
};

void GeometryLoader::Load( content::Inventory* inventory, FCDocument* document )
{
    container::HashMap< core::String, container::List< RTGIBufferDesc > > geometryNameRTGIBufferDescMap;

    // get geometry library
    FCDGeometryLibrary* geometryLibrary     = document->GetGeometryLibrary();
    int                 numGeometryEntities = static_cast< int >( geometryLibrary->GetEntityCount() );

    // for each geometry
    //#pragma omp parallel for num_threads( 4 ) schedule( dynamic )
    for ( int i = 0; i < numGeometryEntities; i++ )
    {
        // get mesh
        FCDGeometry*     geometry         = geometryLibrary->GetEntity( i );
        FCDGeometryMesh* geometryMesh     = geometry->GetMesh();
        core::String     geometryName     = geometry->GetName().c_str();

        // make sure the mesh is only comprised of triangles
        Assert( geometryMesh->IsTriangles() );

        // for each triangle list
        for ( size_t j = 0; j < geometryMesh->GetPolygonsCount(); j++ )
        {
            FCDGeometryPolygons*      geometryPolygons       = geometryMesh->GetPolygons( j );
            core::String              materialSemantic       = geometryPolygons->GetMaterialSemantic().c_str();
            size_t                    numInputs              = geometryPolygons->GetInputCount();

            Assert( numInputs > 0 );

            size_t                    numIndicesPerInput     = geometryPolygons->GetInput( 0 )->GetIndexCount();
            unsigned short*           interleavedIndices     = new unsigned short[ numInputs * numIndicesPerInput ];

            // create a mapping from geometry input IDs to the number of parameters per data value,
            // the offset in the rtgi::Vertex struct, and the source data array
            container::Map< int, GeometrySourceDesc > geometryInputIDSourceDescMap;

            for ( size_t k = 0; k < geometryPolygons->GetInputCount(); k++ )
            {
                FCDGeometryPolygonsInput*    polygonsInput         = geometryPolygons->GetInput( k );
                FCDGeometrySource*           geometrySource        = polygonsInput->GetSource();
                float*                       sourceData            = geometrySource->GetData();
                FUDaeGeometryInput::Semantic semantic              = geometrySource->GetType();
                int                          numParamsPerDataValue = -1;
                int                          offsetInVertex        = -1;

                switch ( semantic )
                {
                case FUDaeGeometryInput::POSITION:
                    offsetInVertex        = VertexLayoutByteOffset_ObjectSpacePosition / sizeof( float );
                    numParamsPerDataValue = VertexLayoutNumCoordinates_ObjectSpacePosition;
                    break;

                case FUDaeGeometryInput::NORMAL:
                    offsetInVertex        = VertexLayoutByteOffset_ObjectSpaceNormal / sizeof( float );
                    numParamsPerDataValue = VertexLayoutNumCoordinates_ObjectSpaceNormal;
                    break;

                case FUDaeGeometryInput::TEXCOORD:
                    offsetInVertex        = VertexLayoutByteOffset_DiffuseMapTextureCoordinate / sizeof( float );
                    numParamsPerDataValue = VertexLayoutNumCoordinates_DiffuseMapTextureCoordinate;
                    break;

                case FUDaeGeometryInput::TEXTANGENT:
                    offsetInVertex        = VertexLayoutByteOffset_TextureSpaceTangent / sizeof( float );
                    numParamsPerDataValue = VertexLayoutNumCoordinates_TextureSpaceTangent;
                    break;

                case FUDaeGeometryInput::TEXBINORMAL:
                    offsetInVertex        = VertexLayoutByteOffset_TextureSpaceBinormal / sizeof( float );
                    numParamsPerDataValue = VertexLayoutNumCoordinates_TextureSpaceBinormal;
                    break;

                default:
                    Assert( 0 );
                }

                GeometrySourceDesc desc;

                desc.geometrySource        = geometrySource;
                desc.offsetInVertex        = offsetInVertex;
                desc.numParamsPerDataValue = numParamsPerDataValue;

                geometryInputIDSourceDescMap.Insert( k, desc );
            }

            // reconstruct the interleaved index array present in the collada document
            for ( size_t k = 0; k < numInputs; k++ )
            {
                FCDGeometryPolygonsInput* polygonsInput           = geometryPolygons->GetInput( k );
                size_t                    numIndicesForInput      = polygonsInput->GetIndexCount();
                const uint32*             indicesForInput         = polygonsInput->GetIndices();               
                int                       interleavedIndicesIndex = k;

                Assert( numIndicesForInput == numIndicesPerInput );

                for ( size_t m = 0; m < numIndicesForInput; m++ )
                {
                    interleavedIndices[ interleavedIndicesIndex ] = indicesForInput[ m ];
                    interleavedIndicesIndex += numInputs;
                }
            }

            // walk the interleaved index array to count the number of unique (i1, i2, ..., iN) tuples
            // (since each unique tuple requires a unique vertex), we can generate an rtgi index buffer
            // at the same time
            size_t          interleavedIndicesIndex = 0;
            int             rtgiIndicesIndex        = 0;
            unsigned short* rtgiIndices             = new unsigned short[ numIndicesPerInput ];

            container::Array< container::Array< int > > uniqueTuples;
            container::HashMap< core::String, int >     uniqueTupleIndexHashMap;

            uniqueTuples.Reserve( numIndicesPerInput );

            while ( interleavedIndicesIndex < numInputs * numIndicesPerInput )
            {
                container::Array< int > currentTuple;
                core::String           currentTupleString;

                for ( size_t m = 0; m < numInputs; m++ )
                {
                    // construct a hash key to make it easier to tell if our current tuple is already in the set of unique tuples
                    core::String indexString  = core::String( "%1" ).arg( interleavedIndices[ interleavedIndicesIndex ] );
                    currentTupleString        = currentTupleString + "-" + indexString;

                    currentTuple.Append( interleavedIndices[ interleavedIndicesIndex++ ] );
                }

                int indexOfCurrentTuple = -1;

                if ( uniqueTupleIndexHashMap.Contains( currentTupleString ) )
                {
                    // if the map contains the current tuple, then do the slow index lookup 
                    indexOfCurrentTuple = uniqueTupleIndexHashMap.Value( currentTupleString );
                }
                else
                {
                    // otherwise we know that the index will be the end of the list
                    indexOfCurrentTuple = uniqueTuples.Size();

                    uniqueTuples.Append( currentTuple );
                    uniqueTupleIndexHashMap.Insert( currentTupleString, indexOfCurrentTuple );
                }

                rtgiIndices[ rtgiIndicesIndex++ ] = indexOfCurrentTuple;
            }


            // we don't need the reconstructed interleaved index array any more, so delete it
            delete[] interleavedIndices;
            interleavedIndices = NULL;

            // allocate space for a vertex buffer now that we know the number of vertices in our mesh.
            // we do this per triangle list since the data sources aren't guaranteed to be consistent
            // across triangle lists.  therefore we risk clobbering the vertex data if we have only
            // one vertex buffer per mesh.
            int    numVertices             = uniqueTuples.Size();
            int    numFloatsInVertex       = sizeof( Vertex ) / sizeof( float );
            int    numFloatsInVertexBuffer = numVertices * numFloatsInVertex;
            float* vertexBufferRaw         = new float[ numFloatsInVertexBuffer ];

#ifdef BUILD_DEBUG
            for( int k = 0; k < numFloatsInVertexBuffer; k++ )
            {
                vertexBufferRaw[ k ] = VERTEX_BUFFER_DUMMY_VALUE;
            }
#endif

            // now we convert our set of unique tuples into a vertex buffer
            int uniqueTuplesIndex = 0;

            foreach ( container::Array< int > tuple, uniqueTuples )
            {
                Assert ( tuple.Size() == geometryPolygons->GetInputCount() );

                int inputIndex = 0;

                foreach ( int geometryInputIndex, tuple )
                {
                    GeometrySourceDesc geometrySourceDesc  = geometryInputIDSourceDescMap.Value( inputIndex );
                    int               vertexBufferRawIndex = ( uniqueTuplesIndex * numFloatsInVertex ) + geometrySourceDesc.offsetInVertex;
                    const float*      sourceData           = geometrySourceDesc.geometrySource->GetValue( geometryInputIndex );

                    for ( int m = 0; m < geometrySourceDesc.numParamsPerDataValue; m++ )
                    {
                        Assert( vertexBufferRaw[ vertexBufferRawIndex ] == VERTEX_BUFFER_DUMMY_VALUE );

                        float sourceDataParam                     = sourceData[ m ];
                        vertexBufferRaw[ vertexBufferRawIndex++ ] = sourceDataParam;
                    }

                    inputIndex++;
                }

                uniqueTuplesIndex++;
            }

            RTGIBufferDesc bufferDesc;

            bufferDesc.indexBufferRaw           = rtgiIndices;
            bufferDesc.numIndices               = numIndicesPerInput;
            bufferDesc.vertexBufferRawFormatted = reinterpret_cast< Vertex* >( vertexBufferRaw );
            bufferDesc.numVertices              = numVertices;
            bufferDesc.geometryName             = geometryName;
            bufferDesc.materialSemantic         = materialSemantic;
            bufferDesc.materialStripName        = geometryName + " - " + materialSemantic;


            container::List< RTGIBufferDesc > rtgiBufferDescList = geometryNameRTGIBufferDescMap.Value( geometryName );
            rtgiBufferDescList.Append( bufferDesc );

            geometryNameRTGIBufferDescMap.Insert( geometryName, rtgiBufferDescList );
        }
    } // end #pragma omp parallel for


    foreach_key_value ( const core::String& geometryName, container::List< RTGIBufferDesc > bufferDescList, geometryNameRTGIBufferDescMap )
    {
        container::List< MaterialStrip* > materialStrips;

        foreach ( RTGIBufferDesc bufferDesc, bufferDescList )
        {
            rtgi::IndexBuffer*  indexBuffer  = rtgi::CreateIndexBuffer( bufferDesc.indexBufferRaw, bufferDesc.numIndices );
            rtgi::VertexBuffer* vertexBuffer = rtgi::CreateVertexBuffer( bufferDesc.vertexBufferRawFormatted, bufferDesc.numVertices * sizeof( Vertex ) );

            // now create a material strip
            MaterialStrip* materialStrip = new MaterialStrip();
            SetDebugName( materialStrip, bufferDesc.materialStripName );
            SetFinalized( materialStrip, false );

            materialStrip->SetMaterialSemantic( bufferDesc.materialSemantic );

            MaterialStripRenderDesc materialStripRenderDesc;

            materialStripRenderDesc.primitiveRenderMode = rtgi::PrimitiveRenderMode_Triangles;
            materialStripRenderDesc.indexingRenderMode  = IndexingRenderMode_Indexed;
            materialStripRenderDesc.indexBuffer         = indexBuffer;
            
            materialStrip->SetMaterialStripRenderDesc( materialStripRenderDesc );

            rtgi::VertexDataSourceDesc vertexDataSourceDesc;

            vertexDataSourceDesc.vertexBuffer              = vertexBuffer;
            vertexDataSourceDesc.numVertices               = bufferDesc.numVertices;
            vertexDataSourceDesc.stride                    = sizeof( Vertex );
            vertexDataSourceDesc.vertexBufferDataType      = rtgi::VertexBufferDataType_Float;
            vertexDataSourceDesc.offset                    = VertexLayoutByteOffset_ObjectSpacePosition;
            vertexDataSourceDesc.numCoordinatesPerSemantic = VertexLayoutNumCoordinates_ObjectSpacePosition;
            materialStrip->AddVertexDataSourceDesc( SEMANTIC_POSITION, vertexDataSourceDesc );

            vertexDataSourceDesc.vertexBuffer              = vertexBuffer;
            vertexDataSourceDesc.numVertices               = bufferDesc.numVertices;
            vertexDataSourceDesc.stride                    = sizeof( Vertex );
            vertexDataSourceDesc.vertexBufferDataType      = rtgi::VertexBufferDataType_Float;
            vertexDataSourceDesc.offset                    = VertexLayoutByteOffset_ObjectSpaceNormal;
            vertexDataSourceDesc.numCoordinatesPerSemantic = VertexLayoutNumCoordinates_ObjectSpaceNormal;
            materialStrip->AddVertexDataSourceDesc( SEMANTIC_NORMAL, vertexDataSourceDesc );

            vertexDataSourceDesc.vertexBuffer              = vertexBuffer;
            vertexDataSourceDesc.numVertices               = bufferDesc.numVertices;
            vertexDataSourceDesc.stride                    = sizeof( Vertex );
            vertexDataSourceDesc.vertexBufferDataType      = rtgi::VertexBufferDataType_Float;
            vertexDataSourceDesc.offset                    = VertexLayoutByteOffset_TextureSpaceTangent;
            vertexDataSourceDesc.numCoordinatesPerSemantic = VertexLayoutNumCoordinates_TextureSpaceTangent;
            materialStrip->AddVertexDataSourceDesc( SEMANTIC_TEXTURE_SPACE_TANGENT, vertexDataSourceDesc );

            vertexDataSourceDesc.vertexBuffer              = vertexBuffer;
            vertexDataSourceDesc.numVertices               = bufferDesc.numVertices;
            vertexDataSourceDesc.stride                    = sizeof( Vertex );
            vertexDataSourceDesc.vertexBufferDataType      = rtgi::VertexBufferDataType_Float;
            vertexDataSourceDesc.offset                    = VertexLayoutByteOffset_TextureSpaceBinormal;
            vertexDataSourceDesc.numCoordinatesPerSemantic = VertexLayoutNumCoordinates_TextureSpaceBinormal;
            materialStrip->AddVertexDataSourceDesc( SEMANTIC_TEXTURE_SPACE_BINORMAL, vertexDataSourceDesc );

            vertexDataSourceDesc.vertexBuffer              = vertexBuffer;
            vertexDataSourceDesc.numVertices               = bufferDesc.numVertices;
            vertexDataSourceDesc.stride                    = sizeof( Vertex );
            vertexDataSourceDesc.vertexBufferDataType      = rtgi::VertexBufferDataType_Float;
            vertexDataSourceDesc.offset                    = VertexLayoutByteOffset_DiffuseMapTextureCoordinate;
            vertexDataSourceDesc.numCoordinatesPerSemantic = VertexLayoutNumCoordinates_DiffuseMapTextureCoordinate;
            materialStrip->AddVertexDataSourceDesc( SEMANTIC_TEXCOORD_DIFFUSE_MAP, vertexDataSourceDesc );

            SetFinalized( materialStrip, true );

            materialStrips.Append( materialStrip );

            delete[] bufferDesc.indexBufferRaw;
            bufferDesc.indexBufferRaw = NULL;

            delete[] bufferDesc.vertexBufferRawFormatted;
            bufferDesc.vertexBufferRawFormatted = NULL;
        }

        Renderable* renderable = new Renderable();
        SetDebugName( renderable, geometryName );

        SetFinalized( renderable, false );

        renderable->SetMaterialStrips( materialStrips );

        SetFinalized( renderable, true );

        Insert( inventory, geometryName, renderable );
    }
}

}
