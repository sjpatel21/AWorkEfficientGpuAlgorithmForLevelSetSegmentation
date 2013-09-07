#include "rendering/MaterialStrip.hpp"

#include "core/RefCounted.hpp"

#include "content/Ref.hpp"

#include "rendering/rtgi/IndexBuffer.hpp"
#include "rendering/rtgi/VertexBuffer.hpp"

namespace rendering
{

MaterialStripRenderDesc::MaterialStripRenderDesc() :
primitiveRenderMode( rtgi::PrimitiveRenderMode_Invalid ),
indexingRenderMode ( IndexingRenderMode_Invalid ),
indexBuffer        ( NULL )
{
}

MaterialStrip::MaterialStrip()
{
}

MaterialStrip::~MaterialStrip()
{
    foreach( rtgi::VertexDataSourceDesc vertexDataSourceDesc, mVertexDataSources )
    {
        vertexDataSourceDesc.vertexBuffer->Release();
    }

    mVertexDataSources.Clear();

    AssignRef( mMaterialStripRenderDesc.indexBuffer, NULL );
}

void MaterialStrip::Render() const
{
    Render( mMaterial );
}

void MaterialStrip::Render( content::Ref< Material > overrideMaterial ) const
{
    // if in indexed mode, bind the index buffer
    if ( mMaterialStripRenderDesc.indexingRenderMode == IndexingRenderMode_Indexed )
    {
        mMaterialStripRenderDesc.indexBuffer->Bind();
    }

    // bind the various vertex buffers to the vertex attributes in the material
    overrideMaterial->Bind( mVertexDataSources );

    // for each pass
    while( overrideMaterial->BindPass() )
    {
        if ( mMaterialStripRenderDesc.indexingRenderMode == IndexingRenderMode_Indexed )
        {
            // if in indexed mode, render from the index buffer
            mMaterialStripRenderDesc.indexBuffer->Render( mMaterialStripRenderDesc.primitiveRenderMode );
        }
        else
        {
            // otherwise render from the bound vertex buffers
            core::String positionSemantic = "POSITION";
            Assert( mVertexDataSources.Contains( positionSemantic ) );
            rtgi::VertexBuffer::Render( mMaterialStripRenderDesc.primitiveRenderMode, mVertexDataSources.Value( positionSemantic ).numVertices );
        }
        
        overrideMaterial->UnbindPass();
    }

    overrideMaterial->Unbind();

    // if we're in indexed mode, then bind the index buffer
    if ( mMaterialStripRenderDesc.indexingRenderMode == IndexingRenderMode_Indexed )
    {
        Assert( mMaterialStripRenderDesc.indexBuffer != NULL );

        mMaterialStripRenderDesc.indexBuffer->Unbind();
    }
}

void MaterialStrip::SetMaterialSemantic( const core::String& materialSemantic )
{ 
    Assert( !IsFinalized() );
    
    mMaterialSemantic = materialSemantic;
}

core::String MaterialStrip::GetMaterialSemantic() const
{
    return mMaterialSemantic;
}

void MaterialStrip::SetMaterial( content::Ref< Material > material )
{
    Assert( !IsFinalized() );

    mMaterial = material;
}

void MaterialStrip::SetMaterialStripRenderDesc( MaterialStripRenderDesc materialStripRenderDesc )
{
    Assert( !IsFinalized() );
    
    AssignRef( mMaterialStripRenderDesc.indexBuffer, materialStripRenderDesc.indexBuffer );

    mMaterialStripRenderDesc = materialStripRenderDesc;

    // sanity check
    if ( mMaterialStripRenderDesc.indexingRenderMode == IndexingRenderMode_Indexed )
    {
        Assert( mMaterialStripRenderDesc.indexBuffer != NULL );
    }
    else
    {
        Assert( mMaterialStripRenderDesc.indexBuffer == NULL );        
    }
}

void MaterialStrip::AddVertexDataSourceDesc( core::String semantic, rtgi::VertexDataSourceDesc vertexDataSourceDesc )
{
    Assert( !IsFinalized() );
    Assert( vertexDataSourceDesc.vertexBuffer != NULL );

    mVertexDataSources.Insert( semantic, vertexDataSourceDesc );

    vertexDataSourceDesc.vertexBuffer->AddRef();
}

}