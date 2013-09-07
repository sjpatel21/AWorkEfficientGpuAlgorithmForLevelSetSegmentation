#ifndef RENDERING_MATERIAL_STRIP_HPP
#define RENDERING_MATERIAL_STRIP_HPP

#include "core/String.hpp"

#include "content/Asset.hpp"
#include "content/Ref.hpp"

#include "rendering/Material.hpp"

namespace rendering
{

class Renderable;

namespace rtgi
{
    class IndexBuffer;
    class VertexBuffer;
}

enum IndexingRenderMode
{
    IndexingRenderMode_Indexed,
    IndexingRenderMode_NonIndexed,

    IndexingRenderMode_Invalid
};

struct MaterialStripRenderDesc
{
    rtgi::PrimitiveRenderMode primitiveRenderMode;
    IndexingRenderMode        indexingRenderMode;
    rtgi::IndexBuffer*        indexBuffer;

    MaterialStripRenderDesc();
};

class MaterialStrip : public content::Asset
{
public:
    MaterialStrip();

    void Render() const;
    void Render( content::Ref< Material > overrideMaterial ) const;

    void         SetMaterialSemantic( const core::String& materialSemantic );
    core::String GetMaterialSemantic() const;

    void SetMaterial                ( content::Ref< Material > material );
    void SetMaterialStripRenderDesc ( MaterialStripRenderDesc materialStripRenderDesc );
    void AddVertexDataSourceDesc    ( core::String semantic, rtgi::VertexDataSourceDesc vertexDataSourceDesc );

protected:
    virtual ~MaterialStrip();
    
    core::String                                               mMaterialSemantic;
    content::Ref< Material >                                   mMaterial;
    MaterialStripRenderDesc                                    mMaterialStripRenderDesc;
    container::Map< core::String, rtgi::VertexDataSourceDesc > mVertexDataSources;
};

}

#endif