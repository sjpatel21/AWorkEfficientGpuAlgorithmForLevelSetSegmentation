#ifndef RENDERING_MATERIAL_HPP
#define RENDERING_MATERIAL_HPP

#include "math/Matrix44.hpp"
#include "math/Vector4.hpp"
#include "math/Vector3.hpp"

#include "container/Map.hpp"

#include "content/Asset.hpp"
#include "content/Ref.hpp"

#include "rendering/rtgi/Texture.hpp"
#include "rendering/rtgi/VertexBuffer.hpp"

namespace rendering
{

namespace rtgi
{
    class Effect;
}

class Material : public content::Asset
{
public:
    Material();

    void Bind();
    void Bind( container::Map< core::String, rtgi::VertexDataSourceDesc > vertexBufferSemantics );
    void Unbind();
    
    bool BindPass();
    void UnbindPass();

    void InsertNewMaterialParameter( const core::String& materialParameterName, const rtgi::Texture*  value );
    void InsertNewMaterialParameter( const core::String& materialParameterName, const math::Matrix44& value );
    void InsertNewMaterialParameter( const core::String& materialParameterName, const math::Vector4&  value );
    void InsertNewMaterialParameter( const core::String& materialParameterName, const math::Vector3&  value );
    void InsertNewMaterialParameter( const core::String& materialParameterName, float                 value );

    void SetMaterialParameter( const core::String& materialParameterName, const rtgi::Texture*  value );
    void SetMaterialParameter( const core::String& materialParameterName, const math::Matrix44& value );
    void SetMaterialParameter( const core::String& materialParameterName, const math::Vector4&  value );
    void SetMaterialParameter( const core::String& materialParameterName, const math::Vector3&  value );
    void SetMaterialParameter( const core::String& materialParameterName, float                 value );

    void SetName( const core::String& name );
    void SetEffect( content::Ref< rtgi::Effect > effect );

    static core::String GetFullyQualifiedMaterialParameterName( const core::String& materialName, const core::String& materialParameterName );

protected:
    enum MaterialParameterType
    {
        MaterialParameterType_Float,
        MaterialParameterType_Vector3,
        MaterialParameterType_Vector4,
        MaterialParameterType_Matrix44,
        MaterialParameterType_Texture
    };

    struct MaterialParameter
    {
        MaterialParameterType     type;
        math::Matrix44            dataMatrix44;
        math::Vector4             dataVector4;
        math::Vector3             dataVector3;
        float                     dataFloat;
        const rtgi::Texture*      dataTexture;
    };

    virtual ~Material();

    virtual void SetEffectParameters();

    container::Map< core::String, MaterialParameter > mParameters;
    content::Ref< rtgi::Effect >                      mEffect;
    core::String                                      mName;
};

}

#endif