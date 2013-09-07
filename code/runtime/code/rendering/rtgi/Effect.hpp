#ifndef RENDERING_RTGI_EFFECT_HPP
#define RENDERING_RTGI_EFFECT_HPP

#include "core/RefCounted.hpp"
#include "core/String.hpp"

#include "container/Map.hpp"

#include "content/Asset.hpp"

#include "rendering/rtgi/VertexBuffer.hpp"

namespace math
{
    class Matrix44;
    class Vector4;
    class Vector3;
}

namespace rendering
{

namespace rtgi
{

class Texture;
class BufferTexture;

enum EffectParameterType
{
    EffectParameterType_Float,
    EffectParameterType_Vector3,
    EffectParameterType_Vector4,
    EffectParameterType_Matrix44,
    EffectParameterType_Texture
};

class Effect : public content::Asset
{
public:

    //
    // vertex data source bindings
    //
    virtual void BindVertexDataSources( container::Map< core::String, rtgi::VertexDataSourceDesc > vertexDataSources ) const = 0;
    virtual void UnbindVertexDataSources() const = 0;

    //
    // pass management
    //
    virtual bool BindPass()   = 0;
    virtual void UnbindPass() = 0;

    //
    // set up all the effect's parameters between begin and end calls
    //
    virtual void BeginSetEffectParameters() = 0;
    virtual void EndSetEffectParameters()   = 0;

    virtual void SetEffectParameter( const core::String& parameterName, const Texture*        texture       ) = 0;
    virtual void SetEffectParameter( const core::String& parameterName, const BufferTexture*  bufferTexture ) = 0;
    virtual void SetEffectParameter( const core::String& parameterName, const math::Matrix44& matrix44      ) = 0;
    virtual void SetEffectParameter( const core::String& parameterName, const math::Vector4&  vector4       ) = 0;
    virtual void SetEffectParameter( const core::String& parameterName, const math::Vector3&  vector3       ) = 0;
    virtual void SetEffectParameter( const core::String& parameterName, float                 scalar        ) = 0;

    virtual bool ContainsEffectParameter( const core::String& parameterName ) const = 0;

protected:
    Effect() {};
    virtual ~Effect() {};
};

}

}

#endif