#ifndef RENDERING_RTGI_SHADER_PROGRAM_HPP
#define RENDERING_RTGI_SHADER_PROGRAM_HPP

#include "core/RefCounted.hpp"
#include "core/String.hpp"

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

class  Texture;
struct TextureSamplerStateDesc;

struct ShaderProgramDesc
{
    core::String vertexProgramFile;
    core::String geometryProgramFile;
    core::String fragmentProgramFile;
    core::String vertexProgramEntryPoint;
    core::String geometryProgramEntryPoint;
    core::String fragmentProgramEntryPoint;
};

class ShaderProgram : public core::RefCounted
{
public:
    virtual void BindVertexDataSources( container::Map< core::String, VertexDataSourceDesc > vertexDataSources ) const = 0;
    virtual void UnbindVertexDataSources() const = 0;

    virtual void Bind() const = 0;
    virtual void Unbind() const = 0;

    virtual void BeginSetShaderParameters() = 0;
    virtual void EndSetShaderParameters()   = 0;

    virtual void SetShaderParameter( const core::String& parameterName, const Texture* texture, const TextureSamplerStateDesc& textureSamplerStateDesc ) = 0;
    virtual void SetShaderParameter( const core::String& parameterName, const BufferTexture*  bufferTexture ) = 0;
    virtual void SetShaderParameter( const core::String& parameterName, const math::Matrix44& matrix ) = 0;
    virtual void SetShaderParameter( const core::String& parameterName, const math::Vector4&  vector ) = 0;
    virtual void SetShaderParameter( const core::String& parameterName, const math::Vector3&  vector ) = 0;
    virtual void SetShaderParameter( const core::String& parameterName, float                 scalar ) = 0;

protected:
    ShaderProgram() {};
    virtual ~ShaderProgram() {};
};

}

}

#endif