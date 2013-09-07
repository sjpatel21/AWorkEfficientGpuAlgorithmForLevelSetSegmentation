#ifndef RENDERING_RTGI_OPENGL_SHADER_PROGRAM_HPP
#define RENDERING_RTGI_OPENGL_SHADER_PROGRAM_HPP

#if defined(PLATFORM_WIN32)

#define NOMINMAX
#include <windows.h>

#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glext.h>

#elif defined(PLATFORM_OSX)

#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#include <OpenGL/glext.h>

#endif

#include <Cg/cg.h>
#include <Cg/cgGL.h>

#include "core/String.hpp"
#include "core/RefCounted.hpp"

#include "container/Map.hpp"

#include "rendering/rtgi/Texture.hpp"
#include "rendering/rtgi/ShaderProgram.hpp"

#define BUFFER_OFFSET( i ) ( ( char* )NULL + ( i ) )

namespace rendering
{

namespace rtgi
{

enum ShaderParameterSetState
{
    ShaderParameterSetState_Set,
    ShaderParameterSetState_Unset,
    ShaderParameterSetState_Shared
};

struct ShaderParameterBindDesc
{
    CGparameter             parameterID;
    CGtype                  type;
    ShaderParameterSetState currentSetState;
};

class OpenGLShaderProgram : public ShaderProgram
{

public:
    OpenGLShaderProgram( const ShaderProgramDesc& shaderProgramDesc );

    virtual ~OpenGLShaderProgram();

    virtual void BindVertexDataSources( container::Map< core::String, rtgi::VertexDataSourceDesc > vertexDataSources ) const;
    virtual void UnbindVertexDataSources() const;

    virtual void Bind() const;
    virtual void Unbind() const;

    virtual void BeginSetShaderParameters();
    virtual void EndSetShaderParameters();

    virtual void SetShaderParameter( const core::String& parameterName, const Texture*        texture,       const TextureSamplerStateDesc& textureSamplerStateDesc );
    virtual void SetShaderParameter( const core::String& parameterName, const BufferTexture*  bufferTexture );
    virtual void SetShaderParameter( const core::String& parameterName, const math::Matrix44& matrix );
    virtual void SetShaderParameter( const core::String& parameterName, const math::Vector4&  vector );
    virtual void SetShaderParameter( const core::String& parameterName, const math::Vector3&  vector );
    virtual void SetShaderParameter( const core::String& parameterName, float                 scalar );

    const container::Map< core::String, ShaderParameterBindDesc >& GetShaderParameterBindDescs() const;

private:

    void InitializeShaderParameters( CGprogram program );
    void TerminateShaderParameters( CGprogram program );

    void InitializeVertexDataSemantics();
    void TerminateVertexDataSemantics();

    unsigned int GetOpenGLVertexBufferDataType( VertexBufferDataType vertexBufferDataType ) const;

    CGprogram   mVertexProgram;
    CGprogram   mGeometryProgram;
    CGprogram   mFragmentProgram;
    CGprofile   mVertexProfile;
    CGprofile   mGeometryProfile;
    CGprofile   mFragmentProfile;

    container::Map< core::String, ShaderParameterBindDesc >        mParameterBindDescs;
    container::Map< core::String, container::List< CGparameter > > mVertexDataSemantics;

};

}

}

#endif