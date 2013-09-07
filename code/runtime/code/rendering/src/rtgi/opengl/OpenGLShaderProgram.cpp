#include "rendering/rtgi/opengl/OpenGLShaderProgram.hpp"

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

#include <stdlib.h>
#include <stdio.h>

#include <boost/filesystem.hpp>

#include "core/Assert.hpp"
#include "core/String.hpp"
#include "core/Printf.hpp"

#include "math/Matrix44.hpp"

#include "rendering/rtgi/RTGI.hpp"

#include "rendering/rtgi/opengl/Extensions.hpp"
#include "rendering/rtgi/opengl/OpenGLTexture.hpp"

namespace rendering
{

namespace rtgi
{

extern CGcontext gCGContext;

OpenGLShaderProgram::OpenGLShaderProgram( const ShaderProgramDesc& shaderProgramDesc ) :
mVertexProgram  ( NULL ),
mGeometryProgram( NULL ),
mFragmentProgram( NULL )
{
    boost::filesystem::path vertexShaderFilePath  ( shaderProgramDesc.vertexProgramFile.ToStdString() );
    boost::filesystem::path geometryShaderFilePath( shaderProgramDesc.geometryProgramFile.ToStdString() );
    boost::filesystem::path fragmentShaderFilePath( shaderProgramDesc.fragmentProgramFile.ToStdString() );

    ReleaseAssert( boost::filesystem::exists( vertexShaderFilePath   ) );
    ReleaseAssert( boost::filesystem::exists( geometryShaderFilePath ) );
    ReleaseAssert( boost::filesystem::exists( fragmentShaderFilePath ) );

    core::String absoluteVertexShaderFile  ( boost::filesystem::system_complete( vertexShaderFilePath   ).native_file_string() );
    core::String absoluteGeometryShaderFile( boost::filesystem::system_complete( geometryShaderFilePath ).native_file_string() );
    core::String absoluteFragmentShaderFile( boost::filesystem::system_complete( fragmentShaderFilePath ).native_file_string() );

    // specify compiler flags
    core::String includePath        = core::String( "-I" ) + core::String( boost::filesystem::current_path().native_file_string() );
    const char*  includePathAscii   = includePath.ToAscii();

    const char*  compilerArgs[ 3 ];
    compilerArgs[ 0 ] = includePathAscii;

#ifdef BUILD_DEBUG
    compilerArgs[ 1 ] = "-debug";
#endif

#ifdef BUILD_RELEASE
    compilerArgs[ 1 ] = NULL;
#endif

    compilerArgs[ 2 ] = NULL;

    // vertex
    mVertexProfile = cgGLGetLatestProfile( CG_GL_VERTEX );
    cgGLSetOptimalOptions( mVertexProfile );

    mVertexProgram = cgCreateProgramFromFile(
        gCGContext,                                          // cg context
        CG_SOURCE,                                           // source or pre-compiled
        absoluteVertexShaderFile.ToAscii(),                  // filename
        mVertexProfile,                                      // profile
        shaderProgramDesc.vertexProgramEntryPoint.ToAscii(), // entry point
        compilerArgs );                                      // compiler arguments

    cgGLLoadProgram( mVertexProgram );
    CheckErrors();

    
    // geometry
#if defined(PLATFORM_WIN32)
    // Use CG to load geometry programs on windows
    mGeometryProfile = cgGLGetLatestProfile( CG_GL_GEOMETRY );
    cgGLSetOptimalOptions( mGeometryProfile );

    mGeometryProgram = cgCreateProgramFromFile(
        gCGContext,                                            // cg context
        CG_SOURCE,                                             // source or pre-compiled
        absoluteGeometryShaderFile.ToAscii(),                  // filename
        mGeometryProfile,                                      // profile
        shaderProgramDesc.geometryProgramEntryPoint.ToAscii(), // entry point
        compilerArgs );                                        // compiler arguments

    cgGLLoadProgram( mGeometryProgram );
    CheckErrors();
#elif defined(PLATFORM_OSX)
    // Don't use CG to load geometry programs on OSX
    // For now, don't even load geometry programs on OSX
#endif

    // fragment
    mFragmentProfile = cgGLGetLatestProfile( CG_GL_FRAGMENT );
    cgGLSetOptimalOptions( mFragmentProfile );

    mFragmentProgram = cgCreateProgramFromFile(
        gCGContext,                                            // cg context
        CG_SOURCE,                                             // source or pre-compiled
        absoluteFragmentShaderFile.ToAscii(),                  // filename
        mFragmentProfile,                                      // profile
        shaderProgramDesc.fragmentProgramEntryPoint.ToAscii(), // entry point
        compilerArgs );                                        // compiler arguments

    cgGLLoadProgram( mFragmentProgram );
    CheckErrors();

    InitializeShaderParameters( mVertexProgram );
#if defined(PLATFORM_WIN32)
    InitializeShaderParameters( mGeometryProgram );
#elif defined(PLATFORM_OSX)
    // Currently, we aren't loading geometry programs on OSX so
    // we don't set parameters either.
#endif
    InitializeShaderParameters( mFragmentProgram );

    InitializeVertexDataSemantics();

    ConnectSharedShaderParameters( this );

    CheckErrors();
}

OpenGLShaderProgram::~OpenGLShaderProgram()
{
    DisconnectSharedShaderParameters( this );

    TerminateVertexDataSemantics();

    TerminateShaderParameters( mFragmentProgram );
    TerminateShaderParameters( mGeometryProgram );
    TerminateShaderParameters( mVertexProgram );

    cgDestroyProgram( mFragmentProgram );
    mFragmentProgram = NULL;

#if defined(PLATFORM_WIN32)
    cgDestroyProgram( mGeometryProgram );
    mGeometryProgram = NULL;
#elif defined(PLATFORM_OSX)
    // CG geometry shaders are not currently supported on OSX
#endif

    cgDestroyProgram( mVertexProgram );
    mVertexProgram = NULL;

    CheckErrors();
}

void OpenGLShaderProgram::BindVertexDataSources( container::Map< core::String, rtgi::VertexDataSourceDesc > vertexDataSources ) const
{
    foreach_key_value( core::String semantic, container::List< CGparameter > parameters, mVertexDataSemantics )
    {
        Assert( vertexDataSources.Contains( semantic ) );

        rtgi::VertexDataSourceDesc vertexDataSourceDesc       = vertexDataSources.Value( semantic );
        GLenum                     openGLVertexBufferDataType = GetOpenGLVertexBufferDataType( vertexDataSourceDesc.vertexBufferDataType );

        vertexDataSourceDesc.vertexBuffer->Bind();

        foreach( CGparameter parameter, parameters )
        {
            Assert( cgGetParameterVariability( parameter ) == CG_VARYING );
            Assert( cgGetParameterDirection( parameter ) == CG_IN );
            Assert( cgGetProgramDomain( cgGetParameterProgram( parameter ) ) == CG_VERTEX_DOMAIN );

            cgGLSetParameterPointer(
                parameter,
                vertexDataSourceDesc.numCoordinatesPerSemantic,
                openGLVertexBufferDataType,
                vertexDataSourceDesc.stride,
                BUFFER_OFFSET( vertexDataSourceDesc.offset ) );

            cgGLEnableClientState( parameter );

            CheckErrors();
        }
    }
}

void OpenGLShaderProgram::UnbindVertexDataSources() const
{
    foreach( container::List< CGparameter > parameters, mVertexDataSemantics )
    {
        foreach( CGparameter parameter, parameters )
        {
            cgGLDisableClientState( parameter );

            CheckErrors();
        }
    }
}

bool sSettingUpShaderParameters = false;
bool sBound                     = false;

void OpenGLShaderProgram::Bind() const
{
    Assert( !sSettingUpShaderParameters );
    Assert( !sBound );

    sBound = true;

    cgGLEnableProfile( mVertexProfile );
    cgGLBindProgram( mVertexProgram );    

#if defined(PLATFORM_WIN32)
    cgGLEnableProfile( mGeometryProfile );
    cgGLBindProgram( mGeometryProgram );    
#elif defined(PLATFORM_OSX)
    // CG Geometry shaders are currently not supported
#endif

    cgGLEnableProfile( mFragmentProfile );
    cgGLBindProgram( mFragmentProgram );

    CheckErrors();
}

void OpenGLShaderProgram::Unbind() const
{
    Assert( !sSettingUpShaderParameters );
    Assert( sBound );

    foreach( ShaderParameterBindDesc shaderParameterBindDesc, mParameterBindDescs )
    {
        if ( shaderParameterBindDesc.type == CG_SAMPLER1D ||
             shaderParameterBindDesc.type == CG_SAMPLER2D ||
             shaderParameterBindDesc.type == CG_SAMPLER3D ||
             shaderParameterBindDesc.type == CG_SAMPLERBUF )
        {
            cgGLDisableTextureParameter( shaderParameterBindDesc.parameterID );
        }
    }

    cgGLUnbindProgram( mFragmentProfile );
    cgGLDisableProfile( mFragmentProfile );

#if defined(PLATFORM_WIN32)
    cgGLUnbindProgram( mGeometryProfile );
    cgGLDisableProfile( mGeometryProfile );
#elif defined(PLATFORM_OSX)
    // CG geometry shaders are currently not supported on OSX
#endif

    cgGLUnbindProgram( mVertexProfile );
    cgGLDisableProfile( mVertexProfile );

    CheckErrors();

    sBound = false;
}


void OpenGLShaderProgram::BeginSetShaderParameters()
{
    Assert( !sSettingUpShaderParameters );
    Assert( !sBound );

    sSettingUpShaderParameters = true;

    foreach_key_value ( core::String shaderParameterName, ShaderParameterBindDesc shaderParameterBindDesc, mParameterBindDescs )
    {
        if ( shaderParameterBindDesc.currentSetState != ShaderParameterSetState_Shared )
        {
            shaderParameterBindDesc.currentSetState = ShaderParameterSetState_Unset;
        }

        mParameterBindDescs.Insert( shaderParameterName, shaderParameterBindDesc );
    }
}

void OpenGLShaderProgram::EndSetShaderParameters()
{
    Assert( sSettingUpShaderParameters );
    Assert( !sBound );

    // check that all expected parameters are uploaded...
    foreach ( ShaderParameterBindDesc shaderParameterBindDesc, mParameterBindDescs )
    {
        //
        // we don't check shared parameters, since they might be updated
        // at different frequencies to the rest of the shader parameters
        //
        Assert( shaderParameterBindDesc.currentSetState != ShaderParameterSetState_Unset );
    }

    sSettingUpShaderParameters = false;
}

void OpenGLShaderProgram::SetShaderParameter( const core::String& parameterName, const Texture* texture, const TextureSamplerStateDesc& textureSamplerStateDesc )
{
    Assert( sSettingUpShaderParameters );
    Assert( !IsSharedShaderParameter( parameterName ) );
    Assert( mParameterBindDescs.Contains( parameterName ) );

    const OpenGLTexture* openGLTexture = dynamic_cast< const OpenGLTexture* >( texture );
    Assert( openGLTexture != NULL );

    ShaderParameterBindDesc shaderParameterBindDesc = mParameterBindDescs.Value( parameterName );

    switch ( openGLTexture->GetTextureDataDesc().dimensions )
    {
        case TextureDimensions_1D:
            Assert( shaderParameterBindDesc.type == CG_SAMPLER1D );
            break;

        case TextureDimensions_2D:
            Assert( shaderParameterBindDesc.type == CG_SAMPLER2D );
            break;

        case TextureDimensions_3D:
            Assert( shaderParameterBindDesc.type == CG_SAMPLER3D );
            break;

        default:
            Assert( 0 );
            break;
    }

    Assert( shaderParameterBindDesc.currentSetState != ShaderParameterSetState_Shared );

    cgGLSetTextureParameter( shaderParameterBindDesc.parameterID, openGLTexture->GetOpenGLTextureID() );
    cgGLEnableTextureParameter( shaderParameterBindDesc.parameterID );

    GLenum openGLTextureUnit = cgGLGetTextureEnum( shaderParameterBindDesc.parameterID );
    GLenum openGLWrapMode    = rtgi::GetOpenGLWrapMode ( textureSamplerStateDesc.textureSamplerWrapMode );
    GLenum openGLMinFilter   = rtgi::GetOpenGLMinFilter( textureSamplerStateDesc.textureSamplerInterpolationMode );
    GLenum openGLMagFilter   = rtgi::GetOpenGLMagFilter( textureSamplerStateDesc.textureSamplerInterpolationMode );

    glClientActiveTextureARB( openGLTextureUnit );
    glBindTexture( openGLTexture->GetOpenGLTextureTarget(), openGLTexture->GetOpenGLTextureID() );
    glTexParameteri( openGLTexture->GetOpenGLTextureTarget(), GL_TEXTURE_MIN_FILTER, openGLMinFilter );
    glTexParameteri( openGLTexture->GetOpenGLTextureTarget(), GL_TEXTURE_MAG_FILTER, openGLMagFilter );
    glTexParameteri( openGLTexture->GetOpenGLTextureTarget(), GL_TEXTURE_WRAP_S,     openGLWrapMode );
    glTexParameteri( openGLTexture->GetOpenGLTextureTarget(), GL_TEXTURE_WRAP_T,     openGLWrapMode );
    glTexParameteri( openGLTexture->GetOpenGLTextureTarget(), GL_TEXTURE_WRAP_R,     openGLWrapMode );

    shaderParameterBindDesc.currentSetState = ShaderParameterSetState_Set;

    mParameterBindDescs.Insert( parameterName, shaderParameterBindDesc );

    CheckErrors();
}

void OpenGLShaderProgram::SetShaderParameter( const core::String& parameterName, const BufferTexture* bufferTexture )
{
    Assert( sSettingUpShaderParameters );
    Assert( !IsSharedShaderParameter( parameterName ) );
    Assert( mParameterBindDescs.Contains( parameterName ) );

    const OpenGLBufferTexture* openGLBufferTexture = dynamic_cast< const OpenGLBufferTexture* >( bufferTexture );
    Assert( openGLBufferTexture != NULL );

    ShaderParameterBindDesc shaderParameterBindDesc = mParameterBindDescs.Value( parameterName );

    Assert( shaderParameterBindDesc.type            == CG_SAMPLERBUF );
    Assert( shaderParameterBindDesc.currentSetState != ShaderParameterSetState_Shared );

    openGLBufferTexture->Bind();

    cgGLSetTextureParameter( shaderParameterBindDesc.parameterID, openGLBufferTexture->GetOpenGLTextureID() );
    cgGLEnableTextureParameter( shaderParameterBindDesc.parameterID );

    shaderParameterBindDesc.currentSetState = ShaderParameterSetState_Set;

    mParameterBindDescs.Insert( parameterName, shaderParameterBindDesc );

    CheckErrors();
}

void OpenGLShaderProgram::SetShaderParameter( const core::String& parameterName, const math::Matrix44& matrix )
{
    Assert( sSettingUpShaderParameters );
    Assert( !IsSharedShaderParameter( parameterName ) );
    Assert( mParameterBindDescs.Contains( parameterName ) );

    ShaderParameterBindDesc shaderParameterBindDesc = mParameterBindDescs.Value( parameterName );

    Assert( shaderParameterBindDesc.type            == CG_FLOAT4x4 );
    Assert( shaderParameterBindDesc.currentSetState != ShaderParameterSetState_Shared );

    cgGLSetMatrixParameterfr( shaderParameterBindDesc.parameterID, matrix.Ref() );

    shaderParameterBindDesc.currentSetState = ShaderParameterSetState_Set;

    mParameterBindDescs.Insert( parameterName, shaderParameterBindDesc );

    CheckErrors();
}

void OpenGLShaderProgram::SetShaderParameter( const core::String& parameterName, const math::Vector4& vector )
{
    Assert( sSettingUpShaderParameters );
    Assert( !IsSharedShaderParameter( parameterName ) );
    Assert( mParameterBindDescs.Contains( parameterName ) );

    ShaderParameterBindDesc shaderParameterBindDesc = mParameterBindDescs.Value( parameterName );

    Assert( shaderParameterBindDesc.type            == CG_FLOAT4 );
    Assert( shaderParameterBindDesc.currentSetState != ShaderParameterSetState_Shared );

    cgGLSetParameter4f( shaderParameterBindDesc.parameterID, vector[0], vector[1], vector[2], vector[3] );

    shaderParameterBindDesc.currentSetState = ShaderParameterSetState_Set;

    mParameterBindDescs.Insert( parameterName, shaderParameterBindDesc );

    CheckErrors();
}

void OpenGLShaderProgram::SetShaderParameter( const core::String& parameterName, const math::Vector3& vector )
{
    Assert( sSettingUpShaderParameters );
    Assert( !IsSharedShaderParameter( parameterName ) );
    Assert( mParameterBindDescs.Contains( parameterName ) );

    ShaderParameterBindDesc shaderParameterBindDesc = mParameterBindDescs.Value( parameterName );

    Assert( shaderParameterBindDesc.type            == CG_FLOAT3 );
    Assert( shaderParameterBindDesc.currentSetState != ShaderParameterSetState_Shared );

    cgGLSetParameter3f( shaderParameterBindDesc.parameterID, vector[0], vector[1], vector[2] );

    shaderParameterBindDesc.currentSetState = ShaderParameterSetState_Set;

    mParameterBindDescs.Insert( parameterName, shaderParameterBindDesc );

    CheckErrors();
}

void OpenGLShaderProgram::SetShaderParameter( const core::String& parameterName, float numFloat )
{
    Assert( sSettingUpShaderParameters );
    Assert( !IsSharedShaderParameter( parameterName ) );
    Assert( mParameterBindDescs.Contains( parameterName ) );

    ShaderParameterBindDesc shaderParameterBindDesc = mParameterBindDescs.Value( parameterName );

    Assert( shaderParameterBindDesc.type == CG_FLOAT );
    Assert( shaderParameterBindDesc.currentSetState != ShaderParameterSetState_Shared );

    cgGLSetParameter1f( shaderParameterBindDesc.parameterID, numFloat );

    shaderParameterBindDesc.currentSetState = ShaderParameterSetState_Set;

    mParameterBindDescs.Insert( parameterName, shaderParameterBindDesc );

    CheckErrors();
}

void OpenGLShaderProgram::InitializeShaderParameters( CGprogram program )
{
    // get all the parameters for each shader...
    CGparameter parameter = cgGetFirstLeafParameter( program, CG_PROGRAM );

    while( parameter != 0 )
    {
        CGenum parameterDirection   = cgGetParameterDirection( parameter );
        CGenum parameterVariablitiy = cgGetParameterVariability( parameter );

        if ( parameterDirection == CG_IN && parameterVariablitiy == CG_UNIFORM )
        {
            const core::String parameterName = cgGetParameterName( parameter );
            CGtype             parameterType = cgGetParameterType( parameter );

            ShaderParameterBindDesc shaderParameterBindDesc;

            // get type
            shaderParameterBindDesc.type        = cgGetParameterType( parameter );
            shaderParameterBindDesc.parameterID = parameter;

            // if shared, then updates are external, so don't try to track
            // updates with begin/end pairs of SetShaderParameter calls
            if ( IsSharedShaderParameter( parameterName ) )
            {
                shaderParameterBindDesc.currentSetState = ShaderParameterSetState_Shared;
            }
            else
            {
                shaderParameterBindDesc.currentSetState = ShaderParameterSetState_Unset;
            }

            mParameterBindDescs.Insert( parameterName, shaderParameterBindDesc );

            CheckErrors();

            parameter = cgGetNextLeafParameter( parameter );
        }
        else
        {
            parameter = cgGetNextLeafParameter( parameter );
        }
    }
}

void OpenGLShaderProgram::TerminateShaderParameters( CGprogram program )
{
}

void OpenGLShaderProgram::InitializeVertexDataSemantics()
{
    // get all the parameters for each shader...
    CGparameter parameter = cgGetFirstLeafParameter( mVertexProgram, CG_PROGRAM );

    while( parameter != 0 )
    {
        CGenum parameterDirection   = cgGetParameterDirection( parameter );
        CGenum parameterVariablitiy = cgGetParameterVariability( parameter );

        if ( parameterDirection == CG_IN && parameterVariablitiy == CG_VARYING )
        {
            core::String parameterSemantic = cgGetParameterSemantic( parameter );

            if ( !mVertexDataSemantics.Contains( parameterSemantic ) )
            {
                container::List< CGparameter > parameters;
                parameters.Append( parameter );
                mVertexDataSemantics.Insert( parameterSemantic, parameters );
            }
            else
            {
                container::List< CGparameter > parameters = mVertexDataSemantics.Value( parameterSemantic );
                parameters.Append( parameter );
                mVertexDataSemantics.Insert( parameterSemantic, parameters );
            }
        }

        parameter = cgGetNextLeafParameter( parameter );
    }
}

void OpenGLShaderProgram::TerminateVertexDataSemantics()
{
    // no-op
}

const container::Map< core::String, ShaderParameterBindDesc >& OpenGLShaderProgram::GetShaderParameterBindDescs() const
{
    return mParameterBindDescs;
}

unsigned int OpenGLShaderProgram::GetOpenGLVertexBufferDataType( VertexBufferDataType vertexBufferDataType ) const
{
    switch( vertexBufferDataType )
    {
    case VertexBufferDataType_Float:
        return GL_FLOAT;

    case VertexBufferDataType_Int:
        return GL_INT;

    default:
        Assert( 0 );
        return GL_NONE;
    }
}

}

}
