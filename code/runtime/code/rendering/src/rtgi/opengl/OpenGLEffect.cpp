#include "rendering/rtgi/opengl/OpenGLEffect.hpp"

#include <boost/filesystem.hpp>

#if defined PLATFORM_WIN32

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

#include "math/Matrix44.hpp"

#include "container/List.hpp"

#include "rendering/rtgi/RTGI.hpp"
#include "rendering/rtgi/Config.hpp"
#include "rendering/rtgi/VertexBuffer.hpp"

#include "rendering/rtgi/opengl/OpenGLShaderProgram.hpp"
#include "rendering/rtgi/opengl/OpenGLTexture.hpp"
#include "rendering/rtgi/opengl/Extensions.hpp"

#define BUFFER_OFFSET( i ) ( ( char* )NULL + ( i ) )

namespace rendering
{

namespace rtgi
{

// HACK
extern CGcontext gCGContext;

GLenum GetOpenGLVertexBufferDataType( VertexBufferDataType vertexBufferDataType );

OpenGLEffect::OpenGLEffect( const core::String& effectFile ) :
mEffect     ( NULL ),
mCurrentPass( NULL )
{
    boost::filesystem::path effectFilePath( effectFile.ToStdString() );
    core::String            absoluteEffectFile( boost::filesystem::system_complete( effectFilePath ).native_file_string() );
    core::String            cgCompilerArguments;
    
#if defined(PLATFORM_WIN32)
    cgCompilerArguments = core::String( "-DPLATFORM_WIN32" );
#elif defined(PLATFORM_OSX)
    cgCompilerArguments = core::String( "-DPLATFORM_OSX" );
#endif
    
    const char* cgCompilerArgumentArray[] = { cgCompilerArguments.ToAscii(), NULL };
    
    mEffect = cgCreateEffectFromFile( gCGContext, absoluteEffectFile.ToAscii(), cgCompilerArgumentArray );
    CheckErrors();

    mTechnique = cgGetFirstTechnique( mEffect );

    while ( mTechnique != NULL && cgValidateTechnique( mTechnique ) != CG_TRUE )
    {
        mTechnique = cgGetNextTechnique( mTechnique );
    }

#ifndef CAPS_BASIC_EXTENSIONS_ONLY
    Assert( cgValidateTechnique( mTechnique ) == CG_TRUE );
    Assert( mTechnique != NULL );
#endif

    InitializeEffectParameters();
    InitializeVertexDataSemantics();

    CheckErrors();
}

OpenGLEffect::~OpenGLEffect()
{
    TerminateVertexDataSemantics();
    TerminateEffectParameters();

    cgDestroyEffect( mEffect );
    mEffect = NULL;
}

void OpenGLEffect::BindVertexDataSources( container::Map< core::String, rtgi::VertexDataSourceDesc > vertexDataSources ) const
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

void OpenGLEffect::UnbindVertexDataSources() const
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

bool sSettingUpEffectParameters = false;
bool sPassBound                 = false;

bool OpenGLEffect::BindPass()
{
    Assert( !sSettingUpEffectParameters );
    Assert( !sPassBound );

    sPassBound = true;

    //
    // get the next pass (or the first pass)
    //
    if ( mCurrentPass == NULL )
    {
        mCurrentPass = cgGetFirstPass( mTechnique );
    }
    else
    {
        mCurrentPass = cgGetNextPass( mCurrentPass );
    }

    //
    // if the current pass is null, that means we've gone through
    // all the passes, so return false.  otherwise set the pass state.
    //
    if ( mCurrentPass == NULL )
    {
        sPassBound = false;
        return false;
    }
    else
    {
        cgSetPassState( mCurrentPass );

        CheckErrors();

        return true;
    }
}

void OpenGLEffect::UnbindPass()
{
    Assert( !sSettingUpEffectParameters );
    Assert( sPassBound );

    cgResetPassState( mCurrentPass );

    CheckErrors();

    sPassBound = false;
}

void OpenGLEffect::BeginSetEffectParameters()
{
    Assert( !sSettingUpEffectParameters );
    Assert( !sPassBound );

    sSettingUpEffectParameters = true;

    foreach_key_value ( core::String effectParameterName, EffectParameterBindDesc effectParameterBindDesc, mParameterBindDescs )
    {
        if ( effectParameterBindDesc.currentSetState != EffectParameterSetState_Shared )
        {
            effectParameterBindDesc.currentSetState = EffectParameterSetState_Unset;
        }

        mParameterBindDescs.Insert( effectParameterName, effectParameterBindDesc );
    }
}

void OpenGLEffect::EndSetEffectParameters()
{
    Assert( sSettingUpEffectParameters );
    Assert( !sPassBound );

    // check that all expected parameters are uploaded...
    foreach ( EffectParameterBindDesc effectParameterBindDesc, mParameterBindDescs )
    {
        //
        // we don't check shared parameters, since they might be updated
        // at different frequencies to the rest of the shader parameters
        //
        Assert( effectParameterBindDesc.currentSetState != EffectParameterSetState_Unset );
    }

    sSettingUpEffectParameters = false;
}

void OpenGLEffect::SetEffectParameter( const core::String& parameterName, const Texture* texture )
{
    Assert( sSettingUpEffectParameters );
    Assert( !IsSharedShaderParameter( parameterName ) );
    Assert( mParameterBindDescs.Contains( parameterName ) );

    const OpenGLTexture* openGLTexture = dynamic_cast< const OpenGLTexture* >( texture );
    Assert( openGLTexture != NULL );

    EffectParameterBindDesc effectParameterBindDesc = mParameterBindDescs.Value( parameterName );

    switch ( openGLTexture->GetTextureDataDesc().dimensions )
    {
    case TextureDimensions_1D:
        Assert( effectParameterBindDesc.type == CG_SAMPLER1D );
        break;

    case TextureDimensions_2D:
        Assert( effectParameterBindDesc.type == CG_SAMPLER2D );
        break;

    case TextureDimensions_3D:
        Assert( effectParameterBindDesc.type == CG_SAMPLER3D );
        break;
    }

    Assert( effectParameterBindDesc.currentSetState != EffectParameterSetState_Shared );

    cgGLSetTextureParameter( effectParameterBindDesc.parameterID, openGLTexture->GetOpenGLTextureID() );
    cgGLSetupSampler( effectParameterBindDesc.parameterID, openGLTexture->GetOpenGLTextureID() );

    effectParameterBindDesc.currentSetState = EffectParameterSetState_Set;

    mParameterBindDescs.Insert( parameterName, effectParameterBindDesc );

    CheckErrors();
}

void OpenGLEffect::SetEffectParameter( const core::String& parameterName, const BufferTexture* bufferTexture )
{
    Assert( sSettingUpEffectParameters );
    Assert( !IsSharedShaderParameter( parameterName ) );
    Assert( mParameterBindDescs.Contains( parameterName ) );

    const OpenGLBufferTexture* openGLBufferTexture = dynamic_cast< const OpenGLBufferTexture* >( bufferTexture );
    Assert( openGLBufferTexture != NULL );

    EffectParameterBindDesc effectParameterBindDesc = mParameterBindDescs.Value( parameterName );

    Assert( effectParameterBindDesc.type            == CG_SAMPLERBUF );
    Assert( effectParameterBindDesc.currentSetState != ShaderParameterSetState_Shared );

    openGLBufferTexture->Bind();

    cgGLSetTextureParameter( effectParameterBindDesc.parameterID, openGLBufferTexture->GetOpenGLTextureID() );
    cgGLSetupSampler( effectParameterBindDesc.parameterID, openGLBufferTexture->GetOpenGLTextureID() );

    effectParameterBindDesc.currentSetState = EffectParameterSetState_Set;

    mParameterBindDescs.Insert( parameterName, effectParameterBindDesc );

    CheckErrors();
}

void OpenGLEffect::SetEffectParameter( const core::String& parameterName, const math::Matrix44& matrix )
{
    Assert( sSettingUpEffectParameters );
    Assert( !IsSharedShaderParameter( parameterName ) );
    Assert( mParameterBindDescs.Contains( parameterName ) );

    EffectParameterBindDesc effectParameterBindDesc = mParameterBindDescs.Value( parameterName );

    Assert( effectParameterBindDesc.type            == CG_FLOAT4x4 );
    Assert( effectParameterBindDesc.currentSetState != EffectParameterSetState_Shared );

    cgGLSetMatrixParameterfr( effectParameterBindDesc.parameterID, matrix.Ref() );

    effectParameterBindDesc.currentSetState = EffectParameterSetState_Set;

    mParameterBindDescs.Insert( parameterName, effectParameterBindDesc );

    CheckErrors();
}

void OpenGLEffect::SetEffectParameter( const core::String& parameterName, const math::Vector4& vector )
{
    Assert( sSettingUpEffectParameters );
    Assert( !IsSharedShaderParameter( parameterName ) );
    Assert( mParameterBindDescs.Contains( parameterName ) );

    EffectParameterBindDesc effectParameterBindDesc = mParameterBindDescs.Value( parameterName );

    Assert( effectParameterBindDesc.type            == CG_FLOAT4 );
    Assert( effectParameterBindDesc.currentSetState != EffectParameterSetState_Shared );

    cgGLSetParameter4f( effectParameterBindDesc.parameterID, vector[0], vector[1], vector[2], vector[3] );

    effectParameterBindDesc.currentSetState = EffectParameterSetState_Set;

    mParameterBindDescs.Insert( parameterName, effectParameterBindDesc );

    CheckErrors();
}

void OpenGLEffect::SetEffectParameter( const core::String& parameterName, const math::Vector3& vector )
{
    Assert( sSettingUpEffectParameters );
    Assert( !IsSharedShaderParameter( parameterName ) );
    Assert( mParameterBindDescs.Contains( parameterName ) );

    EffectParameterBindDesc effectParameterBindDesc = mParameterBindDescs.Value( parameterName );

    Assert( effectParameterBindDesc.type            == CG_FLOAT3 );
    Assert( effectParameterBindDesc.currentSetState != EffectParameterSetState_Shared );

    cgGLSetParameter3f( effectParameterBindDesc.parameterID, vector[0], vector[1], vector[2] );

    effectParameterBindDesc.currentSetState = EffectParameterSetState_Set;

    mParameterBindDescs.Insert( parameterName, effectParameterBindDesc );

    CheckErrors();
}

void OpenGLEffect::SetEffectParameter( const core::String& parameterName, float numFloat )
{
    Assert( sSettingUpEffectParameters );
    Assert( !IsSharedShaderParameter( parameterName ) );
    Assert( mParameterBindDescs.Contains( parameterName ) );

    EffectParameterBindDesc effectParameterBindDesc = mParameterBindDescs.Value( parameterName );

    Assert( effectParameterBindDesc.type == CG_FLOAT );
    Assert( effectParameterBindDesc.currentSetState != EffectParameterSetState_Shared );

    cgGLSetParameter1f( effectParameterBindDesc.parameterID, numFloat );

    effectParameterBindDesc.currentSetState = EffectParameterSetState_Set;

    mParameterBindDescs.Insert( parameterName, effectParameterBindDesc );

    CheckErrors();
}

void OpenGLEffect::InitializeEffectParameters()
{
    // get all the parameters for the effect...
    CGparameter parameter = cgGetFirstEffectParameter( mEffect );

    while( parameter != NULL )
    {
        const core::String parameterName = cgGetParameterName( parameter );
        CGtype             parameterType = cgGetParameterType( parameter );

        EffectParameterBindDesc effectParameterBindDesc;

        // get type
        effectParameterBindDesc.type        = cgGetParameterType( parameter );
        effectParameterBindDesc.parameterID = parameter;

        // if shared, then updates are external, so don't try to track
        // updates with begin/end pairs of SetShaderParameter calls
        if ( IsSharedShaderParameter( parameterName ) )
        {
            effectParameterBindDesc.currentSetState = EffectParameterSetState_Shared;
        }
        else
        {
            effectParameterBindDesc.currentSetState = EffectParameterSetState_Unset;
        }

        CheckErrors();

        mParameterBindDescs.Insert( parameterName, effectParameterBindDesc );

        parameter = cgGetNextParameter( parameter );
    }

    ConnectSharedShaderParameters( this );
}

void OpenGLEffect::InitializeVertexDataSemantics()
{
    CGpass pass = cgGetFirstPass( mTechnique );

    // get all the parameters for the shaders used by this effect...
    while( pass != NULL )
    {
        CGprogram vertexProgram = cgGetPassProgram( pass, CG_VERTEX_DOMAIN );

        if ( vertexProgram != NULL )
        {
            // get all the parameters for each shader...
            CGparameter parameter = cgGetFirstLeafParameter( vertexProgram, CG_PROGRAM );

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

        pass = cgGetNextPass( pass );
    }
}

void OpenGLEffect::TerminateVertexDataSemantics()
{
    // no-op
}

void OpenGLEffect::TerminateEffectParameters()
{
    DisconnectSharedShaderParameters( this );
}

bool OpenGLEffect::ContainsEffectParameter( const core::String& parameterName ) const
{
    return mParameterBindDescs.Contains( parameterName );
}

const container::Map< core::String, EffectParameterBindDesc >& OpenGLEffect::GetEffectParameterBindDescs() const
{
    return mParameterBindDescs;
}

unsigned int OpenGLEffect::GetOpenGLVertexBufferDataType( VertexBufferDataType vertexBufferDataType ) const
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