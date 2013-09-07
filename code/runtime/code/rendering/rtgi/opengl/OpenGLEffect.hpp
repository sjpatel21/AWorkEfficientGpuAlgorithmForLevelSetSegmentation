#ifndef RENDERING_RTGI_OPENGL_EFFECT_HPP
#define RENDERING_RTGI_OPENGL_EFFECT_HPP

#include <Cg/cg.h>

#include "core/RefCounted.hpp"

#include "container/Map.hpp"

#include "rendering/rtgi/Effect.hpp"
#include "rendering/rtgi/Texture.hpp"

namespace core
{
    class String;
}

namespace rendering
{

namespace rtgi
{

enum EffectParameterSetState
{
    EffectParameterSetState_Set,
    EffectParameterSetState_Unset,
    EffectParameterSetState_Shared

};

struct EffectParameterBindDesc
{
    CGparameter             parameterID;
    CGtype                  type;
    EffectParameterSetState currentSetState;
};

class OpenGLTexture;

class OpenGLEffect : public Effect
{
public:
    OpenGLEffect( const core::String& effectFile );

    virtual ~OpenGLEffect();

    virtual void BindVertexDataSources( container::Map< core::String, rtgi::VertexDataSourceDesc > vertexBufferSemantics ) const;
    virtual void UnbindVertexDataSources() const;

    virtual bool BindPass();
    virtual void UnbindPass();

    virtual void BeginSetEffectParameters();
    virtual void EndSetEffectParameters();

    virtual void SetEffectParameter( const core::String& parameterName, const Texture*        texture       );
    virtual void SetEffectParameter( const core::String& parameterName, const BufferTexture*  bufferTexture );
    virtual void SetEffectParameter( const core::String& parameterName, const math::Matrix44& matrix44      );
    virtual void SetEffectParameter( const core::String& parameterName, const math::Vector4&  vector4       );
    virtual void SetEffectParameter( const core::String& parameterName, const math::Vector3&  vector3       );
    virtual void SetEffectParameter( const core::String& parameterName, float                 scalar        );

    virtual bool ContainsEffectParameter( const core::String& parameterName ) const;

    const container::Map< core::String, EffectParameterBindDesc >& GetEffectParameterBindDescs() const;

private:
    void InitializeEffectParameters();
    void TerminateEffectParameters();

    void InitializeVertexDataSemantics();
    void TerminateVertexDataSemantics();

    unsigned int GetOpenGLVertexBufferDataType( VertexBufferDataType vertexBufferDataType ) const;

    CGeffect    mEffect;
    CGpass      mCurrentPass;
    CGtechnique mTechnique;

    container::Map< core::String, EffectParameterBindDesc >        mParameterBindDescs;
    container::Map< core::String, container::List< CGparameter > > mVertexDataSemantics;
};

}

}

#endif