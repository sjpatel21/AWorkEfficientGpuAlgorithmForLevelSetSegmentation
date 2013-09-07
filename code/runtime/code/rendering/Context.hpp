#ifndef RENDERING_CONTEXT_HPP
#define RENDERING_CONTEXT_HPP

#if defined(PLATFORM_WIN32)
#define NOMINMAX
#include <windows.h>
typedef HWND WINDOW_HANDLE;
#elif defined(PLATFORM_OSX)
#include <Carbon/Carbon.h>
typedef HIViewRef WINDOW_HANDLE;
#endif

#include "math/Vector3.hpp"
#include "math/Vector4.hpp"
#include "math/Matrix44.hpp"

#include "content/Ref.hpp"

#include "rendering/rtgi/Texture.hpp"
#include "rendering/rtgi/Effect.hpp"
#include "rendering/Material.hpp"

namespace math
{

class Matrix44;

}

namespace rendering
{

class RenderStrategy;
class Camera;
class Scene;

namespace rtgi
{

class Effect;
class TextFont;

}

class Context
{
public:

    struct EffectParameter
    {
        rtgi::EffectParameterType type;
        math::Matrix44            dataMatrix44;
        math::Vector4             dataVector4;
        math::Vector3             dataVector3;
        float                     dataFloat;
        const rtgi::Texture*      dataTexture;
    };

    // init/terminate
    static void                   Initialize( WINDOW_HANDLE windowHandle );  
    static void                   Terminate();

    // update/render
    static void                   Update( double timeDeltaSeconds );
    static void                   Render();

    // context state
    static RenderStrategy*        GetCurrentRenderStrategy();
    static void                   SetCurrentRenderStrategy( RenderStrategy* renderStrategy );
                   
    static content::Ref< Camera > GetCurrentCamera();
    static void                   SetCurrentCamera( content::Ref< Camera > camera );

    static content::Ref< Scene >  GetCurrentScene();
    static void                   SetCurrentScene( content::Ref< Scene > scene );

    static void                   SetCurrentViewport( int  width, int  height );
    static void                   GetCurrentViewport( int& width, int& height );
    static void                   GetInitialViewport( int& width, int& height );

    static void                   SetCurrentTransformMatrix( const math::Matrix44& transform );
    static void                   GetCurrentTransformMatrix( math::Matrix44& transform );

    // get debug/default content
    static rtgi::Texture*               GetDebugTexture();
    static rtgi::TextFont*              GetDebugTextFont();
    static content::Ref< Material >     GetDebugMaterial();
    static content::Ref< Camera >       GetDebugCamera();
    static content::Ref< rtgi::Effect > GetPhongEffect();

    // context-level effect parameters
    static container::List< core::String >             GetEffectParameterNames();
    static rtgi::EffectParameterType                   GetEffectParameterType( const core::String& effectParameterName );

    template< typename T > inline static T                    GetEffectParameter( const core::String& effectParameterName );
#if defined(PLATFORM_WIN32)
    template <>            static float                GetEffectParameter( const core::String& effectParameterName );
    template <>            static math::Vector3        GetEffectParameter( const core::String& effectParameterName );
    template <>            static math::Vector4        GetEffectParameter( const core::String& effectParameterName );
    template <>            static math::Matrix44       GetEffectParameter( const core::String& effectParameterName );
    template <>            static const rtgi::Texture* GetEffectParameter( const core::String& effectParameterName );
#endif

private:
    static bool IsInitialized();

    static void InsertNewEffectParameter( const core::String& effectParameterName, const rtgi::Texture*  value );
    static void InsertNewEffectParameter( const core::String& effectParameterName, const math::Matrix44& value );
    static void InsertNewEffectParameter( const core::String& effectParameterName, const math::Vector4&  value );
    static void InsertNewEffectParameter( const core::String& effectParameterName, const math::Vector3&  value );
    static void InsertNewEffectParameter( const core::String& effectParameterName, float                 value );

    static void SetEffectParameter( const core::String& effectParameterName, const rtgi::Texture*  value );
    static void SetEffectParameter( const core::String& effectParameterName, const math::Matrix44& value );
    static void SetEffectParameter( const core::String& effectParameterName, const math::Vector4&  value );
    static void SetEffectParameter( const core::String& effectParameterName, const math::Vector3&  value );
    static void SetEffectParameter( const core::String& effectParameterName, float                 value );

    static container::Map< core::String, EffectParameter > sEffectParameters;

};


//
// All valid template specializations are defined below, if not specialized then assert.
//
template< typename T > inline T Context::GetEffectParameter( const core::String& effectParameterName )
{
    Assert( false );

    T t;
    return t;
};

template <> inline float Context::GetEffectParameter( const core::String& effectParameterName )
{
    Assert( sEffectParameters.Contains( effectParameterName ) );

    EffectParameter existingValue;

    existingValue = sEffectParameters.Value( effectParameterName );

    Assert( existingValue.type == rtgi::EffectParameterType_Float );

    return existingValue.dataFloat;
};

template <> inline math::Vector3 Context::GetEffectParameter( const core::String& effectParameterName )
{
    Assert( sEffectParameters.Contains( effectParameterName ) );

    EffectParameter existingValue;

    existingValue = sEffectParameters.Value( effectParameterName );

    Assert( existingValue.type == rtgi::EffectParameterType_Vector3 );

    return existingValue.dataVector3;
};

template <> inline math::Vector4 Context::GetEffectParameter( const core::String& effectParameterName )
{
    Assert( sEffectParameters.Contains( effectParameterName ) );

    EffectParameter existingValue;

    existingValue = sEffectParameters.Value( effectParameterName );

    Assert( existingValue.type == rtgi::EffectParameterType_Vector4 );

    return existingValue.dataVector4;
};

template <> inline math::Matrix44 Context::GetEffectParameter( const core::String& effectParameterName )
{
    Assert( sEffectParameters.Contains( effectParameterName ) );

    EffectParameter existingValue;

    existingValue = sEffectParameters.Value( effectParameterName );

    Assert( existingValue.type == rtgi::EffectParameterType_Matrix44 );

    return existingValue.dataMatrix44;
};

template <> inline const rtgi::Texture* Context::GetEffectParameter( const core::String& effectParameterName )
{
    Assert( sEffectParameters.Contains( effectParameterName ) );

    EffectParameter existingValue;

    existingValue = sEffectParameters.Value( effectParameterName );

    Assert( existingValue.type == rtgi::EffectParameterType_Texture );

    return existingValue.dataTexture;
};

}

#endif