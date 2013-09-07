#include "rendering/Context.hpp"

#if defined PLATFORM_WIN32
#define NOMINMAX
#include <windows.h>
#endif

#include <BMPLoader.h>

#include "core/Assert.hpp"
#include "core/RefCounted.hpp"

#include "content/LoadManager.hpp"
#include "content/Ref.hpp"
#include "content/Inventory.hpp"
#include "content/LoadManager.hpp"
#include "content/ParameterManager.hpp"

#include "rendering/rtgi/RTGI.hpp"
#include "rendering/rtgi/Effect.hpp"
#include "rendering/rtgi/TextFont.hpp"

#include "rendering/Camera.hpp"
#include "rendering/Scene.hpp"
#include "rendering/loaders/TextureLoader.hpp"
#include "rendering/loaders/EffectLoader.hpp"
#include "rendering/loaders/GeometryLoader.hpp"
#include "rendering/loaders/MaterialLoader.hpp"
#include "rendering/loaders/SceneLoader.hpp"
#include "rendering/renderstrategies/RenderStrategy.hpp"

namespace rendering
{

static content::Inventory*          sInventory             = NULL;
static RenderStrategy*              sCurrentRenderStrategy = NULL;
static rtgi::TextFont*              sDebugTextFont         = NULL;
static rtgi::Texture*               sDebugTexture          = NULL;
static bool                         sIsInitialized         = false;                      
static int                          sInitialViewportWidth  = 0;
static int                          sInitialViewportHeight = 0;

static content::Ref< Scene >        sScene;
static content::Ref< Camera >       sCamera;
static content::Ref< Camera >       sDebugCamera;
static content::Ref< Material >     sDebugMaterial;
static content::Ref< rtgi::Effect > sPhongEffect;

static math::Matrix44               sTransform;
static math::Matrix44               sView;
static math::Matrix44               sProjection;

container::Map< core::String, Context::EffectParameter > Context::sEffectParameters;

void Context::Initialize( WINDOW_HANDLE windowHandle )
{
    Assert( !sIsInitialized );

    // initialize rtgi
    rtgi::Initialize( windowHandle );

    rtgi::BeginRender();
    rtgi::ClearColorBuffer( rtgi::ColorRGBA( 0, 0, 0, 1 ) );
    rtgi::ClearDepthBuffer();
    rtgi::ClearStencilBuffer();
    rtgi::EndRender();
    rtgi::Present();

    rtgi::GetViewport( sInitialViewportWidth, sInitialViewportHeight );
    rtgi::CreateSharedShaderParameter( "modelViewProjectionMatrix",  rtgi::SharedShaderParameterType_Matrix44 );
    rtgi::CreateSharedShaderParameter( "objectSpaceLightPosition",   rtgi::SharedShaderParameterType_Vector3  );
    rtgi::CreateSharedShaderParameter( "objectSpaceCameraPosition",  rtgi::SharedShaderParameterType_Vector3  );

    // initialize matrices
    sTransform.SetToIdentity();
    sView.SetToIdentity();
    sProjection.SetToIdentity();

    // load debug/default content
    content::LoadManager::InstallLoader( new TextureLoader  );
    content::LoadManager::InstallLoader( new EffectLoader   );
    content::LoadManager::InstallLoader( new MaterialLoader );
    content::LoadManager::InstallLoader( new GeometryLoader );
    content::LoadManager::InstallLoader( new SceneLoader    );

    sInventory = new content::Inventory();
    sInventory->AddRef();

    content::LoadManager::Load( "runtime/art/DebugCamera.dae",   sInventory );
    content::LoadManager::Load( "runtime/art/DebugMaterial.dae", sInventory );
    content::LoadManager::Load( "runtime/art/PhongEffect.dae",   sInventory );

    sDebugCamera   = sInventory->Find< Camera >      ( "debugCameraShape" );
    sDebugMaterial = sInventory->Find< Material >    ( "debugMaterial" );
    sPhongEffect   = sInventory->Find< rtgi::Effect >( "phongEffect" );
    sCamera        = sDebugCamera;

    BMPClass bmp;
    BMPError errorCode = BMPLoad( "runtime/art/DebugTexture.bmp", bmp );
    Assert( errorCode == BMPNOERROR );

    rtgi::TextureDataDesc textureDataDesc;
    textureDataDesc.dimensions                   = rtgi::TextureDimensions_2D;
    textureDataDesc.pixelFormat                  = rtgi::TexturePixelFormat_R8_G8_B8_UI_NORM;
    textureDataDesc.data                         = bmp.bytes;
    textureDataDesc.width                        = bmp.width;
    textureDataDesc.height                       = bmp.height;
    textureDataDesc.generateMipMaps              = true;

    sDebugTexture = rtgi::CreateTexture( textureDataDesc );
    sDebugTexture->AddRef();

    sDebugTextFont = rtgi::CreateTextFont();
    sDebugTextFont->AddRef();

    sIsInitialized = true;
}

void Context::Terminate()
{
    Assert( sIsInitialized );

    sIsInitialized = false;

    sScene         = content::Ref< Scene >();
    sCamera        = content::Ref< Camera >();
    sPhongEffect   = content::Ref< rtgi::Effect >();
    sDebugCamera   = content::Ref< Camera >();
    sDebugMaterial = content::Ref< Material >();

    content::LoadManager::Unload( sInventory );

    sInventory->Release();
    sInventory = NULL;

    sDebugTextFont->Release();
    sDebugTextFont = NULL;

    sDebugTexture->Release();
    sDebugTexture = NULL;

    AssignRef( sCurrentRenderStrategy, NULL );

    rtgi::DestroySharedShaderParameter( "modelViewProjectionMatrix" );
    rtgi::DestroySharedShaderParameter( "objectSpaceLightPosition"  );
    rtgi::DestroySharedShaderParameter( "objectSpaceCameraPosition" );

    rtgi::Terminate();
}

bool Context::IsInitialized()
{
    return sIsInitialized;
}

RenderStrategy* Context::GetCurrentRenderStrategy()
{
    Assert( sIsInitialized );

    return sCurrentRenderStrategy;
}

void Context::SetCurrentRenderStrategy( RenderStrategy* renderStrategy )
{
    Assert( sIsInitialized );

    AssignRef( sCurrentRenderStrategy, renderStrategy );
}

content::Ref< Camera > Context::GetCurrentCamera()
{
    Assert( sIsInitialized );
    return sCamera;
}

void Context::SetCurrentCamera( content::Ref< Camera > camera )
{
    Assert( sIsInitialized );

    sCamera = camera;

    if ( RefIsNull( sCamera ) )
    {
        sCamera = GetDebugCamera();
    }
}

void Context::SetCurrentViewport( int width, int height )
{
    Assert( sIsInitialized );

    rtgi::SetViewport( width, height );
}

void Context::GetCurrentViewport( int& width, int& height )
{
    Assert( sIsInitialized );

    rtgi::GetViewport( width, height );
}

void Context::GetInitialViewport( int& width, int& height )
{
    Assert( sIsInitialized );

    width  = sInitialViewportWidth;
    height = sInitialViewportHeight;
}

rtgi::Texture* Context::GetDebugTexture()
{
    Assert( sIsInitialized );

    return sDebugTexture;
}

rtgi::TextFont* Context::GetDebugTextFont()
{
    Assert( sIsInitialized );

    return sDebugTextFont;
}

content::Ref< Camera > Context::GetDebugCamera()
{
    Assert( sIsInitialized );

    return sDebugCamera;
}

content::Ref< Material > Context::GetDebugMaterial()
{
    Assert( sIsInitialized );

    return sDebugMaterial;
}

content::Ref< rtgi::Effect > Context::GetPhongEffect()
{
    Assert( sIsInitialized );

    return sPhongEffect;
}

content::Ref< Scene > Context::GetCurrentScene()
{
    Assert( sIsInitialized );

    return sScene;
}

void Context::SetCurrentScene( content::Ref< Scene > scene )
{
    Assert( sIsInitialized );

    sScene = scene;
}

// engine-level shader parameters
void Context::SetCurrentTransformMatrix( const math::Matrix44& transform )
{
    sTransform = transform;

    math::Matrix44 lookAtMatrix, projectionMatrix;

    content::Ref< Camera > camera = GetCurrentCamera();
    camera->GetLookAtMatrix( lookAtMatrix );
    camera->GetProjectionMatrix( projectionMatrix );

    math::Matrix44 modelViewProjectionMatrix = projectionMatrix * lookAtMatrix * sTransform;

    // compute object space positions based on current object transform
    math::Matrix44 transformInverseTranspose = transform;
    transformInverseTranspose.InvertTranspose();
    
    math::Vector3 lightPosition( 0.0f, 1000.0f, 0.0f );

    if ( content::ParameterManager::Contains( "rendering", "debugLightWorldSpacePosition" ) )
    {
        lightPosition = content::ParameterManager::GetParameter< math::Vector3 >( "rendering", "debugLightWorldSpacePosition" );
    }

    math::Vector3 objectSpaceLightPosition  = transformInverseTranspose.Transform( lightPosition );
    math::Vector3 objectSpaceCameraPosition = transformInverseTranspose.Transform( camera->GetPosition() );

    rtgi::SetSharedShaderParameter( "modelViewProjectionMatrix", modelViewProjectionMatrix );
    rtgi::SetSharedShaderParameter( "objectSpaceLightPosition",  objectSpaceLightPosition );
    rtgi::SetSharedShaderParameter( "objectSpaceCameraPosition", objectSpaceCameraPosition );

    if ( sEffectParameters.Contains( "modelViewProjectionMatrix" ) )
    {
        SetEffectParameter( "modelViewProjectionMatrix", modelViewProjectionMatrix );
    }
}

container::List< core::String > Context::GetEffectParameterNames()
{
    return sEffectParameters.Keys();
}

rtgi::EffectParameterType Context::GetEffectParameterType( const core::String& effectParameterName )
{
    Assert( sEffectParameters.Contains( effectParameterName ) );

    return sEffectParameters.Value( effectParameterName ).type;
}

void Context::Update( double timeDeltaSeconds )
{
    Assert( sIsInitialized );

    if ( sCurrentRenderStrategy != NULL )
    {
        sCurrentRenderStrategy->Update( sScene, sCamera, timeDeltaSeconds );
    }
}

void Context::Render()
{
    Assert( sIsInitialized );

    rtgi::BeginRender();

    if ( sCurrentRenderStrategy != NULL )
    {
        sCurrentRenderStrategy->Render( sScene, sCamera );
    }
    else
    {
        rtgi::ClearColorBuffer( rtgi::ColorRGBA( 0, 0, 0, 1 ) );
        rtgi::ClearDepthBuffer();
        rtgi::ClearStencilBuffer();
    }

    rtgi::EndRender();

    rtgi::Present();
}

void Context::SetEffectParameter( const core::String& effectParameterName, const rtgi::Texture* value )
{
    Assert( sEffectParameters.Contains( effectParameterName ) );

    EffectParameter existingValue;

    existingValue = sEffectParameters.Value( effectParameterName );
    
    Assert( existingValue.type ==  rtgi::EffectParameterType_Texture );

    existingValue.dataTexture = value;

    sEffectParameters.Insert( effectParameterName, existingValue );    
}

void Context::SetEffectParameter( const core::String& effectParameterName, const math::Matrix44& value )
{
    Assert( sEffectParameters.Contains( effectParameterName ) );

    EffectParameter existingValue;

    existingValue = sEffectParameters.Value( effectParameterName );

    Assert( existingValue.type == rtgi::EffectParameterType_Matrix44 );

    existingValue.dataMatrix44 = value;

    sEffectParameters.Insert( effectParameterName, existingValue );    
}

void Context::SetEffectParameter( const core::String& effectParameterName, const math::Vector4& value )
{
    Assert( sEffectParameters.Contains( effectParameterName ) );

    EffectParameter existingValue;

    existingValue = sEffectParameters.Value( effectParameterName );

    Assert( existingValue.type == rtgi::EffectParameterType_Vector4 );

    existingValue.dataVector4 = value;

    sEffectParameters.Insert( effectParameterName, existingValue );    
}

void Context::SetEffectParameter( const core::String& effectParameterName, const math::Vector3& value )
{
    Assert( sEffectParameters.Contains( effectParameterName ) );

    EffectParameter existingValue;

    existingValue = sEffectParameters.Value( effectParameterName );

    Assert( existingValue.type == rtgi::EffectParameterType_Vector3 );

    existingValue.dataVector3 = value;

    sEffectParameters.Insert( effectParameterName, existingValue );    
}

void Context::SetEffectParameter( const core::String& effectParameterName, float value )
{
    Assert( sEffectParameters.Contains( effectParameterName ) );

    EffectParameter existingValue;

    existingValue = sEffectParameters.Value( effectParameterName );

    Assert( existingValue.type == rtgi::EffectParameterType_Float );

    existingValue.dataFloat = value;

    sEffectParameters.Insert( effectParameterName, existingValue );    
}

void Context::InsertNewEffectParameter( const core::String& effectParameterName, const rtgi::Texture* value )
{
    EffectParameter newValue;

    newValue.type        = rtgi::EffectParameterType_Texture;
    newValue.dataTexture = value;

    sEffectParameters.Insert( effectParameterName, newValue );
}

void Context::InsertNewEffectParameter( const core::String& effectParameterName, const math::Matrix44& value )
{
    EffectParameter newValue;

    newValue.type         = rtgi::EffectParameterType_Matrix44;
    newValue.dataMatrix44 = value;

    sEffectParameters.Insert( effectParameterName, newValue );
}

void Context::InsertNewEffectParameter( const core::String& effectParameterName, const math::Vector4& value )
{
    EffectParameter newValue;

    newValue.type        = rtgi::EffectParameterType_Vector4;
    newValue.dataVector4 = value;

    sEffectParameters.Insert( effectParameterName, newValue );
}

void Context::InsertNewEffectParameter( const core::String& effectParameterName, const math::Vector3& value )
{
    EffectParameter newValue;

    newValue.type        = rtgi::EffectParameterType_Vector3;
    newValue.dataVector3 = value;

    sEffectParameters.Insert( effectParameterName, newValue );
}

void Context::InsertNewEffectParameter( const core::String& effectParameterName, float value )
{
    EffectParameter newValue;

    newValue.type      = rtgi::EffectParameterType_Float;
    newValue.dataFloat = value;

    sEffectParameters.Insert( effectParameterName, newValue );
}

}
