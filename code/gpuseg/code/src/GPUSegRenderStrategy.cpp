#include "GPUSegRenderStrategy.hpp"

#include "core/Functor.hpp"
#include "core/Printf.hpp"
#include "core/String.hpp"
#include "core/Time.hpp"

#include "math/Vector3.hpp"
#include "math/Matrix44.hpp"

#include "container/List.hpp"
#include "container/Array.hpp"

#include "content/Ref.hpp"
#include "content/Parameter.hpp"

#include "rendering/DebugDraw.hpp"
#include "rendering/Scene.hpp"
#include "rendering/Context.hpp"
#include "rendering/Camera.hpp"
#include "rendering/TextConsole.hpp"

#include "rendering/rtgi/ShaderProgram.hpp"
#include "rendering/rtgi/RTGI.hpp"
#include "rendering/rtgi/FrameBufferObject.hpp"
#include "rendering/rtgi/Texture.hpp"

#include "Config.hpp"
#include "VolumeLoader.hpp"
#include "VolumeFileDesc.hpp"
#include "VolumeDesc.hpp"
#include "Engine.hpp"

static rendering::rtgi::FrameBufferObject* sFrameBufferObject                = NULL;
static rendering::rtgi::ShaderProgram*     sObjectSpacePositionAsColorShader = NULL;
static rendering::rtgi::ShaderProgram*     sVolumeShader                     = NULL;
static rendering::rtgi::Texture*           sBackFacesTexture                 = NULL;
static rendering::rtgi::Texture*           sFrontFacesTexture                = NULL;
static rendering::rtgi::Texture*           sRaycastTexture                   = NULL;
static rendering::rtgi::Texture*           sSourceVolumeTexture              = NULL;
static rendering::rtgi::Texture*           sCurrentLevelSetVolumeTexture     = NULL;
static rendering::rtgi::Texture*           sFrozenLevelSetVolumeTexture      = NULL;
static rendering::rtgi::Texture*           sActiveElementsVolumeTexture      = NULL;

static container::List< math::Vector3 > sSketchPoints;

static content::Parameter< float > sUpdateOpenGLTexture ( "Segmenter",            "updateOpenGLTexture" );

static content::Parameter< float > sCuttingPlaneZValue  ( "GPUSegRenderStrategy", "cuttingPlaneZValue" );
static content::Parameter< float > sRenderingIsosurface ( "GPUSegRenderStrategy", "renderingIsosurface" );
static content::Parameter< float > sDebugRender         ( "GPUSegRenderStrategy", "debugRender" ); 
static content::Parameter< float > sShowSourceData      ( "GPUSegRenderStrategy", "showSourceData" );
static content::Parameter< float > sZeroOutsideWindow   ( "GPUSegRenderStrategy", "zeroOutsideWindow" );
static content::Parameter< float > sRenderLevelSet      ( "GPUSegRenderStrategy", "renderLevelSet" );
static content::Parameter< float > sRenderHalo          ( "GPUSegRenderStrategy", "renderHalo" );
static content::Parameter< float > sConvergenceThreshold( "GPUSegRenderStrategy", "convergenceThreshold" );
static content::Parameter< float > sSourceWindow        ( "GPUSegRenderStrategy", "sourceWindow" );
static content::Parameter< float > sSourceLevel         ( "GPUSegRenderStrategy", "sourceLevel" );

static bool       sCurrentlySketching       = false;
static bool       sVolumeLoaded             = false;
static Engine*    sEngine                   = NULL;

static VolumeDesc sOriginalVolumeDesc;
static double     sTimeDeltaSeconds;

GPUSegRenderStrategy::GPUSegRenderStrategy()
{
    //
    // create rtgi objects
    //
    rendering::rtgi::ShaderProgramDesc shaderProgramDesc;

    shaderProgramDesc.vertexProgramFile         = "shaders/ObjectSpacePositionAsColor.cg";
    shaderProgramDesc.geometryProgramFile       = "shaders/ObjectSpacePositionAsColor.cg";
    shaderProgramDesc.fragmentProgramFile       = "shaders/ObjectSpacePositionAsColor.cg";
    shaderProgramDesc.vertexProgramEntryPoint   = "ObjectSpacePositionAsColorVertexProgram";
    shaderProgramDesc.geometryProgramEntryPoint = "ObjectSpacePositionAsColorGeometryProgram";
    shaderProgramDesc.fragmentProgramEntryPoint = "ObjectSpacePositionAsColorFragmentProgram";

    sObjectSpacePositionAsColorShader = rendering::rtgi::CreateShaderProgram( shaderProgramDesc );
    sObjectSpacePositionAsColorShader->AddRef();

    shaderProgramDesc.vertexProgramFile         = "shaders/RaycastVolume.cg";
    shaderProgramDesc.geometryProgramFile       = "shaders/RaycastVolume.cg";
    shaderProgramDesc.fragmentProgramFile       = "shaders/RaycastVolume.cg";
    shaderProgramDesc.vertexProgramEntryPoint   = "RaycastVolumeVertexProgram";
    shaderProgramDesc.geometryProgramEntryPoint = "RaycastVolumeGeometryProgram";
    shaderProgramDesc.fragmentProgramEntryPoint = "RaycastVolumeFragmentProgram";

    sVolumeShader = rendering::rtgi::CreateShaderProgram( shaderProgramDesc );
    sVolumeShader->AddRef();

    sFrameBufferObject = rendering::rtgi::CreateFrameBufferObject();
    sFrameBufferObject->AddRef();

    rendering::rtgi::TextureDataDesc textureDataDesc;

    textureDataDesc.dimensions  = rendering::rtgi::TextureDimensions_2D;
    textureDataDesc.pixelFormat = rendering::rtgi::TexturePixelFormat_R32_G32_B32_A32_F_DENORM;
    textureDataDesc.width       = 0;
    textureDataDesc.height      = 0;

    sFrontFacesTexture = rendering::rtgi::CreateTexture( textureDataDesc );
    sFrontFacesTexture->AddRef();

    sBackFacesTexture = rendering::rtgi::CreateTexture( textureDataDesc );
    sBackFacesTexture->AddRef();

    textureDataDesc.dimensions  = rendering::rtgi::TextureDimensions_2D;
    textureDataDesc.pixelFormat = rendering::rtgi::TexturePixelFormat_R8_G8_B8_UI_NORM;
    textureDataDesc.width       = 0;
    textureDataDesc.height      = 0;

    sRaycastTexture = rendering::rtgi::CreateTexture( textureDataDesc );
    sRaycastTexture->AddRef();
}

GPUSegRenderStrategy::~GPUSegRenderStrategy()
{
    sRaycastTexture->Release();
    sRaycastTexture = NULL;

    sFrontFacesTexture->Release();
    sFrontFacesTexture = NULL;

    sBackFacesTexture->Release();
    sBackFacesTexture = NULL;

    sFrameBufferObject->Release();
    sFrameBufferObject = NULL;

    sVolumeShader->Release();
    sVolumeShader = NULL;

    sObjectSpacePositionAsColorShader->Release();
    sObjectSpacePositionAsColorShader = NULL;
}

void GPUSegRenderStrategy::SetSourceVolumeTexture( rendering::rtgi::Texture* texture )
{
    AssignRef( sSourceVolumeTexture, texture );
}

void GPUSegRenderStrategy::SetCurrentLevelSetVolumeTexture( rendering::rtgi::Texture* texture )
{
    AssignRef( sCurrentLevelSetVolumeTexture, texture );
}

void GPUSegRenderStrategy::SetFrozenLevelSetVolumeTexture( rendering::rtgi::Texture* texture )
{
    AssignRef( sFrozenLevelSetVolumeTexture, texture );
}

void GPUSegRenderStrategy::SetActiveElementsVolumeTexture( rendering::rtgi::Texture* texture )
{
    AssignRef( sActiveElementsVolumeTexture, texture );
}

void GPUSegRenderStrategy::SetEngine( Engine* engine )
{
    AssignRef( sEngine, engine );
}

void GPUSegRenderStrategy::LoadVolume( const VolumeDesc& volumeDesc )
{
    sOriginalVolumeDesc = volumeDesc;
    sVolumeLoaded       = true;
}

void GPUSegRenderStrategy::UnloadVolume()
{
    sVolumeLoaded = false;

    AssignRef( sSourceVolumeTexture,          NULL );
    AssignRef( sCurrentLevelSetVolumeTexture, NULL );
    AssignRef( sActiveElementsVolumeTexture,  NULL ); 
}

void GPUSegRenderStrategy::Update( content::Ref< rendering::Scene > scene, content::Ref< rendering::Camera > camera, double timeDeltaSeconds )
{
    sTimeDeltaSeconds = timeDeltaSeconds;
}

void GPUSegRenderStrategy::Render( content::Ref< rendering::Scene > scene, content::Ref< rendering::Camera > camera )
{
    if ( sVolumeLoaded )
    {
        //
        // calculate bounding box
        //
        VolumeDesc volumeDesc = sOriginalVolumeDesc;

        math::Vector3 textureSpaceVoxelDimensions( 1.0f / volumeDesc.numVoxelsX, 1.0f / volumeDesc.numVoxelsY, 1.0f / volumeDesc.numVoxelsZ );
        math::Vector3 boundingBoxDimensions( volumeDesc.numVoxelsX, volumeDesc.numVoxelsY, volumeDesc.numVoxelsZ * volumeDesc.zAnisotropy );
        math::Vector3 boundingBoxHalfDimensions = boundingBoxDimensions * 0.5f;

        math::Vector3 boxP1, boxP2, currentBoundingBoxExtent, cuttingPlaneP1, cuttingPlaneP2, cuttingPlaneP3, cuttingPlaneP4;
        bool renderCuttingPlane;

        if ( sOriginalVolumeDesc.upDirection == math::Vector3(  0,  0, -1 ) )
        {
            currentBoundingBoxExtent = boundingBoxHalfDimensions;

            currentBoundingBoxExtent[ math::Z ] =
                - boundingBoxHalfDimensions[ math::Z ] + ( boundingBoxHalfDimensions[ math::Z ] * math::Clamp( 0, 1, sCuttingPlaneZValue.GetValue() ) * 2 );

            boxP1            = currentBoundingBoxExtent;
            boxP2            = - boundingBoxHalfDimensions;
            boxP2[ math::Z ] = - boxP2[ math::Z ];

            cuttingPlaneP1 = math::Vector3( - currentBoundingBoxExtent[ math::X ], - currentBoundingBoxExtent[ math::Y ], currentBoundingBoxExtent[ math::Z ] );
            cuttingPlaneP2 = math::Vector3( - currentBoundingBoxExtent[ math::X ],   currentBoundingBoxExtent[ math::Y ], currentBoundingBoxExtent[ math::Z ] );
            cuttingPlaneP3 = math::Vector3(   currentBoundingBoxExtent[ math::X ],   currentBoundingBoxExtent[ math::Y ], currentBoundingBoxExtent[ math::Z ] );
            cuttingPlaneP4 = math::Vector3(   currentBoundingBoxExtent[ math::X ], - currentBoundingBoxExtent[ math::Y ], currentBoundingBoxExtent[ math::Z ] );

            if ( sCuttingPlaneZValue.GetValue() <= 0 )
            {
                renderCuttingPlane = false;
            }
            else
            {
                renderCuttingPlane = true;
            }
        }
        else
        {
            currentBoundingBoxExtent = boundingBoxHalfDimensions;

            currentBoundingBoxExtent[ math::Z ] =
                - boundingBoxHalfDimensions[ math::Z ] + ( boundingBoxHalfDimensions[ math::Z ] * math::Clamp( 0, 1, sCuttingPlaneZValue.GetValue() ) * 2 );

            boxP1 = - boundingBoxHalfDimensions;
            boxP2 = currentBoundingBoxExtent;

            cuttingPlaneP1 = math::Vector3( - currentBoundingBoxExtent[ math::X ], - currentBoundingBoxExtent[ math::Y ], currentBoundingBoxExtent[ math::Z ] );
            cuttingPlaneP2 = math::Vector3(   currentBoundingBoxExtent[ math::X ], - currentBoundingBoxExtent[ math::Y ], currentBoundingBoxExtent[ math::Z ] );
            cuttingPlaneP3 = math::Vector3(   currentBoundingBoxExtent[ math::X ],   currentBoundingBoxExtent[ math::Y ], currentBoundingBoxExtent[ math::Z ] );
            cuttingPlaneP4 = math::Vector3( - currentBoundingBoxExtent[ math::X ],   currentBoundingBoxExtent[ math::Y ], currentBoundingBoxExtent[ math::Z ] );

            if ( sCuttingPlaneZValue.GetValue() >= 1.0 )
            {
                renderCuttingPlane = false;
            }
            else
            {
                renderCuttingPlane = true;
            }
        }

        float fovYRadians = 0.0f;
        float aspectRatio = 0.0f;
        float nearPlane   = 0.0f;
        float farPlane    = 0.0f;

        camera->GetProjectionParameters( fovYRadians, aspectRatio, nearPlane, farPlane );

        Assert( fovYRadians != 0.0f );
        Assert( aspectRatio != 0.0f );
        Assert( nearPlane   != 0.0f );
        Assert( farPlane    != 0.0f );

        float imagePlaneXYScaleFactor = 5;
        float imagePlaneZScaleFactor  = 2;
        float imagePlaneHalfHeight    = atan( nearPlane * fovYRadians * 0.5 ) * imagePlaneXYScaleFactor;
        float imagePlaneHalfWidth     = imagePlaneHalfHeight * aspectRatio;

        math::Vector3 imagePlaneTopLeftCameraSpace     = math::Vector3( - imagePlaneHalfWidth,   imagePlaneHalfHeight, - nearPlane * imagePlaneZScaleFactor );
        math::Vector3 imagePlaneTopRightCameraSpace    = math::Vector3(   imagePlaneHalfWidth,   imagePlaneHalfHeight, - nearPlane * imagePlaneZScaleFactor );
        math::Vector3 imagePlaneBottomRightCameraSpace = math::Vector3(   imagePlaneHalfWidth, - imagePlaneHalfHeight, - nearPlane * imagePlaneZScaleFactor );
        math::Vector3 imagePlaneBottomLeftCameraSpace  = math::Vector3( - imagePlaneHalfWidth, - imagePlaneHalfHeight, - nearPlane * imagePlaneZScaleFactor );

        math::Matrix44 cameraToWorldMatrix;
        camera->GetLookAtMatrix( cameraToWorldMatrix );
        cameraToWorldMatrix.InvertTranspose();

        math::Vector3 imagePlaneTopLeftWorldSpace     = cameraToWorldMatrix.Transform( imagePlaneTopLeftCameraSpace );
        math::Vector3 imagePlaneTopRightWorldSpace    = cameraToWorldMatrix.Transform( imagePlaneTopRightCameraSpace );
        math::Vector3 imagePlaneBottomRightWorldSpace = cameraToWorldMatrix.Transform( imagePlaneBottomRightCameraSpace );
        math::Vector3 imagePlaneBottomLeftWorldSpace  = cameraToWorldMatrix.Transform( imagePlaneBottomLeftCameraSpace );

        int initialWidth, initialHeight;
        rendering::Context::GetInitialViewport( initialWidth, initialHeight );

        //
        // render back faces to a texture
        //
        sFrameBufferObject->Bind( sBackFacesTexture );
        
        rendering::rtgi::SetFaceCullingEnabled( true );
        rendering::rtgi::SetFaceCullingFace( rendering::rtgi::FaceCullingFace_Front );
        rendering::rtgi::ClearColorBuffer( rendering::rtgi::ColorRGBA( 0, 0, 0, 0 ) );
        rendering::rtgi::ClearDepthBuffer();
        rendering::rtgi::ClearStencilBuffer();

        sObjectSpacePositionAsColorShader->BeginSetShaderParameters();
        sObjectSpacePositionAsColorShader->SetShaderParameter( "boundingBoxDimensions", boundingBoxDimensions );
        sObjectSpacePositionAsColorShader->EndSetShaderParameters();
        sObjectSpacePositionAsColorShader->Bind();

        rendering::DebugDraw::DrawCube( boxP1, boxP2, rendering::rtgi::ColorRGB( 1, 1, 1 ) );

        sObjectSpacePositionAsColorShader->Unbind();

        sFrameBufferObject->Unbind();

        //
        // render front faces to a texture
        //
        sFrameBufferObject->Bind( sFrontFacesTexture );

        rendering::rtgi::ClearColorBuffer( rendering::rtgi::ColorRGBA( 0, 1, 1, 0 ) );
        rendering::rtgi::ClearDepthBuffer();
        rendering::rtgi::ClearStencilBuffer();

        sObjectSpacePositionAsColorShader->BeginSetShaderParameters();
        sObjectSpacePositionAsColorShader->SetShaderParameter( "boundingBoxDimensions", boundingBoxDimensions );
        sObjectSpacePositionAsColorShader->EndSetShaderParameters();
        sObjectSpacePositionAsColorShader->Bind();

        rendering::rtgi::SetFaceCullingEnabled( false );
        rendering::rtgi::SetDepthTestingEnabled( false );
        rendering::rtgi::SetDepthWritingEnabled( true );

        rendering::DebugDraw::DrawQuad(
            imagePlaneTopLeftWorldSpace,
            imagePlaneTopRightWorldSpace,
            imagePlaneBottomRightWorldSpace,
            imagePlaneBottomLeftWorldSpace,
            rendering::rtgi::ColorRGB( 1, 1, 1 ) );

        rendering::rtgi::SetFaceCullingEnabled( true );
        rendering::rtgi::SetFaceCullingFace( rendering::rtgi::FaceCullingFace_Back );
        rendering::rtgi::SetDepthTestingEnabled( true );
        rendering::rtgi::SetDepthWritingEnabled( true );

        rendering::DebugDraw::DrawCube( boxP1, boxP2, rendering::rtgi::ColorRGB( 1, 1, 1 ) );

        sObjectSpacePositionAsColorShader->Unbind();

        sFrameBufferObject->Unbind();

        //
        // render the volume
        //
        sFrameBufferObject->Bind( sRaycastTexture );

        rendering::rtgi::TextureSamplerStateDesc textureSamplerStateDesc;

        textureSamplerStateDesc.textureSamplerInterpolationMode = rendering::rtgi::TextureSamplerInterpolationMode_Smooth;
        textureSamplerStateDesc.textureSamplerWrapMode          = rendering::rtgi::TextureSamplerWrapMode_ClampToEdge;

        rendering::rtgi::ClearColorBuffer( GetClearColor() );
        rendering::rtgi::ClearDepthBuffer();
        rendering::rtgi::ClearStencilBuffer();
        rendering::rtgi::SetFaceCullingEnabled( false );

        sVolumeShader->BeginSetShaderParameters();
        sVolumeShader->SetShaderParameter( "sourceVolumeSampler",                   sSourceVolumeTexture,                   textureSamplerStateDesc );
        sVolumeShader->SetShaderParameter( "levelSetVolumeSampler",                 sCurrentLevelSetVolumeTexture,          textureSamplerStateDesc );
        sVolumeShader->SetShaderParameter( "frozenLevelSetVolumeSampler",           sFrozenLevelSetVolumeTexture,           textureSamplerStateDesc );
        sVolumeShader->SetShaderParameter( "activeElementsSampler",                 sActiveElementsVolumeTexture,           textureSamplerStateDesc );
        sVolumeShader->SetShaderParameter( "frontFacesSampler",                     sFrontFacesTexture,                     textureSamplerStateDesc );
        sVolumeShader->SetShaderParameter( "backFacesSampler",                      sBackFacesTexture,                      textureSamplerStateDesc );
        sVolumeShader->SetShaderParameter( "textureSpaceVoxelDimensions",           textureSpaceVoxelDimensions );
        sVolumeShader->SetShaderParameter( "cuttingPlaneZValue",                    sCuttingPlaneZValue.GetValue() );
        sVolumeShader->SetShaderParameter( "isosurface",                            sRenderingIsosurface.GetValue() );
        sVolumeShader->SetShaderParameter( "debugRender",                           sUpdateOpenGLTexture.GetValue() );
        sVolumeShader->SetShaderParameter( "renderLevelSet",                        sRenderLevelSet.GetValue() );
        sVolumeShader->SetShaderParameter( "renderHalo",                            sRenderHalo.GetValue() );
        sVolumeShader->SetShaderParameter( "showSourceData",                        sShowSourceData.GetValue() );
        sVolumeShader->SetShaderParameter( "zeroOutsideWindow",                     sZeroOutsideWindow.GetValue() );
        sVolumeShader->SetShaderParameter( "sourceWindow",                          sSourceWindow.GetValue() );
        sVolumeShader->SetShaderParameter( "sourceLevel",                           sSourceLevel.GetValue() );
        sVolumeShader->EndSetShaderParameters();

        sVolumeShader->Bind();

        rendering::DebugDraw::DrawCube( boxP1, boxP2, rendering::rtgi::ColorRGB( 1, 1, 1 ) );
        
        sVolumeShader->Unbind();

        //
        // render sketch interaction
        //
        if ( sSketchPoints.Size() > 0 )
        {
            rendering::rtgi::SetLineWidth( 1 );

             rendering::DebugDraw::DrawLine2D(
                sSketchPoints.At( 0 )[ math::X ],
                sSketchPoints.At( 0 )[ math::Y ],
                sSketchPoints.At( sSketchPoints.Size() - 1 )[ math::X ],
                sSketchPoints.At( sSketchPoints.Size() - 1 )[ math::Y ],
                rendering::rtgi::ColorRGB( 0, 1, 0 ) );

             rendering::DebugDraw::DrawPoint2D(
                sSketchPoints.At( 0 )[ math::X ],
                sSketchPoints.At( 0 )[ math::Y ],
                rendering::rtgi::ColorRGB( 0, 1, 0 ),
                4 );

             rendering::DebugDraw::DrawPoint2D(
                sSketchPoints.At( sSketchPoints.Size() - 1 )[ math::X ],
                sSketchPoints.At( sSketchPoints.Size() - 1 )[ math::Y ],
                rendering::rtgi::ColorRGB( 0, 1, 0 ),
                4 );
        }

        //
        // render debug stuff
        //
        rendering::rtgi::SetLineWidth( 2 );

        rendering::rtgi::SetColorWritingEnabled( true );
        rendering::rtgi::SetDepthTestingEnabled( true );
        rendering::rtgi::SetWireframeRenderingEnabled( true );
        rendering::rtgi::SetFaceCullingEnabled( true );
        rendering::rtgi::SetFaceCullingFace( rendering::rtgi::FaceCullingFace_Back );

        //if ( sDebugRender.GetValue() == 1.0f )
        //{
        //    rendering::DebugDraw::DrawCube( boxP1, boxP2, rendering::rtgi::ColorRGB( 1, 1, 1 ) );
        //}

        if ( renderCuttingPlane )
        {
#ifdef WHITE_BACKGROUND
            rendering::DebugDraw::DrawQuad( cuttingPlaneP1, cuttingPlaneP2, cuttingPlaneP3, cuttingPlaneP4, rendering::rtgi::ColorRGB( 0, 0, 0 ) );
#else
            rendering::DebugDraw::DrawQuad( cuttingPlaneP1, cuttingPlaneP2, cuttingPlaneP3, cuttingPlaneP4, rendering::rtgi::ColorRGB( 1, 1, 1 ) );
#endif
        }

        rendering::rtgi::SetWireframeRenderingEnabled( false );
        rendering::rtgi::SetFaceCullingEnabled( false );

        if ( sDebugRender.GetValue() > 0.0f )
        {
            //rendering::DebugDraw::DrawLine( math::Vector3( 0, 0, 0 ), math::Vector3( boundingBoxDimensions[ math::X ], 0, 0 ), rendering::rtgi::ColorRGB( 1, 0, 0 ) );
            //rendering::DebugDraw::DrawLine( math::Vector3( 0, 0, 0 ), math::Vector3( 0, boundingBoxDimensions[ math::Y ], 0 ), rendering::rtgi::ColorRGB( 0, 1, 0 ) );
            //rendering::DebugDraw::DrawLine( math::Vector3( 0, 0, 0 ), math::Vector3( 0, 0, boundingBoxDimensions[ math::Z ] ), rendering::rtgi::ColorRGB( 0, 0, 1 ) );

#ifdef WHITE_BACKGROUND
            rendering::rtgi::SetColor( rendering::rtgi::ColorRGB( 0, 0, 0 ) );
#else
            rendering::rtgi::SetColor( rendering::rtgi::ColorRGB( 1, 1, 1 ) );
#endif


            rendering::TextConsole::RenderCallback();
        }

        sFrameBufferObject->Unbind();

        sRaycastTexture->Bind( textureSamplerStateDesc );
        rendering::DebugDraw::DrawFullScreenQuad2D( rendering::rtgi::ColorRGB( 1, 1, 1 ) );
        sRaycastTexture->Unbind();
    }
}

void GPUSegRenderStrategy::BeginPlaceSeed()
{
    sCurrentlySketching = true;
}

void GPUSegRenderStrategy::AddSeedPoint( int virtualScreenX, int virtualScreenY )
{
    math::Vector3 point( virtualScreenX, virtualScreenY, 0 );

    math::Vector3 intersection = ComputeIntersectionWithCuttingPlane( point );

    VolumeDesc volumeDesc = sOriginalVolumeDesc;
    math::Vector3 boundingBoxDimensions( volumeDesc.numVoxelsX, volumeDesc.numVoxelsY, volumeDesc.numVoxelsZ * volumeDesc.zAnisotropy );
    math::Vector3 boundingBoxHalfDimensions = boundingBoxDimensions * 0.5f;

    if ( intersection[ math::X ] >= - boundingBoxHalfDimensions[ math::X ] && intersection[ math::X ] <= boundingBoxHalfDimensions[ math::X ] &&
         intersection[ math::Y ] >= - boundingBoxHalfDimensions[ math::Y ] && intersection[ math::Y ] <= boundingBoxHalfDimensions[ math::Y ]  )
    {
        sSketchPoints.Append( point );
    }
}

void GPUSegRenderStrategy::EndPlaceSeed()
{
    if ( sSketchPoints.Size() > 2 )
    {
        math::Vector3 startIntersection = ComputeIntersectionWithCuttingPlane( sSketchPoints.At( 0 ) );
        math::Vector3 endIntersection   = ComputeIntersectionWithCuttingPlane( sSketchPoints.At( sSketchPoints.Size() - 1 ) );
        math::Vector3 startToEnd        = endIntersection - startIntersection;
        float         length            = startToEnd.Length();
        float         radius            = length / 2.0f;
        math::Vector3 midpoint          = startIntersection + ( startToEnd / 2.0f );

        VolumeDesc paddedVolumeDesc = sOriginalVolumeDesc;

        math::Vector3 paddedBoundingBoxDimensionsWorldSpace = math::Vector3(
            paddedVolumeDesc.numVoxelsX,
            paddedVolumeDesc.numVoxelsY,
            paddedVolumeDesc.numVoxelsZ * paddedVolumeDesc.zAnisotropy );

        math::Vector3 paddedBoundingBoxHalfDimensionsWorldSpace = paddedBoundingBoxDimensionsWorldSpace * 0.5f;
        math::Vector3 intersectionPaddedBoundingBoxSpace        = midpoint + paddedBoundingBoxHalfDimensionsWorldSpace;
        math::Vector3 intersectionNonPaddedTextureSpace         = math::Vector3(
            intersectionPaddedBoundingBoxSpace[ math::X ],
            intersectionPaddedBoundingBoxSpace[ math::Y ],
            intersectionPaddedBoundingBoxSpace[ math::Z ] / sOriginalVolumeDesc.zAnisotropy );

        sEngine->InitializeSeed( intersectionNonPaddedTextureSpace, radius );
    }

    sSketchPoints.Clear();

    sCurrentlySketching = false;
}

math::Vector3 GPUSegRenderStrategy::ComputeIntersectionWithCuttingPlane( math::Vector3 virtualScreenCoordinates )
{
    int initialViewportWidth, initialViewportHeight;

    rendering::Context::GetInitialViewport( initialViewportWidth, initialViewportHeight );

    content::Ref< rendering::Camera > camera = rendering::Context::GetCurrentCamera();

    float fovYRadians = 0.0f;
    float aspectRatio = 0.0f;
    float nearPlane   = 0.0f;
    float farPlane    = 0.0f;

    camera->GetProjectionParameters( fovYRadians, aspectRatio, nearPlane, farPlane );

    Assert( fovYRadians != 0.0f );
    Assert( aspectRatio != 0.0f );
    Assert( nearPlane   != 0.0f );
    Assert( farPlane    != 0.0f );

    float nearPlaneHalfHeight = nearPlane * tan( fovYRadians / 2.0f );
    float normalizedX         = ( virtualScreenCoordinates[ math::X ] - ( initialViewportWidth  / 2.0f ) ) / ( initialViewportWidth  / 2.0f );
    float normalizedY         = ( virtualScreenCoordinates[ math::Y ] - ( initialViewportHeight / 2.0f ) ) / ( initialViewportHeight / 2.0f );

    Assert( aspectRatio == 1.0f );

    float nearPlaneX          = normalizedX * nearPlaneHalfHeight;
    float nearPlaneY          = normalizedY * nearPlaneHalfHeight;  // this depends on aspect ratio being 1

    math::Matrix44 inverseViewMatrix;

    rendering::Context::GetCurrentCamera()->GetLookAtMatrix( inverseViewMatrix );
    inverseViewMatrix.InvertTranspose();

    math::Vector3 eyeToPointOnNearPlaneViewSpace  = math::Vector3( nearPlaneX, nearPlaneY, - nearPlane );
    math::Vector3 eyeToPointOnNearPlaneWorldSpace = inverseViewMatrix.TransformVector( eyeToPointOnNearPlaneViewSpace );

    eyeToPointOnNearPlaneWorldSpace.Normalize();

    VolumeDesc volumeDesc                   = sOriginalVolumeDesc;
    float      cuttingPlaneWorldSpaceZValue =
        ( sCuttingPlaneZValue.GetValue() * volumeDesc.numVoxelsZ * volumeDesc.zAnisotropy ) -
        ( volumeDesc.numVoxelsZ * volumeDesc.zAnisotropy * 0.5f );

    math::Vector3 eye                          = rendering::Context::GetCurrentCamera()->GetPosition();
    float         signedDistanceToCuttingPlane = ( cuttingPlaneWorldSpaceZValue - eye[ math::Z ] ) / eyeToPointOnNearPlaneWorldSpace[ math::Z ];
    math::Vector3 intersection                 = eye + ( eyeToPointOnNearPlaneWorldSpace * signedDistanceToCuttingPlane );

    return intersection;
}