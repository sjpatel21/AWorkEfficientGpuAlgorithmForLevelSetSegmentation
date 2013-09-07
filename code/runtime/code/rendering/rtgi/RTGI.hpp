#ifndef RENDERING_RTGI_HPP
#define RENDERING_RTGI_HPP

#if defined(PLATFORM_WIN32)
#define NOMINMAX
#include <windows.h>
#elif defined(PLATFORM_OSX)
#include <Carbon/Carbon.h>
#include <AGL/AGL.h>
#endif

#include <vector>

#include "core/NameSpaceID.hpp"

#include "rendering/rtgi/Color.hpp"

namespace core
{
    class String;
}

namespace math
{
    class Vector3;
    class Vector4;
    class Matrix44;
}

namespace rendering
{

namespace rtgi
{

CORE_NAMESPACE_ID

class VertexBuffer;
class IndexBuffer;
class PixelBuffer;
class Effect;
class ShaderProgram;
class Texture;
class BufferTexture;
class TextFont;
class FrameBufferObject;

struct TextureDataDesc;
struct BufferTextureDataDesc;
struct ShaderProgramDesc;

#if defined(PLATFORM_WIN32)
typedef HWND WINDOW_HANDLE;
#elif defined(PLATFORM_OSX)
typedef HIViewRef WINDOW_HANDLE;
#endif

enum SharedShaderParameterType
{
    SharedShaderParameterType_Float,
    SharedShaderParameterType_Vector3,
    SharedShaderParameterType_Vector4,
    SharedShaderParameterType_Matrix44,
    SharedShaderParameterType_Texture
};

// This is a debug function to draw arrays of vertices/texture coords straight through (no indices)
// Uses shorts (didn't want to have to implement a fancy system for switching).
enum DebugDrawVerticesPrimitive
{
    DebugDrawVerticesPrimitive_Point,
    DebugDrawVerticesPrimitive_Line,
    DebugDrawVerticesPrimitive_Quad
};

enum FaceCullingFace
{
    FaceCullingFace_Front,
    FaceCullingFace_Back
};

enum PrimitiveRenderMode
{
    PrimitiveRenderMode_Points,
    PrimitiveRenderMode_Lines,
    PrimitiveRenderMode_LineStrip,
    PrimitiveRenderMode_Triangles,
    PrimitiveRenderMode_TriangleStrip,
    PrimitiveRenderMode_TriangleFan,

    PrimitiveRenderMode_Invalid
};

// initialize / terminate
void Initialize( const WINDOW_HANDLE windowHandle );
void Terminate();


// debug
void CheckErrors();
void Debug();

// finish
void Finish();

// debug 3D
void DebugDrawPoint   ( const math::Vector3& position, const ColorRGB& color, int pointSize );
void DebugDrawLine    ( const math::Vector3& p1, const math::Vector3& p2, const ColorRGB& color );
void DebugDrawCube    ( const math::Vector3& p1, const math::Vector3& p2, const ColorRGB& color );
void DebugDrawTriangle( const math::Vector3& p1, const math::Vector3& p2, const math::Vector3& p3, const ColorRGB& color );
void DebugDrawQuad    ( const math::Vector3& p1, const math::Vector3& p2, const math::Vector3& p3, const math::Vector3& p4, const ColorRGB& color );
void DebugDrawSphere  ( const math::Vector3& position, float sphereRadius, const ColorRGB& color );
void DebugDrawTeapot  ( float size, const ColorRGB& color );

// debug 2D
void DebugDrawPoint2D( int x,  int y, const ColorRGB& color, int pointSize );
void DebugDrawLine2D ( int x1, int y1, int x2, int y2, const ColorRGB& color );
void DebugDrawFullScreenQuad2D( const ColorRGB& color );
void DebugDrawQuad2D( int topLeftX, int topLeftY, int width, int height, const rtgi::ColorRGB& color );

void DebugDrawVerticesTextureCoordinatesShort( const std::vector<short>* vertices,
                                               const std::vector< std::vector<short> >* textureCoordinateArray,
                                               int numUnitsPerVertex, int numUnitsPerTextureCoordinate,
                                               DebugDrawVerticesPrimitive primitiveType);

// global shader parameter management
void CreateSharedShaderParameter ( const core::String& parameterName, SharedShaderParameterType parameterType );
void DestroySharedShaderParameter( const core::String& parameterName );
void SetSharedShaderParameter    ( const core::String& parameterName, const math::Matrix44& value );
void SetSharedShaderParameter    ( const core::String& parameterName, const math::Vector3&  value );

// used internally by shaders/effects themselves
bool IsSharedShaderParameter         ( const core::String& parameterName );
void ConnectSharedShaderParameters   ( ShaderProgram*      shaderProgram );
void ConnectSharedShaderParameters   ( Effect*             effect );
void DisconnectSharedShaderParameters( ShaderProgram*      shaderProgram );
void DisconnectSharedShaderParameters( Effect*             effect );

// render state
void SetDepthClampEnabled        ( bool                  depthClampEnabled );
void SetFaceCullingEnabled       ( bool                  faceCullingEnabled );
void SetFaceCullingFace          ( FaceCullingFace       faceCullingFace );
void SetAlphaBlendingEnabled     ( const bool            alphaBlendingEnabled );
void SetStencilTestingEnabled    ( const bool            stencilTestingEnabled );
void SetDepthTestingEnabled      ( const bool            depthTestingEnabled );
void SetStencilWritingEnabled    ( const bool            stencilWritingEnabled );
void SetDepthWritingEnabled      ( const bool            depthWritingEnabled );
void SetColorWritingEnabled      ( const bool            colorWritingEnabled );
void SetWireframeRenderingEnabled( const bool            wireframeRenderingEnabled );
void SetPolygonOffsetEnabled     ( const bool            polygonOffsetEnabled );
void SetLineWidth                ( const unsigned int    lineWidth );
void SetPointSize                ( const unsigned int    pointSize );
void SetColor                    ( const ColorRGB&       color );
void SetPolygonOffset            ( float factor, float scale );

// frame buffer object state
void SetFrameBufferObjectBound( bool frameBufferObjectBound );
bool IsFrameBufferObjectBound();

// matrix state
void SetTransformMatrix ( const math::Matrix44& transformMatrix );
void SetViewMatrix      ( const math::Matrix44& viewMatrix );
void SetProjectionMatrix( const math::Matrix44& projectionMatrix );

// frame buffer
void ClearColorBuffer( const rtgi::ColorRGBA& clearColor );
void ClearDepthBuffer();
void ClearStencilBuffer();

// viewport
void SetViewport( int  width, int  height );
void GetViewport( int& width, int& height );

// unproject
void GetVirtualScreenCoordinates(
     int  initialViewportWidth,
     int  initialViewportHeight,
     int  screenX,
     int  screenY,
     int& virtualScreenX,
     int& virtualScreenY );


// factories
TextFont*            CreateTextFont();
VertexBuffer*        CreateVertexBuffer ( const void* rawVertexData, const unsigned int numBytes );
IndexBuffer*         CreateIndexBuffer  ( const unsigned short* rawIndexData, const unsigned int numIndices );
PixelBuffer*         CreatePixelBuffer  ( const unsigned int numBytes );
Texture*             CreateTexture      ( const TextureDataDesc& textureDataDesc );
BufferTexture*       CreateBufferTexture( const BufferTextureDataDesc& bufferTextureDataDesc );
ShaderProgram*       CreateShaderProgram( const ShaderProgramDesc& shaderProgramDesc );
Effect*              CreateEffect       ( const core::String& effectFile );
FrameBufferObject*   CreateFrameBufferObject();

// scene
void BeginRender();
void EndRender();
void Present();

};

}

#endif
