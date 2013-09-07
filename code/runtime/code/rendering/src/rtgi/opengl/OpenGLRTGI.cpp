#if defined(PLATFORM_WIN32)

#define NOMINMAX
#include <windows.h>

#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>

#elif defined(PLATFORM_OSX)

#include <dlfcn.h>

#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#include <OpenGL/glext.h>
#include <GLUT/glut.h>

#endif

#include <Cg/cg.h>
#include <Cg/cgGL.h>

#include "core/Printf.hpp"
#include "core/Assert.hpp"
#include "core/String.hpp"

#include "math/Vector3.hpp"
#include "math/Vector4.hpp"
#include "math/Matrix44.hpp"
#include "math/Utility.hpp"

#include "rendering/rtgi/Config.hpp"
#include "rendering/rtgi/Color.hpp"
#include "rendering/rtgi/RTGI.hpp"
#include "rendering/rtgi/Effect.hpp"
#include "rendering/rtgi/ShaderProgram.hpp"
#include "rendering/rtgi/Texture.hpp"
#include "rendering/rtgi/TextFont.hpp"
#include "rendering/rtgi/IndexBuffer.hpp"
#include "rendering/rtgi/VertexBuffer.hpp"
#include "rendering/rtgi/PixelBuffer.hpp"
#include "rendering/rtgi/FrameBufferObject.hpp"

#include "rendering/rtgi/opengl/Extensions.hpp"
#include "rendering/rtgi/opengl/OpenGLVertexBuffer.hpp"
#include "rendering/rtgi/opengl/OpenGLIndexBuffer.hpp"
#include "rendering/rtgi/opengl/OpenGLPixelBuffer.hpp"
#include "rendering/rtgi/opengl/OpenGLEffect.hpp"
#include "rendering/rtgi/opengl/OpenGLShaderProgram.hpp"
#include "rendering/rtgi/opengl/OpenGLTexture.hpp"
#include "rendering/rtgi/opengl/OpenGLTextFont.hpp"
#include "rendering/rtgi/opengl/OpenGLFrameBufferObject.hpp"

namespace rendering
{

namespace rtgi
{

// windowing context
#if defined(PLATFORM_WIN32)
static HDC   sDeviceContext = NULL;
static HGLRC sRenderContext = NULL;
static HWND  sWindowHandle  = NULL;
#elif defined(PLATFORM_OSX)
static void* openGL_libraryPtr = NULL;
static AGLContext sRenderContext = NULL;
static HIViewRef sWindowHandle = NULL;
#endif

// shader context
CGcontext gCGContext = NULL;
container::Map< core::String, CGparameter > gSharedShaderParameters;

// matrix state
static math::Matrix44 sModelMatrix;
static math::Matrix44 sViewMatrix;

// frame buffer object state
static bool sFrameBufferObjectBound = false;

void  CheckErrors();

void* GetExtensionAddress( const core::String& extension );
void  LoadExtensions();

void InitializeGL( const WINDOW_HANDLE windowHandle );
void InitializeGLState();
void TerminateGL();
void InitializeCg();
void TerminateCg();


#if defined(PLATFORM_WIN32)
void Initialize( const WINDOW_HANDLE windowHandle )
{
    InitializeGL( windowHandle );
    
    InitializeGLState();
    
    InitializeCg();
    
    // disable vsync
    wglSwapIntervalEXT( 0 );

}
    
void InitializeGL( const WINDOW_HANDLE windowHandle )
{
    sWindowHandle  = windowHandle;
    sDeviceContext = GetDC( windowHandle );
    
    PIXELFORMATDESCRIPTOR idealPixelFormatDesc, actualPixelFormatDesc;
    
    // set the pixel format for the DC
    ZeroMemory( &idealPixelFormatDesc,  sizeof( idealPixelFormatDesc  ) );
    ZeroMemory( &actualPixelFormatDesc, sizeof( actualPixelFormatDesc ) );
    
    idealPixelFormatDesc.nSize        = sizeof( idealPixelFormatDesc );
    idealPixelFormatDesc.nVersion     = 1;
    idealPixelFormatDesc.dwFlags      = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
    idealPixelFormatDesc.iPixelType   = PFD_TYPE_RGBA;
    idealPixelFormatDesc.cColorBits   = 32;
    idealPixelFormatDesc.cDepthBits   = 24;
    idealPixelFormatDesc.cStencilBits = 8;
    idealPixelFormatDesc.iLayerType   = PFD_MAIN_PLANE;
    
    int pixelFormat;
    
    pixelFormat = ChoosePixelFormat( sDeviceContext, &idealPixelFormatDesc );
    SetPixelFormat( sDeviceContext, pixelFormat, &idealPixelFormatDesc );
    
    DescribePixelFormat( sDeviceContext, pixelFormat, sizeof( actualPixelFormatDesc ), &actualPixelFormatDesc );
    
    ReleaseAssert( idealPixelFormatDesc.cColorBits   == actualPixelFormatDesc.cColorBits );
    ReleaseAssert( idealPixelFormatDesc.cDepthBits   == actualPixelFormatDesc.cDepthBits );
    ReleaseAssert( idealPixelFormatDesc.cStencilBits == actualPixelFormatDesc.cStencilBits );
    
    // create and enable the render context
    sRenderContext = wglCreateContext( sDeviceContext );
    wglMakeCurrent( sDeviceContext, sRenderContext );
    
    LoadExtensions();
}

void Terminate()
{
    TerminateGL();
    
    TerminateCg();

    wglMakeCurrent( NULL, NULL );
    
    wglDeleteContext( sRenderContext );
    
    ReleaseDC( sWindowHandle, sDeviceContext );
}

#elif defined(PLATFORM_OSX)
    
void Initialize( const WINDOW_HANDLE windowHandle )
{
    InitializeGL( windowHandle );
    
    InitializeGLState();
    
    InitializeCg();
}
    
void InitializeGL( const WINDOW_HANDLE windowHandle )
{
    sWindowHandle = windowHandle;
    
    GLint pixelAttributes[] = { AGL_RGBA,
        AGL_DOUBLEBUFFER,
        AGL_ACCELERATED,
        AGL_RED_SIZE, 8,
        AGL_GREEN_SIZE, 8,
        AGL_BLUE_SIZE, 8,
        AGL_ALPHA_SIZE, 8,
        AGL_DEPTH_SIZE, 24,
        AGL_STENCIL_SIZE, 8,
        AGL_NONE };
    
    AGLPixelFormat pixelFormat = aglChoosePixelFormat( NULL, 0, pixelAttributes );
    
    ReleaseAssert( pixelFormat != NULL );
    
    GLint checkSize = 0;
    
    aglDescribePixelFormat( pixelFormat, AGL_DEPTH_SIZE, &checkSize );
    ReleaseAssert( checkSize == pixelAttributes[ 12 ] );
    checkSize = 0;
    aglDescribePixelFormat( pixelFormat, AGL_STENCIL_SIZE, &checkSize );
    ReleaseAssert( checkSize == pixelAttributes[ 14 ] );
    checkSize = 0;
    
    GLint checkSizes[4];
    memset( &checkSizes, 0, sizeof( GLint ) * 4 );
    
    aglDescribePixelFormat( pixelFormat, AGL_RED_SIZE, &checkSizes[ 0 ] );
    aglDescribePixelFormat( pixelFormat, AGL_BLUE_SIZE, &checkSizes[ 1 ] );
    aglDescribePixelFormat( pixelFormat, AGL_GREEN_SIZE, &checkSizes[ 2 ] );
    aglDescribePixelFormat( pixelFormat, AGL_ALPHA_SIZE, &checkSizes[ 3 ] );
    
    ReleaseAssert( ( checkSizes[ 0 ] + checkSizes[ 1 ] + checkSizes[ 2 ] + checkSizes[ 3 ] ) == 
                   ( pixelAttributes[ 4 ] + pixelAttributes[ 6 ] + pixelAttributes[ 8 ] + pixelAttributes[ 10 ] ) );
    
    sRenderContext = aglCreateContext( pixelFormat, NULL );
    aglSetHIViewRef( sRenderContext, sWindowHandle );
    
    aglSetCurrentContext( sRenderContext );
    
    aglDestroyPixelFormat( pixelFormat );
    
    GLint swapInterval = 0;
    aglSetInteger( sRenderContext, AGL_SWAP_INTERVAL, &swapInterval );
    
    openGL_libraryPtr = dlopen( "/System/Library/Frameworks/OpenGL.framework/OpenGL", RTLD_LAZY );
    ReleaseAssert( openGL_libraryPtr != NULL );
}
    
void Terminate()
{
    TerminateGL();
    TerminateCg();

    aglDestroyContext( sRenderContext );
    
    dlclose( openGL_libraryPtr );
    openGL_libraryPtr = NULL;
}
#endif
    
void InitializeGLState()
{
    glEnable( GL_DEPTH_TEST );    
    glDepthFunc( GL_LEQUAL );
        
    sModelMatrix.SetToIdentity();
    sViewMatrix.SetToIdentity();
    
    CheckErrors();
}

void TerminateGL()
{
    CheckErrors();
}
    
void InitializeCg()
{
    gCGContext = cgCreateContext();
    
    ReleaseAssert( gCGContext != NULL );
    
    cgGLRegisterStates( gCGContext );
    cgGLSetManageTextureParameters( gCGContext, CG_TRUE );
}
    
void TerminateCg()
{
    cgDestroyContext( gCGContext );
    gCGContext = NULL;
}

void Finish()
{
    glFinish();
}

void DebugDrawLine( const math::Vector3& p1, const math::Vector3& p2, const ColorRGB& color )
{
    glColor3f( color[ R ], color[ G ], color[ B ] );

    glBegin( GL_LINES );

    glVertex3f( p1[ math::X ], p1[ math::Y ], p1[ math::Z ] );
    glVertex3f( p2[ math::X ], p2[ math::Y ], p2[ math::Z ] );

    glEnd();
}

void DebugDrawTriangle( const math::Vector3& p1, const math::Vector3& p2, const math::Vector3& p3, const ColorRGB& color )
{
    glColor3f( color[ R ], color[ G ], color[ B ] );

    glBegin( GL_TRIANGLES );

    glVertex3f( p1[ math::X ], p1[ math::Y ], p1[ math::Z ] );
    glVertex3f( p2[ math::X ], p2[ math::Y ], p2[ math::Z ] );
    glVertex3f( p3[ math::X ], p3[ math::Y ], p3[ math::Z ] );

    glEnd();
}

void DebugDrawQuad( const math::Vector3& p1, const math::Vector3& p2, const math::Vector3& p3, const math::Vector3& p4, const ColorRGB& color )
{
    glColor3f( color[ R ], color[ G ], color[ B ] );

    glBegin( GL_QUADS );

    glVertex3f( p1[ math::X ], p1[ math::Y ], p1[ math::Z ] );
    glVertex3f( p2[ math::X ], p2[ math::Y ], p2[ math::Z ] );
    glVertex3f( p3[ math::X ], p3[ math::Y ], p3[ math::Z ] );
    glVertex3f( p4[ math::X ], p4[ math::Y ], p4[ math::Z ] );

    glEnd();
}

void DebugDrawPoint( const math::Vector3& position, const ColorRGB& color, int pointSize )
{
    glPointSize( pointSize );
    glColor3f( color[ R ], color[ G ], color[ B ] );

    glBegin( GL_POINTS );

    glVertex3f( position[ math::X ], position[ math::Y ], position[ math::Z ] );

    glEnd();

    CheckErrors();
}

void DebugDrawSphere( const math::Vector3& position, float sphereRadius, const ColorRGB& color )
{
    int numLatitudeLines  = 10;
    int numLongitudeLines = 10;

    float latitudeStep  = - math::PI / numLatitudeLines;
    float longitudeStep = ( 2 * math::PI ) / numLongitudeLines;

    math::Vector3 currentPoint, furtherSouthPoint, furtherEastWestPoint;

    glColor3f( color[ R ], color[ G ], color[ B ] );

    glBegin( GL_TRIANGLE_STRIP );

    // northSouthTheta traces from north pole to south pole
    float northSouthTheta = math::PI / 2;

    // eastWestTheta traces around each latitude line
    float eastWestTheta = 0;

    for ( int i = 0; i <= numLatitudeLines; i++ )
    {
        float currentLatitudeRadius = cos( northSouthTheta )                * sphereRadius;
        float nextLatitudeRadius    = cos( northSouthTheta + latitudeStep ) * sphereRadius;

        for ( int j = 0; j <= numLongitudeLines; j++ )
        {
            currentPoint[ math::X ] = position[ math::X ] + ( cos( eastWestTheta   ) * currentLatitudeRadius );
            currentPoint[ math::Y ] = position[ math::Y ] + ( sin( northSouthTheta ) * sphereRadius );
            currentPoint[ math::Z ] = position[ math::Z ] + ( sin( eastWestTheta   ) * currentLatitudeRadius );

            glVertex3f( currentPoint[ math::X ], currentPoint[ math::Y ], currentPoint[ math::Z ] );

            furtherSouthPoint[ math::X ] = position[ math::X ] + ( cos( eastWestTheta                  ) * nextLatitudeRadius );
            furtherSouthPoint[ math::Y ] = position[ math::Y ] + ( sin( northSouthTheta + latitudeStep ) * sphereRadius );
            furtherSouthPoint[ math::Z ] = position[ math::Z ] + ( sin( eastWestTheta                  ) * nextLatitudeRadius );

            glVertex3f( furtherSouthPoint[ math::X ], furtherSouthPoint[ math::Y ], furtherSouthPoint[ math::Z ] );

            eastWestTheta += longitudeStep;
        }

        northSouthTheta += latitudeStep;
    }

    glEnd();
}

void DebugDrawCube( const math::Vector3& p1, const math::Vector3& p2, const ColorRGB& color )
{
    glColor3f( color[ R ], color[ G ], color[ B ] );
    
    math::Vector3 base1( p1[ math::X ], p1[ math::Y ], p1[ math::Z ] );
    math::Vector3 base2( p1[ math::X ], p1[ math::Y ], p2[ math::Z ] );
    math::Vector3 base3( p2[ math::X ], p1[ math::Y ], p2[ math::Z ] );
    math::Vector3 base4( p2[ math::X ], p1[ math::Y ], p1[ math::Z ] );

    math::Vector3 lid1( p1[ math::X ], p2[ math::Y ], p1[ math::Z ] );
    math::Vector3 lid2( p1[ math::X ], p2[ math::Y ], p2[ math::Z ] );
    math::Vector3 lid3( p2[ math::X ], p2[ math::Y ], p2[ math::Z ] );
    math::Vector3 lid4( p2[ math::X ], p2[ math::Y ], p1[ math::Z ] );

    glBegin( GL_QUADS );

        glVertex3f( base1[ math::X ], base1[ math::Y ], base1[ math::Z ] );
        glVertex3f( base4[ math::X ], base4[ math::Y ], base4[ math::Z ] );
        glVertex3f( base3[ math::X ], base3[ math::Y ], base3[ math::Z ] );
        glVertex3f( base2[ math::X ], base2[ math::Y ], base2[ math::Z ] );

        glVertex3f( lid1[ math::X ], lid1[ math::Y ], lid1[ math::Z ] );
        glVertex3f( lid2[ math::X ], lid2[ math::Y ], lid2[ math::Z ] );
        glVertex3f( lid3[ math::X ], lid3[ math::Y ], lid3[ math::Z ] );
        glVertex3f( lid4[ math::X ], lid4[ math::Y ], lid4[ math::Z ] );

        glVertex3f( base1[ math::X ], base1[ math::Y ], base1[ math::Z ] );
        glVertex3f( base2[ math::X ], base2[ math::Y ], base2[ math::Z ] );
        glVertex3f( lid2[ math::X ],  lid2[ math::Y ],  lid2[ math::Z ] );
        glVertex3f( lid1[ math::X ],  lid1[ math::Y ],  lid1[ math::Z ] );

        glVertex3f( base2[ math::X ], base2[ math::Y ], base2[ math::Z ] );
        glVertex3f( base3[ math::X ], base3[ math::Y ], base3[ math::Z ] );
        glVertex3f( lid3[ math::X ],  lid3[ math::Y ],  lid3[ math::Z ] );
        glVertex3f( lid2[ math::X ],  lid2[ math::Y ],  lid2[ math::Z ] );

        glVertex3f( base3[ math::X ], base3[ math::Y ], base3[ math::Z ] );
        glVertex3f( base4[ math::X ], base4[ math::Y ], base4[ math::Z ] );
        glVertex3f( lid4[ math::X ],  lid4[ math::Y ],  lid4[ math::Z ] );
        glVertex3f( lid3[ math::X ],  lid3[ math::Y ],  lid3[ math::Z ] );

        glVertex3f( base1[ math::X ], base1[ math::Y ], base1[ math::Z ] );
        glVertex3f( lid1[ math::X ],  lid1[ math::Y ],  lid1[ math::Z ] );
        glVertex3f( lid4[ math::X ],  lid4[ math::Y ],  lid4[ math::Z ] );
        glVertex3f( base4[ math::X ], base4[ math::Y ], base4[ math::Z ] );

    glEnd();
}

void DebugDrawTeapot( float size, const ColorRGB& color )
{
    glColor3f( color[ R ], color[ G ], color[ B ] );

    int polygonMode[ 2 ];

    glGetIntegerv( GL_POLYGON_MODE, polygonMode );

    if ( polygonMode[ 1 ] == GL_LINE )
    {
        glutWireTeapot( size );
    }
    else
    if ( polygonMode[ 1 ] == GL_FILL )
    {
        glutSolidTeapot( size );
    }
}

void DebugDrawPoint2D( int x, int y, const ColorRGB& color, int pointSize )
{
    glPointSize( pointSize );
    glColor3f( color[ R ], color[ G ], color[ B ] );

    glBegin( GL_POINTS );

    glVertex2i( x, y );

    glEnd();

    CheckErrors();
}

void DebugDrawLine2D( int x1, int y1, int x2, int y2, const ColorRGB& color )
{
    glColor3f( color[ R ], color[ G ], color[ B ] );

    glBegin( GL_LINES );

    glVertex2i( x1, y1 );
    glVertex2i( x2, y2 );

    glEnd();

    CheckErrors();
}

void DebugDrawFullScreenQuad2D( const ColorRGB& color )
{
    glColor3f( color[ R ], color[ G ], color[ B ] );

    glEnable( GL_TEXTURE_2D );

    glBegin( GL_QUADS );

    glTexCoord2f( 0, 0 );
    glVertex3i(-1, -1, 0);

    glTexCoord2f( 1, 0 );
    glVertex3i(1, -1, 0);

    glTexCoord2f( 1, 1 );
    glVertex3i(1, 1, 0);

    glTexCoord2f( 0, 1 );
    glVertex3i(-1, 1, 0);

    glEnd();

    glDisable( GL_TEXTURE_2D );
}

void DebugDrawQuad2D( int topLeftX, int topLeftY, int width, int height, const rtgi::ColorRGB& color )
{
    glColor4f( color[ R ], color[ G ], color[ B ], 1.0f );

    glBegin( GL_QUADS );

    glVertex3i(topLeftX, topLeftY+height, 0);
    glVertex3i(topLeftX+width, topLeftY+height, 0);
    glVertex3i(topLeftX+width, topLeftY, 0);
    glVertex3i(topLeftX, topLeftY, 0);

    glEnd();

    CheckErrors();
}

void DebugDrawVerticesTextureCoordinatesShort( const std::vector<short>* vertices,
                                              const std::vector< std::vector<short> >* textureCoordinateArray,
                                              int numUnitsPerVertex, int numUnitsPerTextureCoordinate, 
                                              DebugDrawVerticesPrimitive primitiveType)
{
    GLenum primitive;
    switch(primitiveType)
    {
    case DebugDrawVerticesPrimitive_Line:
        primitive = GL_LINES;
        break;

    case DebugDrawVerticesPrimitive_Point:
        primitive = GL_POINTS;
        glPointSize( 1.0 );
        break;

    case DebugDrawVerticesPrimitive_Quad:
        primitive = GL_QUADS;
        break;
    }

    int numVertices = vertices->size() / numUnitsPerVertex;
    int numTextureUnits = textureCoordinateArray->size();

    assert(numTextureUnits <= 8);

    glPushClientAttrib(GL_CLIENT_VERTEX_ARRAY_BIT);
    glEnableClientState(GL_VERTEX_ARRAY);
    glVertexPointer( numUnitsPerVertex,
                     GL_SHORT,
                     0,        // tightly packed.
                     &((*vertices)[0]) );

    for( int i = 0; i < numTextureUnits; ++i )
    {
        glClientActiveTextureARB( GL_TEXTURE0 + i );
        glEnableClientState( GL_TEXTURE_COORD_ARRAY );
        glTexCoordPointer( numUnitsPerTextureCoordinate,
                           GL_SHORT,
                           0,            // tightly packed.
                           &((*textureCoordinateArray)[i][0]) );

    }

    glDrawArrays( primitive, 0,  numVertices );

    // Disable client states.
    glDisableClientState(GL_VERTEX_ARRAY);
    for( int i = 0; i < numTextureUnits; ++i )
    {
        glClientActiveTextureARB( GL_TEXTURE0 + i );
        glDisableClientState( GL_TEXTURE_COORD_ARRAY );
    }

    CheckErrors();
    glPopClientAttrib();
}

void Debug()
{
#if defined(PLATFORM_WIN32)
    glStringMarkerGREMEDY( 0, "RTGI Debug String\0" );
#elif defined(PLATFORM_OSX)
    // no-op
#endif
}




void CreateSharedShaderParameter( const core::String& parameterName, SharedShaderParameterType parameterType )
{
    Assert( !gSharedShaderParameters.Contains( parameterName ) );

    CGparameter parameter;

    switch ( parameterType )
    {
        case SharedShaderParameterType_Matrix44:
            parameter = cgCreateParameter( gCGContext, CG_FLOAT4x4 );
            break;

        case SharedShaderParameterType_Vector3:
            parameter = cgCreateParameter( gCGContext, CG_FLOAT3 );
            break;

        default:
            Assert( 0 );
    }

    gSharedShaderParameters.Insert( parameterName, parameter );

    CheckErrors();
}

void DestroySharedShaderParameter( const core::String& parameterName )
{
    Assert( gSharedShaderParameters.Contains( parameterName ) );

    cgDestroyParameter( gSharedShaderParameters.Value( parameterName ) );

    CheckErrors();
}

void SetSharedShaderParameter( const core::String& parameterName, const math::Matrix44& value )
{
    Assert( gSharedShaderParameters.Contains( parameterName ) );

    cgGLSetMatrixParameterfr( gSharedShaderParameters.Value( parameterName ), value.Ref() );

    CheckErrors();
}

void SetSharedShaderParameter( const core::String& parameterName, const math::Vector3& value )
{
    Assert( gSharedShaderParameters.Contains( parameterName ) );

    cgGLSetParameter3f( gSharedShaderParameters.Value( parameterName ), value[0], value[1], value[2] );

    CheckErrors();
}

bool IsSharedShaderParameter( const core::String& parameterName )
{
    return gSharedShaderParameters.Contains( parameterName );
}

void ConnectSharedShaderParameters( ShaderProgram* shaderProgram )
{
    OpenGLShaderProgram* openGLShaderProgram = dynamic_cast< OpenGLShaderProgram* >( shaderProgram );

    foreach_key_value ( core::String parameterName, ShaderParameterBindDesc shaderParameter, openGLShaderProgram->GetShaderParameterBindDescs() )
    {
        if ( shaderParameter.currentSetState == ShaderParameterSetState_Shared )
        {
            Assert( gSharedShaderParameters.Contains( parameterName ) );

            cgConnectParameter( gSharedShaderParameters.Value( parameterName ), shaderParameter.parameterID );
        }
    }

    CheckErrors();
}

void ConnectSharedShaderParameters( Effect* effect )
{
    OpenGLEffect* openGLEffect = dynamic_cast< OpenGLEffect* >( effect );

    foreach_key_value ( core::String parameterName, EffectParameterBindDesc effectParameterBindDesc, openGLEffect->GetEffectParameterBindDescs() )
    {
        if ( effectParameterBindDesc.currentSetState == EffectParameterSetState_Shared )
        {
            Assert( gSharedShaderParameters.Contains( parameterName ) );

            cgConnectParameter( gSharedShaderParameters.Value( parameterName ), effectParameterBindDesc.parameterID );
        }
    }

    CheckErrors();
}

void DisconnectSharedShaderParameters( ShaderProgram* shaderProgram )
{
    OpenGLShaderProgram* openGLShaderProgram = dynamic_cast< OpenGLShaderProgram* >( shaderProgram );

    foreach_key_value ( core::String parameterName, ShaderParameterBindDesc shaderParameter, openGLShaderProgram->GetShaderParameterBindDescs() )
    {
        if ( shaderParameter.currentSetState == ShaderParameterSetState_Shared )
        {
            Assert( gSharedShaderParameters.Contains( parameterName ) );

            cgDisconnectParameter( shaderParameter.parameterID );
        }
    }

    CheckErrors();
}

void DisconnectSharedShaderParameters( Effect* effect )
{
    OpenGLEffect* openGLEffect = dynamic_cast< OpenGLEffect* >( effect );

    foreach_key_value ( core::String parameterName, EffectParameterBindDesc effectParameterBindDesc, openGLEffect->GetEffectParameterBindDescs() )
    {
        if ( effectParameterBindDesc.currentSetState == EffectParameterSetState_Shared )
        {
            Assert( gSharedShaderParameters.Contains( parameterName ) );

            cgDisconnectParameter( effectParameterBindDesc.parameterID );
        }
    }

    CheckErrors();
}

void SetViewport( int width, int height )
{
    glViewport( 0, 0, width, height );
}

void GetViewport( int& width, int& height )
{
    int viewport[ 4 ];

    glGetIntegerv( GL_VIEWPORT, viewport );

    width  = viewport[ 2 ];
    height = viewport[ 3 ];
}

void GetVirtualScreenCoordinates(
    int initialViewportWidth,
    int initialViewportHeight,
    int screenX,
    int screenY,
    int& virtualScreenX,
    int& virtualScreenY )
{
    GLdouble modelViewMatrix[ 16 ];
    GLdouble projectionMatrix[ 16 ];
    GLint    viewport[ 4 ];

    GLdouble objectX, objectY, objectZ;

    glGetIntegerv( GL_VIEWPORT, viewport );

    glMatrixMode( GL_PROJECTION );

    glPushMatrix();
    glLoadIdentity();
    gluOrtho2D( 0, initialViewportWidth, 0, initialViewportHeight );

    glMatrixMode( GL_MODELVIEW );

    glPushMatrix();
    glLoadIdentity();

    glGetDoublev( GL_MODELVIEW_MATRIX,  modelViewMatrix );
    glGetDoublev( GL_PROJECTION_MATRIX, projectionMatrix );

    gluUnProject( screenX, screenY, 0, modelViewMatrix, projectionMatrix, viewport, &objectX, &objectY, &objectZ );

    glMatrixMode( GL_MODELVIEW );
    glPopMatrix();

    glMatrixMode( GL_PROJECTION );
    glPopMatrix();

    virtualScreenX = static_cast< int >( objectX ) + 1;
    virtualScreenY = static_cast< int >( objectY );
}

TextFont* CreateTextFont()
{
    return new OpenGLTextFont();
}

IndexBuffer* CreateIndexBuffer( const unsigned short* rawIndexData, const unsigned int numIndices )
{
    return new OpenGLIndexBuffer( rawIndexData, numIndices );
}

VertexBuffer* CreateVertexBuffer( const void* rawVertexData, const unsigned int numBytes )
{
    return new OpenGLVertexBuffer( rawVertexData, numBytes );
}

PixelBuffer* CreatePixelBuffer( const unsigned int numBytes )
{
    return new OpenGLPixelBuffer( numBytes );
}

Texture* CreateTexture( const TextureDataDesc& textureDataDesc )
{
    return new OpenGLTexture( textureDataDesc );
}

BufferTexture* CreateBufferTexture( const BufferTextureDataDesc& bufferTextureDataDesc )
{
    return new OpenGLBufferTexture( bufferTextureDataDesc );
}

ShaderProgram* CreateShaderProgram( const ShaderProgramDesc& shaderProgramDesc )
{
    return new OpenGLShaderProgram( shaderProgramDesc );
}

Effect* CreateEffect( const core::String& effectFile )
{
    return new OpenGLEffect( effectFile );
}

FrameBufferObject* CreateFrameBufferObject()
{
    return new OpenGLFrameBufferObject();
}

void ClearColorBuffer( const rtgi::ColorRGBA& clearColor )
{
    glClearColor( clearColor[ rtgi::R ], clearColor[ rtgi::G ], clearColor[ rtgi::B ], clearColor[ rtgi::A ] );
    glClear( GL_COLOR_BUFFER_BIT );
}

void ClearDepthBuffer()
{
    glClear( GL_DEPTH_BUFFER_BIT );
}

void ClearStencilBuffer()
{
    glClear( GL_STENCIL_BUFFER_BIT );
}

void BeginRender()
{
#if defined(PLATFORM_WIN32)
    SwapBuffers( sDeviceContext );
#elif defined(PLATFORM_OSX)
    aglSwapBuffers( sRenderContext );
#endif
    
    CheckErrors();
}

void EndRender()
{
    // no-op
}

void Present()
{
    // no-op
}

void CheckErrors()
{
#ifdef BUILD_DEBUG
    GLenum openGLErrorCode;
    
    openGLErrorCode = glGetError();

    const char* errorString = reinterpret_cast< const char* >( gluErrorString( openGLErrorCode ) );

    Assert( openGLErrorCode != GL_INVALID_ENUM );
    Assert( openGLErrorCode != GL_INVALID_VALUE );
    Assert( openGLErrorCode != GL_INVALID_OPERATION );
    Assert( openGLErrorCode != GL_STACK_OVERFLOW );
    Assert( openGLErrorCode != GL_STACK_UNDERFLOW );
    Assert( openGLErrorCode != GL_OUT_OF_MEMORY );

    Assert( openGLErrorCode == GL_NO_ERROR );

#ifndef CAPS_BASIC_EXTENSIONS_ONLY
    openGLErrorCode = glCheckFramebufferStatusEXT( GL_FRAMEBUFFER_EXT );

    Assert( openGLErrorCode != GL_FRAMEBUFFER_UNSUPPORTED_EXT );
    Assert( openGLErrorCode != GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT_EXT );
    Assert( openGLErrorCode != GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT_EXT );
    Assert( openGLErrorCode != GL_FRAMEBUFFER_INCOMPLETE_DIMENSIONS_EXT );
    Assert( openGLErrorCode != GL_FRAMEBUFFER_INCOMPLETE_FORMATS_EXT );
    Assert( openGLErrorCode != GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER_EXT );
    Assert( openGLErrorCode != GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER_EXT );

    Assert( openGLErrorCode == GL_FRAMEBUFFER_COMPLETE_EXT );
#endif

    if ( gCGContext != NULL )
    {
        CGerror cgErrorCode;

        const core::String cgErrorString         = cgGetLastErrorString( &cgErrorCode );
        const core::String cgCompileStatusString = cgGetLastListing( gCGContext );

        if ( cgCompileStatusString != "" )
        {
            core::Printf( "\n" );
            core::Printf( "\n" );
            core::Printf( "\n" );
            core::Printf( cgCompileStatusString );
            core::Printf( "\n" );
            core::Printf( "\n" );
            core::Printf( "\n" );
        }

#ifndef CAPS_BASIC_EXTENSIONS_ONLY
        Assert( cgCompileStatusString == "" );
        Assert( cgErrorCode           == CG_NO_ERROR );
#endif
    }
#endif
}

bool CheckExtension( const core::String& extensionString )
{
    const char*    extension  = extensionString.ToAscii();
    const GLubyte* extensions = NULL;
    const GLubyte* start;
    GLubyte*       where, *terminator;

    // Extension names should not have spaces.
    where = (GLubyte *) strchr(extension, ' ');
    if (where || *extension == '\0')
    {
        return false;
    }

    extensions = glGetString(GL_EXTENSIONS);

    // It takes a bit of care to be fool-proof about parsing the
    // OpenGL extensions string. Don't be fooled by sub-strings,
    // etc.
    start = extensions;
    for (;;)
    {
        where = (GLubyte *) strstr( (const char *) start, extension );
        if ( !where )
        {
            break;
        }
        terminator = where + strlen(extension);

        if (where == start || *(where - 1) == ' ')
        {
            if (*terminator == ' ' || *terminator == '\0')
            {
                return true;
            }
        }

        start = terminator;
    }

    return false;
}


#if defined(PLATFORM_WIN32)
void* GetExtensionAddress( const core::String& extension )
{
    void* address = wglGetProcAddress( extension.ToAscii() );
    Assert( address != NULL );

    return address;
}
#elif defined(PLATFORM_OSX)
    
void* GetExtensionAddress (const char *name)
{
    void* address = dlsym( openGL_libraryPtr, name );
    
    Assert( address != NULL );
    
    return address;
}
    
#endif

#if defined(PLATFORM_WIN32)
void APIENTRY DummyStringMarkerGREMEDY( GLsizei len, const void* string )
{
}

void LoadExtensions()
{
    //
    // if we're in gDebugger, then get this extension, otherwise
    // point glStringMarkerGREMEDY to a dummy function
    //
    if ( CheckExtension( "GL_GREMEDY_string_marker" ) )
    {
        glStringMarkerGREMEDY = ( PFNGLSTRINGMARKERGREMEDYPROC ) GetExtensionAddress( "glStringMarkerGREMEDY" );
    }
    else
    {
        glStringMarkerGREMEDY = ( PFNGLSTRINGMARKERGREMEDYPROC ) DummyStringMarkerGREMEDY;
    }

    //
    // vertex buffer objects
    //
    ReleaseAssert( CheckExtension( "GL_ARB_vertex_buffer_object" ) );

    glBindBufferARB    = ( PFNGLBINDBUFFERARBPROC )    GetExtensionAddress( "glBindBufferARB" );
    glGenBuffersARB    = ( PFNGLGENBUFFERSARBPROC )    GetExtensionAddress( "glGenBuffersARB" );
    glBufferDataARB    = ( PFNGLBUFFERDATAARBPROC )    GetExtensionAddress( "glBufferDataARB" );
    glDeleteBuffersARB = ( PFNGLDELETEBUFFERSARBPROC ) GetExtensionAddress( "glDeleteBuffersARB" );
    glMapBufferARB     = ( PFNGLMAPBUFFERARBPROC )     GetExtensionAddress( "glMapBufferARB" );
    glUnmapBufferARB   = ( PFNGLUNMAPBUFFERARBPROC )   GetExtensionAddress( "glUnmapBufferARB" );

    //
    // multi-texture
    //
#ifndef CAPS_BASIC_EXTENSIONS_ONLY
    ReleaseAssert( CheckExtension( "GL_ARB_multitexture" ) );

    glActiveTextureARB       = ( PFNGLCLIENTACTIVETEXTUREARBPROC ) GetExtensionAddress( "glActiveTextureARB" );
    glClientActiveTextureARB = ( PFNGLCLIENTACTIVETEXTUREARBPROC ) GetExtensionAddress( "glClientActiveTextureARB" );

    int numTextureCoords     = -1;
    int numTextureImageUnits = -1;
    glGetIntegerv( GL_MAX_TEXTURE_COORDS,      &numTextureCoords );
    glGetIntegerv( GL_MAX_TEXTURE_IMAGE_UNITS, &numTextureImageUnits );

    ReleaseAssert( numTextureCoords     >= 8 );
    ReleaseAssert( numTextureImageUnits >= 8 );
#endif

    //
    // there is a bug on my laptop such that the extension string returned by the driver is corrupted.
    // wglSwapIntervalEXT is actually there, but it doesn't look like it is due to the corrupted
    // extension string.
    //
#ifndef CAPS_BASIC_EXTENSIONS_ONLY
    ReleaseAssert( CheckExtension( "WGL_EXT_swap_control" ) );
#endif

    wglSwapIntervalEXT = ( PFNWGLSWAPINTERVALEXTPROC ) GetExtensionAddress( "wglSwapIntervalEXT" );

    //
    // frame-buffer objects
    //
#ifndef CAPS_BASIC_EXTENSIONS_ONLY
    ReleaseAssert( CheckExtension( "GL_EXT_packed_depth_stencil" ) );
    ReleaseAssert( CheckExtension( "GL_ARB_texture_float" ) );
    ReleaseAssert( CheckExtension( "GL_EXT_framebuffer_object" ) );

    glIsRenderbufferEXT             = ( PFNGLISRENDERBUFFEREXTPROC )             GetExtensionAddress( "glIsRenderbufferEXT" );
    glBindRenderbufferEXT           = ( PFNGLBINDRENDERBUFFEREXTPROC )           GetExtensionAddress( "glBindRenderbufferEXT" );
    glDeleteRenderbuffersEXT        = ( PFNGLDELETERENDERBUFFERSEXTPROC )        GetExtensionAddress( "glDeleteRenderbuffersEXT" );
    glGenRenderbuffersEXT           = ( PFNGLGENRENDERBUFFERSEXTPROC )           GetExtensionAddress( "glGenRenderbuffersEXT" );
    glRenderbufferStorageEXT        = ( PFNGLRENDERBUFFERSTORAGEEXTPROC )        GetExtensionAddress( "glRenderbufferStorageEXT" );
    glGetRenderbufferParameterivEXT = ( PFNGLGETRENDERBUFFERPARAMETERIVEXTPROC ) GetExtensionAddress( "glGetRenderbufferParameterivEXT" );
    glIsFramebufferEXT              = ( PFNGLISFRAMEBUFFEREXTPROC )              GetExtensionAddress( "glIsFramebufferEXT" );
    glBindFramebufferEXT            = ( PFNGLBINDFRAMEBUFFEREXTPROC )            GetExtensionAddress( "glBindFramebufferEXT" );
    glDeleteFramebuffersEXT         = ( PFNGLDELETEFRAMEBUFFERSEXTPROC )         GetExtensionAddress( "glDeleteFramebuffersEXT" );
    glGenFramebuffersEXT            = ( PFNGLGENFRAMEBUFFERSEXTPROC )            GetExtensionAddress( "glGenFramebuffersEXT" );
    glCheckFramebufferStatusEXT     = ( PFNGLCHECKFRAMEBUFFERSTATUSEXTPROC )     GetExtensionAddress( "glCheckFramebufferStatusEXT" );
    glFramebufferTexture1DEXT       = ( PFNGLFRAMEBUFFERTEXTURE1DEXTPROC )       GetExtensionAddress( "glFramebufferTexture1DEXT" );
    glFramebufferTexture2DEXT       = ( PFNGLFRAMEBUFFERTEXTURE2DEXTPROC )       GetExtensionAddress( "glFramebufferTexture2DEXT" );
    glFramebufferTexture3DEXT       = ( PFNGLFRAMEBUFFERTEXTURE3DEXTPROC )       GetExtensionAddress( "glFramebufferTexture3DEXT" );
    glFramebufferRenderbufferEXT    = ( PFNGLFRAMEBUFFERRENDERBUFFEREXTPROC )    GetExtensionAddress( "glFramebufferRenderbufferEXT" );
    glGenerateMipmapEXT             = ( PFNGLGENERATEMIPMAPEXTPROC )             GetExtensionAddress( "glGenerateMipmapEXT" );
    glGetFramebufferAttachmentParameterivEXT = ( PFNGLGETFRAMEBUFFERATTACHMENTPARAMETERIVEXTPROC ) GetExtensionAddress( "glGetFramebufferAttachmentParameterivEXT" );

    ReleaseAssert( CheckExtension( "GL_ARB_draw_buffers" ) );
    glDrawBuffersARB   = ( PFNGLDRAWBUFFERSARBPROC )       GetExtensionAddress( "glDrawBuffersARB" );
#endif

    //
    // 3D textures
    //
#ifndef CAPS_BASIC_EXTENSIONS_ONLY
    ReleaseAssert( CheckExtension( "GL_EXT_texture3D" ) );
    ReleaseAssert( CheckExtension( "GL_ARB_texture_rg" ) );
    ReleaseAssert( CheckExtension( "GL_ARB_texture_float" ) );
    ReleaseAssert( CheckExtension( "GL_EXT_texture_integer" ) );

    glTexImage3D    = ( PFNGLTEXIMAGE3DPROC )    GetExtensionAddress( "glTexImage3D" );
    glTexSubImage3D = ( PFNGLTEXSUBIMAGE3DPROC ) GetExtensionAddress( "glTexSubImage3D" );

    ReleaseAssert( CheckExtension( "GL_EXT_texture_buffer_object" ) );
#endif

    //
    // buffer textures
    //
#ifndef CAPS_BASIC_EXTENSIONS_ONLY
    glTexBufferEXT  = ( PFNGLTEXBUFFEREXTPROC )  GetExtensionAddress( "glTexBufferEXT" );
    ReleaseAssert( CheckExtension( "GL_EXT_texture_buffer_object" ) );
#endif

    //
    // geometry shaders
    //
#ifndef CAPS_BASIC_EXTENSIONS_ONLY
    glFramebufferTextureEXT = ( PFNGLFRAMEBUFFERTEXTUREEXTPROC ) GetExtensionAddress( "glFramebufferTextureEXT" );
    ReleaseAssert( CheckExtension( "GL_NV_gpu_program4" ) );
#endif

}
#endif


void SetWireframeRenderingEnabled( bool wireframeRendering )
{
    if ( wireframeRendering )
    {
        glPolygonMode( GL_FRONT_AND_BACK, GL_LINE );
    }
    else
    {
        glPolygonMode( GL_FRONT_AND_BACK, GL_FILL );
    }
}

void SetPolygonOffsetEnabled( bool polygonOffsetEnabled )
{
    if ( polygonOffsetEnabled )
    {
        glEnable( GL_POLYGON_OFFSET_FILL );
        glEnable( GL_POLYGON_OFFSET_LINE );
        glEnable( GL_POLYGON_OFFSET_POINT );
    }
    else
    {
        glDisable( GL_POLYGON_OFFSET_FILL );
        glDisable( GL_POLYGON_OFFSET_LINE );
        glDisable( GL_POLYGON_OFFSET_POINT );
    }
}

void SetPolygonOffset( float factor, float scale )
{
    glPolygonOffset( factor, scale );
}

void SetFrameBufferObjectBound( bool frameBufferObjectBound )
{
    sFrameBufferObjectBound = frameBufferObjectBound;
}

bool IsFrameBufferObjectBound()
{
    return sFrameBufferObjectBound;
}

void SetTransformMatrix( const math::Matrix44& transformMatrix )
{
    math::Matrix44 modelViewMatrix;

    sModelMatrix    = transformMatrix;
    modelViewMatrix = sViewMatrix * sModelMatrix;

    modelViewMatrix.Transpose();

    glMatrixMode( GL_MODELVIEW );
    glLoadIdentity();
    glMultMatrixf( modelViewMatrix.Ref() );
}

void SetViewMatrix( const math::Matrix44& viewMatrix )
{
    math::Matrix44 modelViewMatrix;

    sViewMatrix     = viewMatrix;
    modelViewMatrix = sViewMatrix * sModelMatrix;

    modelViewMatrix.Transpose();

    glMatrixMode( GL_MODELVIEW );
    glLoadIdentity();
    glMultMatrixf( modelViewMatrix.Ref() );
}

void SetProjectionMatrix( const math::Matrix44& projectionMatrix )
{
    math::Matrix44 openGLProjectionMatrix = projectionMatrix;
    
    openGLProjectionMatrix.Transpose();

    glMatrixMode( GL_PROJECTION );
    glLoadIdentity();
    glMultMatrixf( openGLProjectionMatrix.Ref() );

    //
    // put the matrix mode back to modelview since the glut
    // debug rendering calls depend on being in modelview mode
    //
    glMatrixMode( GL_MODELVIEW );
}



void SetLineWidth( const unsigned int lineWidth )
{
    glLineWidth( lineWidth );
}

void SetPointSize( const unsigned int pointSize )
{
    glPointSize( pointSize );
}

void SetColor( const ColorRGB& color )
{
    glColor3f( color[ R ], color[ G ], color[ B ] );
}

void SetDepthClampEnabled( bool depthClampEnabled )
{
    if ( depthClampEnabled )
    {
        glEnable( GL_DEPTH_CLAMP_NV );
    }
    else
    {
        glDisable( GL_DEPTH_CLAMP_NV );
    }
}

void SetBackFaceCullingEnabled( bool backFaceCullingEnabled )
{
    if ( backFaceCullingEnabled )
    {
        glEnable( GL_CULL_FACE );
        glCullFace( GL_BACK );
    }
}

void SetFaceCullingEnabled( bool faceCullingEnabled )
{
    if ( faceCullingEnabled )
    {
        glEnable( GL_CULL_FACE );
    }
    else
    {
        glDisable( GL_CULL_FACE );
    }
}

void SetFaceCullingFace( FaceCullingFace faceCullingFace )
{
    if ( faceCullingFace == FaceCullingFace_Back )
    {
        glCullFace( GL_BACK );
    }
    else
    {
        glCullFace( GL_FRONT );
    }
}

void SetAlphaBlendingEnabled( const bool alphaBlendingEnabled )
{
    if ( alphaBlendingEnabled )
    {
        glEnable( GL_BLEND );
        glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
    }
    else
    {
        glDisable( GL_BLEND );
    }
}

void SetDepthTestingEnabled( const bool depthTestEnabled )
{
    if ( depthTestEnabled )
    {
        glDepthFunc( GL_LEQUAL );

        glEnable( GL_DEPTH_TEST );
    }
    else
    {
        glDisable( GL_DEPTH_TEST );
    }
}

void SetStencilTestingEnabled( const bool stencilTestEnabled )
{
    if ( stencilTestEnabled )
    {
        glEnable( GL_STENCIL_TEST );
        glStencilFunc( GL_EQUAL, 0xff, 0xff );
        glStencilOp( GL_KEEP, GL_KEEP, GL_KEEP );
    }
    else
    {
        glDisable( GL_STENCIL_TEST );
    }
}

void SetColorWritingEnabled( const bool colorWritingEnabled )
{
    glColorMask( colorWritingEnabled, colorWritingEnabled, colorWritingEnabled, colorWritingEnabled );
}

void SetDepthWritingEnabled( const bool depthWritingEnabled )
{
    glDepthMask( depthWritingEnabled );
}

void SetStencilWritingEnabled( const bool stencilWritingEnabled )
{
    if ( stencilWritingEnabled )
    {
        glEnable( GL_STENCIL_TEST );
        glStencilFunc( GL_ALWAYS, 0xff, 0xff );
        glStencilOp( GL_KEEP, GL_KEEP, GL_REPLACE );
    }
    else
    {
        glDisable( GL_STENCIL_TEST );
    }
}

}

}
