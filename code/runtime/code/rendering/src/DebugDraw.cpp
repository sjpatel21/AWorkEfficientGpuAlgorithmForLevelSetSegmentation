#include "rendering/DebugDraw.hpp"

#include "math/Vector3.hpp"
#include "math/Matrix44.hpp"
#include "math/Utility.hpp"

#include "rendering/Camera.hpp"
#include "rendering/Context.hpp"
#include "rendering/rtgi/Color.hpp"
#include "rendering/rtgi/RTGI.hpp"
#include "rendering/rtgi/Effect.hpp"

namespace rendering
{

namespace DebugDraw
{

enum DebugDrawMatrixMode
{
    DebugDrawMatrixMode_2D,
    DebugDrawMatrixMode_3D
};

void SetDebugDrawMatrixState( DebugDrawMatrixMode debugDrawMode )
{
    math::Matrix44 modelMatrix, viewMatrix, projectionMatrix;

    switch( debugDrawMode )
    {
        case( DebugDrawMatrixMode_3D ):

            //
            // model matrix
            //
            modelMatrix.SetToIdentity();

            //
            // view matrix
            //
            Context::GetCurrentCamera()->GetLookAtMatrix( viewMatrix );

            //
            // projection matrix
            //
            Context::GetCurrentCamera()->GetProjectionMatrix( projectionMatrix );

            break;

        case( DebugDrawMatrixMode_2D ):

            //
            // model matrix
            //
            modelMatrix.SetToIdentity();

            //
            // view matrix
            //
            viewMatrix.SetToIdentity();

            //
            // projection matrix
            //
            int width, height;

            Context::GetInitialViewport( width, height );

            projectionMatrix.SetTo2DOrthographic( 0, width, 0, height );

            break;

        default:

            Assert( 0 );
    }

    //
    // now set the matrix state
    //
    rtgi::SetTransformMatrix( modelMatrix );
    rtgi::SetViewMatrix( viewMatrix );
    rtgi::SetProjectionMatrix( projectionMatrix );

    //
    // mirror our changes in the high-level context in case we are binding a shader
    //
    Context::SetCurrentTransformMatrix( modelMatrix );
}

void DrawPoint( const math::Vector3& position, const rtgi::ColorRGB& color, int pointSize )
{
    SetDebugDrawMatrixState( DebugDrawMatrixMode_3D );
    rtgi::DebugDrawPoint( position, color, pointSize );
}


void DrawLine( const math::Vector3& p1, const math::Vector3& p2, const rtgi::ColorRGB& color )
{
    SetDebugDrawMatrixState( DebugDrawMatrixMode_3D );
    rtgi::DebugDrawLine( p1, p2, color );
}

void DrawTriangle( const math::Vector3& p1, const math::Vector3& p2, const math::Vector3& p3, const rtgi::ColorRGB& color )
{
    SetDebugDrawMatrixState( DebugDrawMatrixMode_3D );

    rtgi::DebugDrawTriangle( p1, p2, p3, color );
}

void DrawQuad( const math::Vector3& p1, const math::Vector3& p2, const math::Vector3& p3, const math::Vector3& p4, const rtgi::ColorRGB& color )
{
    SetDebugDrawMatrixState( DebugDrawMatrixMode_3D );

    rtgi::DebugDrawQuad( p1, p2, p3, p4, color );
}

void DrawSphere( const math::Vector3& position, float sphereRadius, const rtgi::ColorRGB& color )
{
    SetDebugDrawMatrixState( DebugDrawMatrixMode_3D );
    rtgi::DebugDrawSphere( position, sphereRadius, color );
}

void DrawTeapot( const math::Vector3& position, float size, const rtgi::ColorRGB& color )
{
    //
    // set matrix state
    //
    math::Matrix44 modelMatrix, viewMatrix, projectionMatrix;

    modelMatrix.SetToIdentity();
    projectionMatrix.SetToIdentity();

    modelMatrix.SetToTranslate( position );

    // scale the size of the teapot so that the size is given by bounding sphere radius
    modelMatrix.SetToScale( math::Vector3( 0.561f, 0.561f, 0.561f ) );

    Context::GetCurrentCamera()->GetProjectionMatrix( projectionMatrix );
    Context::GetCurrentCamera()->GetLookAtMatrix( viewMatrix );

    rtgi::SetTransformMatrix( modelMatrix );
    rtgi::SetViewMatrix( viewMatrix );
    rtgi::SetProjectionMatrix( projectionMatrix );

    Context::SetCurrentTransformMatrix( modelMatrix );

    //
    // draw teapot
    //
    rtgi::DebugDrawTeapot( size, color );
}

void DrawCube( const math::Vector3& p1, const math::Vector3& p2, const rtgi::ColorRGB& color )
{
    SetDebugDrawMatrixState( DebugDrawMatrixMode_3D );

    rtgi::DebugDrawCube( p1, p2, color );
}

void DrawPoint2D( int x, int y, const rtgi::ColorRGB& color, int pointSize )
{
    SetDebugDrawMatrixState( DebugDrawMatrixMode_2D );

    rtgi::SetDepthTestingEnabled( false );
    rtgi::SetDepthWritingEnabled( false );

    rtgi::DebugDrawPoint2D( x, y, color, pointSize );

    rtgi::SetDepthTestingEnabled( true );
    rtgi::SetDepthWritingEnabled( true );
}

void DrawLine2D( int virtualScreenX1, int virtualScreenY1, int virtualScreenX2, int virtualScreenY2, const rtgi::ColorRGB& color )
{
    SetDebugDrawMatrixState( DebugDrawMatrixMode_2D );

    rtgi::SetDepthTestingEnabled( false );
    rtgi::SetDepthWritingEnabled( false );

    rtgi::DebugDrawLine2D( virtualScreenX1, virtualScreenY1, virtualScreenX2, virtualScreenY2, color );

    rtgi::SetDepthTestingEnabled( true );
    rtgi::SetDepthWritingEnabled( true );
}

void DrawFullScreenQuad2D( const rtgi::ColorRGB& color )
{
    math::Matrix44 identityMatrix;

    identityMatrix.SetToIdentity();

    rtgi::SetTransformMatrix( identityMatrix );
    rtgi::SetViewMatrix( identityMatrix );
    rtgi::SetProjectionMatrix( identityMatrix );

    rtgi::SetDepthTestingEnabled( false );
    rtgi::SetDepthWritingEnabled( false );

    rtgi::DebugDrawFullScreenQuad2D( color );

    rtgi::SetDepthTestingEnabled( true );
    rtgi::SetDepthWritingEnabled( true );
}

void DrawQuad2D( int topLeftX, int topLeftY, int width, int height, const rtgi::ColorRGB& color )
{
    rtgi::SetDepthTestingEnabled( false );
    rtgi::SetDepthWritingEnabled( false );

    rtgi::DebugDrawQuad2D( topLeftX, topLeftY, width, height, color );

    rtgi::SetDepthTestingEnabled( true );
    rtgi::SetDepthWritingEnabled( true );
}

}

}