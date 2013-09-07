#include "rendering/rtgi/opengl/OpenGLTextFont.hpp"

#include "core/String.hpp"
#include "core/Assert.hpp"

#include "container/ForEach.hpp"

// include glut last because it conditionally redefines some windows definitions that we have to define anyway
#if defined(PLATFORM_WIN32)

#include <GL/glut.h>

#elif defined(PLATFORM_OSX)

#include <GLUT/glut.h>

#endif

namespace rendering
{

namespace rtgi
{

OpenGLTextFont::OpenGLTextFont()
{
}

OpenGLTextFont::~OpenGLTextFont()
{
}

void OpenGLTextFont::Render( int x, int y, const core::String& text ) const
{
    int viewport[ 4 ];

    glGetIntegerv( GL_VIEWPORT, viewport );

    //
    // set up 2d orthographic projection
    //
    glMatrixMode( GL_PROJECTION );

    glPushMatrix();
    glLoadIdentity();

    gluOrtho2D( 0, viewport[ 2 ], 0, viewport[ 3 ] );

    glMatrixMode( GL_MODELVIEW );

    glPushMatrix();
    glLoadIdentity();

    glRasterPos2i( x, y );

    glDisable( GL_DEPTH_TEST );

    foreach ( char i, text )
    {
        glutBitmapCharacter( GLUT_BITMAP_9_BY_15, i );
    }

    glEnable( GL_DEPTH_TEST );

    glPopMatrix();

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();

    glMatrixMode(GL_MODELVIEW);
}

}

}