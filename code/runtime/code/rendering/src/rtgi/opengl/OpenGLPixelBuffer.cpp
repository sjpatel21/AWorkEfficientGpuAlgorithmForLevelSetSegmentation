#include "rendering/rtgi/opengl/OpenGLPixelBuffer.hpp"

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

#include "core/Assert.hpp"

#include "rendering/rtgi/RTGI.hpp"

#include "rendering/rtgi/opengl/OpenGLTexture.hpp"
#include "rendering/rtgi/opengl/Extensions.hpp"

namespace rendering
{

namespace rtgi
{


OpenGLPixelBuffer::OpenGLPixelBuffer( const unsigned int numBytes ) :
mPixelBufferID( 0 )
{
    // pixel buffer
    glGenBuffersARB( 1, &mPixelBufferID );
    glBindBufferARB( GL_PIXEL_UNPACK_BUFFER_ARB, mPixelBufferID );
    glBufferDataARB( GL_PIXEL_UNPACK_BUFFER_ARB, numBytes, 0, GL_STATIC_DRAW_ARB );
    glBindBufferARB( GL_PIXEL_UNPACK_BUFFER_ARB, 0 );

    CheckErrors();
}

OpenGLPixelBuffer::~OpenGLPixelBuffer()
{
    glDeleteBuffersARB( 1, &mPixelBufferID );

    CheckErrors();
}

void OpenGLPixelBuffer::Bind() const
{
    glBindBufferARB( GL_PIXEL_UNPACK_BUFFER_ARB, mPixelBufferID );

    CheckErrors();
}

void OpenGLPixelBuffer::Unbind() const
{
    glBindBufferARB( GL_PIXEL_UNPACK_BUFFER_ARB, 0 );

    CheckErrors();
}

unsigned int OpenGLPixelBuffer::GetOpenGLPixelBufferID() const
{
    return mPixelBufferID;
}

}

}
