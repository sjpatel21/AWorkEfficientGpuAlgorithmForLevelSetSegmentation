#include "rendering/rtgi/opengl/OpenGLIndexBuffer.hpp"

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

#include <stdlib.h>
#include <stdio.h>

#include "core/Assert.hpp"

#include "rendering/rtgi/opengl/Extensions.hpp"

#define BUFFER_OFFSET( i ) ( ( char* )NULL + ( i ) )

namespace rendering
{

namespace rtgi
{

OpenGLIndexBuffer::OpenGLIndexBuffer(
    const unsigned short* rawIndexData,
    const unsigned int    numIndices ) :
mIndexBufferID            ( 0 ),
mIndexBufferNumElements   ( 0 )
{
    mIndexBufferNumElements = numIndices;

    // index buffer
    glGenBuffersARB( 1, &mIndexBufferID );
    glBindBufferARB( GL_ELEMENT_ARRAY_BUFFER_ARB, mIndexBufferID );
    glBufferDataARB( GL_ELEMENT_ARRAY_BUFFER_ARB, numIndices * sizeof( unsigned short ), rawIndexData, GL_DYNAMIC_DRAW_ARB );

    CheckErrors();
}

OpenGLIndexBuffer::~OpenGLIndexBuffer()
{
    glDeleteBuffersARB( 1, &mIndexBufferID );

    CheckErrors();
}

void OpenGLIndexBuffer::Bind() const
{
    glBindBufferARB( GL_ELEMENT_ARRAY_BUFFER_ARB, mIndexBufferID );

    CheckErrors();
}

void OpenGLIndexBuffer::Render( PrimitiveRenderMode primitiveRenderMode ) const
{
    GLenum glRenderMode;

    switch ( primitiveRenderMode )
    {
        case PrimitiveRenderMode_Points:
            glRenderMode = GL_POINTS;
            break;

        case PrimitiveRenderMode_Lines:
            glRenderMode = GL_LINES;
            break;

        case PrimitiveRenderMode_LineStrip:
            glRenderMode = GL_LINE_STRIP;
            break;

        case PrimitiveRenderMode_Triangles:
            glRenderMode = GL_TRIANGLES;
            break;

        case PrimitiveRenderMode_TriangleStrip:
            glRenderMode = GL_TRIANGLE_STRIP;
            break;

        case PrimitiveRenderMode_TriangleFan:
            glRenderMode = GL_TRIANGLE_FAN;
            break;
    }

    CheckErrors();

    glDrawElements(
        glRenderMode,            // mode
        mIndexBufferNumElements, // number of indices
        GL_UNSIGNED_SHORT,       // type
        0 );                     // offset for starting index where start_of_array + offset = first_index

    CheckErrors();
}

void OpenGLIndexBuffer::Unbind() const
{
    glBindBufferARB( GL_ELEMENT_ARRAY_BUFFER_ARB, 0 );
}

}

}