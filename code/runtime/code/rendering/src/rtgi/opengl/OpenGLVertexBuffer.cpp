#include "rendering/rtgi/opengl/OpenGLVertexBuffer.hpp"

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

#include "rendering/rtgi/Config.hpp"
#include "rendering/rtgi/RTGI.hpp"

#include "rendering/rtgi/opengl/OpenGLTexture.hpp"
#include "rendering/rtgi/opengl/Extensions.hpp"

namespace rendering
{

namespace rtgi
{

VertexDataSourceDesc::VertexDataSourceDesc() :
numVertices              ( 0xffffffff ),
numCoordinatesPerSemantic( 0xffffffff ),
offset                   ( 0xffffffff ),
stride                   ( 0xffffffff ),
vertexBufferDataType     ( VertexBufferDataType_Invalid ),
vertexBuffer             ( NULL )
{
}

void VertexBuffer::Render( PrimitiveRenderMode primitiveRenderMode, unsigned int numVertices )
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

    glDrawArrays(
        glRenderMode,            // mode
        0,                       // offset for starting index where start_of_array + offset = first_index
        numVertices );           // number of vertices

    CheckErrors();
}

OpenGLVertexBuffer::OpenGLVertexBuffer(
    const void*        rawVertexData,
    const unsigned int numBytes ) :
mVertexBufferID( 0 )
{
    glGenBuffersARB( 1, &mVertexBufferID );
    glBindBufferARB( GL_ARRAY_BUFFER_ARB, mVertexBufferID );
    glBufferDataARB( GL_ARRAY_BUFFER_ARB, numBytes, rawVertexData, GL_STATIC_DRAW_ARB );
    CheckErrors();
}

OpenGLVertexBuffer::~OpenGLVertexBuffer()
{
    glDeleteBuffersARB( 1, &mVertexBufferID );
    CheckErrors();
}

void OpenGLVertexBuffer::Bind() const
{
    glBindBufferARB( GL_ARRAY_BUFFER_ARB, mVertexBufferID );
    CheckErrors();
}

void OpenGLVertexBuffer::Unbind() const
{
    glBindBufferARB( GL_ARRAY_BUFFER_ARB, 0 );
    CheckErrors();
}

int OpenGLVertexBuffer::GetOpenGLVertexBufferID() const
{
    return mVertexBufferID;
}

}

}
