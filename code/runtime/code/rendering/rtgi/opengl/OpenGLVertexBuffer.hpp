#ifndef RENDERING_RTGI_OPENGL_VERTEX_BUFFER_HPP
#define RENDERING_RTGI_OPENGL_VERTEX_BUFFER_HPP

#include "core/RefCounted.hpp"

#include "rendering/rtgi/VertexBuffer.hpp"

namespace rendering
{

namespace rtgi
{

struct Vertex;

class OpenGLVertexBuffer : public VertexBuffer
{
public:
    OpenGLVertexBuffer( const void* rawVertexData, const unsigned int numBytes );

    virtual ~OpenGLVertexBuffer();

    virtual void Bind()   const;
    virtual void Unbind() const;

    virtual int GetOpenGLVertexBufferID() const;

private:
    unsigned int mVertexBufferID;
};

}

}

#endif