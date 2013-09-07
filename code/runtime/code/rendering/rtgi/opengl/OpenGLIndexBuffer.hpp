#ifndef RENDERING_RTGI_OPENGL_INDEX_BUFFER_HPP
#define RENDERING_RTGI_OPENGL_INDEX_BUFFER_HPP

#include "core/RefCounted.hpp"

#include "rendering/rtgi/IndexBuffer.hpp"

namespace rendering
{

namespace rtgi
{

struct Vertex;

class OpenGLIndexBuffer : public IndexBuffer
{
public:
    OpenGLIndexBuffer( const unsigned short* const rawIndexData, const unsigned int numBytesIndexData );

    virtual ~OpenGLIndexBuffer();

    virtual void Bind()                                            const;
    virtual void Render( PrimitiveRenderMode primitiveRenderMode ) const;
    virtual void Unbind()                                          const;

private:
    unsigned int mIndexBufferID;
    unsigned int mIndexBufferNumElements;
};

}

}

#endif