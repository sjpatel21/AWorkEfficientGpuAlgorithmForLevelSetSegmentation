#ifndef RENDERING_RTGI_VERTEX_BUFFER_HPP
#define RENDERING_RTGI_VERTEX_BUFFER_HPP

#include "core/RefCounted.hpp"

#include "rendering/rtgi/RTGI.hpp"

namespace rendering
{

namespace rtgi
{

class VertexBuffer;

enum VertexBufferDataType
{
    VertexBufferDataType_Int,
    VertexBufferDataType_Float,
    VertexBufferDataType_Invalid
};

struct VertexDataSourceDesc
{
    unsigned int         numVertices;
    unsigned int         numCoordinatesPerSemantic;
    unsigned int         offset;
    unsigned int         stride;
    VertexBufferDataType vertexBufferDataType;
    rtgi::VertexBuffer*  vertexBuffer;

    VertexDataSourceDesc();
};

class VertexBuffer : public core::RefCounted
{
public:

    static void Render( PrimitiveRenderMode primitiveRenderMode, unsigned int numVertices );

    virtual void Bind()   const = 0;
    virtual void Unbind() const = 0;

protected:
    VertexBuffer() {};
    virtual ~VertexBuffer() {};
};

}

}

#endif