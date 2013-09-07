#ifndef RENDERING_RTGI_INDEX_BUFFER_HPP
#define RENDERING_RTGI_INDEX_BUFFER_HPP

#include "core/RefCounted.hpp"

#include "rendering/rtgi/RTGI.hpp"

namespace rendering
{

namespace rtgi
{

class IndexBuffer : public core::RefCounted
{
public:
    virtual void Bind()                                            const = 0;
    virtual void Unbind()                                          const = 0;
    virtual void Render( PrimitiveRenderMode primitiveRenderMode ) const = 0;

protected:
    IndexBuffer() {};
    virtual ~IndexBuffer() {};
};


}

}

#endif