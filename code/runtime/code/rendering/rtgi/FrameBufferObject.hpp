#ifndef RENDERING_RTGI_FRAME_BUFFER_OBJECT_HPP
#define RENDERING_RTGI_FRAME_BUFFER_OBJECT_HPP

#include "core/RefCounted.hpp"

#include "rendering/rtgi/Texture.hpp"

namespace rendering
{

namespace rtgi
{

class FrameBufferObject : public core::RefCounted
{
public:
    virtual void Bind( Texture* texture )                                          = 0;
    virtual void Bind( Texture* texture, unsigned int width, unsigned int height ) = 0;
    virtual void Unbind()                                                          = 0;

protected:
    FrameBufferObject() {};
    virtual ~FrameBufferObject() {};
};

}

}

#endif