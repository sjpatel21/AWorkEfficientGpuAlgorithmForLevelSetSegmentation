#ifndef RENDERING_RTGI_PIXEL_BUFFER_HPP
#define RENDERING_RTGI_PIXEL_BUFFER_HPP

#include "core/RefCounted.hpp"

namespace rendering
{

namespace rtgi
{

class PixelBuffer : public core::RefCounted
{
public:

    virtual void Bind()   const = 0;
    virtual void Unbind() const = 0;

protected:
    PixelBuffer() {};
    virtual ~PixelBuffer() {};
};

}

}

#endif