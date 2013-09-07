#ifndef RENDERING_RTGI_OPENGL_PIXEL_BUFFER_HPP
#define RENDERING_RTGI_OPENGL_PIXEL_BUFFER_HPP

#include "core/RefCounted.hpp"

#include "rendering/rtgi/PixelBuffer.hpp"

namespace rendering
{

namespace rtgi
{

class OpenGLPixelBuffer : public PixelBuffer
{
public:
    OpenGLPixelBuffer( const unsigned int numBytes );
    virtual ~OpenGLPixelBuffer();

    virtual unsigned int GetOpenGLPixelBufferID() const;

    virtual void Bind() const;
    virtual void Unbind() const;

private:
    unsigned int mPixelBufferID;
};

}

}

#endif