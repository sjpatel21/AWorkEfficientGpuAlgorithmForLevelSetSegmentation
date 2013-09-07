#ifndef RENDERING_RTGI_OPENGL_FRAME_BUFFER_OBJECT_RGBA32F_HPP
#define RENDERING_RTGI_OPENGL_FRAME_BUFFER_OBJECT_RGBA32F_HPP

#include "rendering/rtgi/opengl/OpenGLFrameBufferObject.hpp"

namespace lefohn
{

class OpenGLFrameBufferObjectRGBA32F : public rendering::rtgi::OpenGLFrameBufferObject
{
public:
    OpenGLFrameBufferObjectRGBA32F();
    virtual ~OpenGLFrameBufferObjectRGBA32F();

    virtual void Bind( rendering::rtgi::Texture* texture );
    virtual void Bind( rendering::rtgi::Texture* textureOne, rendering::rtgi::Texture* textureTwo );
    virtual void Unbind() const;

private:
    int mViewportHeight;
    int mViewportWidth;
};

}

#endif
