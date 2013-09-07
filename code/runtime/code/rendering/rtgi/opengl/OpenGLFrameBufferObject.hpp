#ifndef RENDERING_RTGI_OPENGL_FRAME_BUFFER_OBJECT_HPP
#define RENDERING_RTGI_OPENGL_FRAME_BUFFER_OBJECT_HPP


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

#include "core/RefCounted.hpp"

#include "rendering/rtgi/FrameBufferObject.hpp"
#include "rendering/rtgi/opengl/OpenGLTexture.hpp"

class QString;

namespace rendering
{

namespace rtgi
{

class OpenGLFrameBufferObject : public FrameBufferObject
{
public:
    OpenGLFrameBufferObject();

    virtual ~OpenGLFrameBufferObject();

    virtual void Bind( Texture* texture );
    virtual void Bind( Texture* texture, unsigned int width, unsigned int height );
    virtual void Unbind();

protected:
    unsigned int getFrameBufferObjectID() const {return mFrameBufferObjectID;}
    unsigned int getDepthStencilBufferTextureID() const {return mDepthStencilBufferTextureID;}

private:
    OpenGLTexture* mCurrentlyBoundOpenGLTexture;

    unsigned int mFrameBufferObjectID;
    unsigned int mDepthStencilBufferTextureID;

    // HACK - should be unsigned
    int mViewportHeight;
    int mViewportWidth;
};

}

}

#endif