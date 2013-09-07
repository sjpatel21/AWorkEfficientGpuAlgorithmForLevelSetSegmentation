#include "rendering/rtgi/opengl/OpenGLFrameBufferObject.hpp"

#include "core/Assert.hpp"

#include "rendering/rtgi/opengl/Extensions.hpp"
#include "rendering/rtgi/opengl/OpenGLTexture.hpp"

#include "rendering/rtgi/RTGI.hpp"

namespace rendering
{

namespace rtgi
{

OpenGLFrameBufferObject::OpenGLFrameBufferObject() :
mCurrentlyBoundOpenGLTexture( NULL ),
mFrameBufferObjectID        ( 0 ),
mDepthStencilBufferTextureID( 0 )
{
    glGenFramebuffersEXT(  1, &mFrameBufferObjectID );
    glGenTextures( 1, &mDepthStencilBufferTextureID );

    CheckErrors();
}

OpenGLFrameBufferObject::~OpenGLFrameBufferObject()
{
    glDeleteTextures( 1, &mDepthStencilBufferTextureID );
    glDeleteFramebuffersEXT(  1, &mFrameBufferObjectID );

    CheckErrors();
}

void OpenGLFrameBufferObject::Bind( Texture* texture, unsigned int width, unsigned int height )
{    
    OpenGLTexture* openGLTexture       = dynamic_cast< OpenGLTexture* >( texture );
    GLenum         internalPixelFormat = openGLTexture->GetOpenGLTextureInternalPixelFormat();
    GLenum         pixelFormat         = openGLTexture->GetOpenGLTexturePixelFormat();
    GLenum         dataType            = openGLTexture->GetOpenGLTextureDataType();

    Assert( mCurrentlyBoundOpenGLTexture == NULL && openGLTexture != NULL );
    AssignRef( mCurrentlyBoundOpenGLTexture, openGLTexture );

    switch( openGLTexture->GetTextureDataDesc().dimensions )
    {
        case TextureDimensions_2D:

            // save the old viewport
            rtgi::GetViewport( mViewportWidth, mViewportHeight );

            // set up color texture
            glBindTexture( GL_TEXTURE_2D, openGLTexture->GetOpenGLTextureID() );
            glTexImage2D( GL_TEXTURE_2D, 0, internalPixelFormat, width, height, 0, pixelFormat, dataType, NULL );

            // when texture area is small, bilinear filter the closest mipmap
            glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );

            // when texture area is large, bilinear filter the original
            glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );

            // the texture wraps over at the edges
            glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP );
            glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP );


            // set up combined depth stencil texture
            glBindTexture( GL_TEXTURE_2D, mDepthStencilBufferTextureID );
            glTexImage2D( GL_TEXTURE_2D, 0, GL_DEPTH24_STENCIL8_EXT, width, height, 0, GL_DEPTH_STENCIL_EXT, GL_UNSIGNED_INT_24_8_EXT, NULL );

            // when texture area is small, bilinear filter the closest mipmap
            glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );

            // when texture area is large, bilinear filter the original
            glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );

            // the texture wraps over at the edges
            glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP );
            glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP );

            // set up frame buffer object
            glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, mFrameBufferObjectID );

            glFramebufferTexture2DEXT( GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT,  GL_TEXTURE_2D, openGLTexture->GetOpenGLTextureID(), 0 );
            glFramebufferTexture2DEXT( GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT,   GL_TEXTURE_2D, mDepthStencilBufferTextureID,        0 );
            glFramebufferTexture2DEXT( GL_FRAMEBUFFER_EXT, GL_STENCIL_ATTACHMENT_EXT, GL_TEXTURE_2D, mDepthStencilBufferTextureID,        0 );

            // set the new viewport
            rtgi::SetViewport( width, height );
            rtgi::SetFrameBufferObjectBound( true );

            CheckErrors();

            break;

        case TextureDimensions_3D:

            // save the old viewport
            rtgi::GetViewport( mViewportWidth, mViewportHeight );

            // set up color texture
            glBindTexture( GL_TEXTURE_3D, openGLTexture->GetOpenGLTextureID() );

            // when texture area is small, bilinear filter the closest mipmap
            glTexParameterf( GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );

            // when texture area is large, bilinear filter the original
            glTexParameterf( GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );

            // the texture wraps over at the edges
            glTexParameterf( GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP );
            glTexParameterf( GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP );
            glTexParameterf( GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP );

            // set up the frame buffer object
            glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, mFrameBufferObjectID );
            glFramebufferTextureEXT( GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, openGLTexture->GetOpenGLTextureID(), 0 );

            // set the new viewport
            rtgi::SetViewport( openGLTexture->GetTextureDataDesc().width, openGLTexture->GetTextureDataDesc().height );
            rtgi::SetFrameBufferObjectBound( true );

            CheckErrors();

            break;

        default:
            Assert( 0 );
            break;
    }
}

void OpenGLFrameBufferObject::Bind( Texture* texture )
{
    int width, height;
    rtgi::GetViewport( width, height );

    Bind( texture, width, height );
}

void OpenGLFrameBufferObject::Unbind()
{
    Assert( mCurrentlyBoundOpenGLTexture != NULL );

    switch( mCurrentlyBoundOpenGLTexture->GetTextureDataDesc().dimensions )
    {
        case TextureDimensions_2D:

            // detach the frame buffer object
            glFramebufferTexture2DEXT( GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT,  GL_TEXTURE_2D, 0, 0 );
            glFramebufferTexture2DEXT( GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT,   GL_TEXTURE_2D, 0, 0 );
            glFramebufferTexture2DEXT( GL_FRAMEBUFFER_EXT, GL_STENCIL_ATTACHMENT_EXT, GL_TEXTURE_2D, 0, 0 );

            glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, 0 );

            // restore the view port
            rtgi::SetViewport( mViewportWidth, mViewportHeight );
            rtgi::SetFrameBufferObjectBound( false );

            CheckErrors();

            break;

        case TextureDimensions_3D:

            // detach the frame buffer object
            glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, 0 );

            // detach color texture
            glBindTexture( GL_TEXTURE_3D, 0 );

            // restore the view port
            rtgi::SetViewport( mViewportWidth, mViewportHeight );
            rtgi::SetFrameBufferObjectBound( false );

            CheckErrors();

            break;

        default:
            Assert( 0 );
            break;
    }

    AssignRef( mCurrentlyBoundOpenGLTexture, NULL );
}

}

}