#include "rendering/rtgi/OpenGL/OpenGLFrameBufferObject_RGBA32F.hpp"

#if defined PLATFORM_WIN32
#define NOMINMAX
#include <windows.h>
#endif

#include "core/Assert.hpp"

#include "rendering/rtgi/opengl/Extensions.hpp"
#include "rendering/rtgi/opengl/OpenGLTexture.hpp"

#include "rendering/rtgi/RTGI.hpp"


namespace rendering
{

namespace rtgi
{

OpenGLFrameBufferObject_RGBA32F::OpenGLFrameBufferObject_RGBA32F() : OpenGLFrameBufferObject()
{

}

OpenGLFrameBufferObject_RGBA32F::~OpenGLFrameBufferObject_RGBA32F()
{
}

void OpenGLFrameBufferObject_RGBA32F::Bind( Texture* texture )
{
	OpenGLTexture* openGLTexture = dynamic_cast< OpenGLTexture* >( texture );
	//int            viewport[ 4 ];

	//
	// get old viewport
	//
	//glGetIntegerv( GL_VIEWPORT, viewport );
    rtgi::GetViewport( mViewportWidth, mViewportHeight );

	//
	// set up color texture
	//
	glBindTexture( GL_TEXTURE_2D, openGLTexture->GetOpenGLTextureID() );

	glTexImage2D( GL_TEXTURE_2D, 0, GL_LUMINANCE32F_ARB, mViewportWidth, mViewportHeight, 0, GL_LUMINANCE, GL_FLOAT, NULL );

	// when texture area is small, bilinear filter the closest mipmap
	glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );

	// when texture area is large, bilinear filter the original
	glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );

	// the texture wraps over at the edges (repeat)
	glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP );
	glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP );

	/***
	NOTE: Since this format was hacked into the system for a test we were doing and we didn't
	need the depth stencil for it we took this out as it would slow down our tests and botch
	our test results.
	***/
	//
	// set up combined depth stencil texture
	//
	/*
	glBindTexture( GL_TEXTURE_2D, getDepthStencilBufferTextureID() );

	glTexImage2D( GL_TEXTURE_2D, 0, GL_DEPTH24_STENCIL8_EXT, viewport[ 2 ], viewport[ 3 ], 0, GL_DEPTH_STENCIL_EXT, GL_UNSIGNED_INT_24_8_EXT, NULL );

	// when texture area is small, bilinear filter the closest mipmap
	glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );

	// when texture area is large, bilinear filter the original
	glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );

	// the texture wraps over at the edges (repeat)
	glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP );
	glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP );
	*/
	glBindTexture( GL_TEXTURE_2D, 0 );
	
	//
	// set up frame buffer object
	//
	glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, getFrameBufferObjectID() );

	glFramebufferTexture2DEXT( GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT,  GL_TEXTURE_2D, openGLTexture->GetOpenGLTextureID(), 0 );
	// Taken out for tests.
	//glFramebufferTexture2DEXT( GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT,   GL_TEXTURE_2D, getDepthStencilBufferTextureID(),  0 );
	//glFramebufferTexture2DEXT( GL_FRAMEBUFFER_EXT, GL_STENCIL_ATTACHMENT_EXT, GL_TEXTURE_2D, getDepthStencilBufferTextureID(),  0 );

	GLenum buffers[] = { GL_COLOR_ATTACHMENT0_EXT };
	glDrawBuffersARB(1, buffers);

	CheckErrors();
}

void OpenGLFrameBufferObject_RGBA32F::Bind(Texture* textureOne, Texture* textureTwo)
{
	OpenGLTexture* openGLTextureOne = dynamic_cast< OpenGLTexture* >( textureOne );
	OpenGLTexture* openGLTextureTwo = dynamic_cast< OpenGLTexture* >( textureTwo );
	int            viewport[ 4 ];

	//
	// get old viewport
	//
	glGetIntegerv( GL_VIEWPORT, viewport );

	//
	// set up color texture
	//
	glBindTexture( GL_TEXTURE_2D, openGLTextureOne->GetOpenGLTextureID() );

	glTexImage2D( GL_TEXTURE_2D, 0, GL_LUMINANCE32F_ARB, viewport[ 2 ], viewport[ 3 ], 0, GL_LUMINANCE, GL_FLOAT, NULL );

	// when texture area is small, bilinear filter the closest mipmap
	glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );

	// when texture area is large, bilinear filter the original
	glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );

	// the texture wraps over at the edges (repeat)
	glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP );
	glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP );

	glBindTexture( GL_TEXTURE_2D, openGLTextureTwo->GetOpenGLTextureID() );

	glTexImage2D( GL_TEXTURE_2D, 0, GL_LUMINANCE32F_ARB, viewport[ 2 ], viewport[ 3 ], 0, GL_LUMINANCE, GL_FLOAT, NULL );

	// when texture area is small, bilinear filter the closest mipmap
	glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );

	// when texture area is large, bilinear filter the original
	glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );

	// the texture wraps over at the edges (repeat)
	glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP );
	glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP );

	/***
	NOTE: Since this format was hacked into the system for a test we were doing and we didn't
	need the depth stencil for it we took this out as it would slow down our tests and botch
	our test results.
	***/
	//
	// set up combined depth stencil texture
	//
	/*
	glBindTexture( GL_TEXTURE_2D, getDepthStencilBufferTextureID() );

	glTexImage2D( GL_TEXTURE_2D, 0, GL_DEPTH24_STENCIL8_EXT, viewport[ 2 ], viewport[ 3 ], 0, GL_DEPTH_STENCIL_EXT, GL_UNSIGNED_INT_24_8_EXT, NULL );

	// when texture area is small, bilinear filter the closest mipmap
	glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );

	// when texture area is large, bilinear filter the original
	glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );

	// the texture wraps over at the edges (repeat)
	glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP );
	glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP );
	*/
	glBindTexture( GL_TEXTURE_2D, 0 );

	//
	// set up frame buffer object
	//
	glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, getFrameBufferObjectID() );

	glFramebufferTexture2DEXT( GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT,  GL_TEXTURE_2D, openGLTextureOne->GetOpenGLTextureID(), 0 );
	glFramebufferTexture2DEXT( GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT1_EXT,  GL_TEXTURE_2D, openGLTextureTwo->GetOpenGLTextureID(), 0 );

	GLenum buffers[] = { GL_COLOR_ATTACHMENT0_EXT, GL_COLOR_ATTACHMENT1_EXT };
	glDrawBuffersARB(2, buffers);

	// Taken out for tests.
	//glFramebufferTexture2DEXT( GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT,   GL_TEXTURE_2D, getDepthStencilBufferTextureID(),  0 );
	//glFramebufferTexture2DEXT( GL_FRAMEBUFFER_EXT, GL_STENCIL_ATTACHMENT_EXT, GL_TEXTURE_2D, getDepthStencilBufferTextureID(),  0 );

	CheckErrors();
}

void OpenGLFrameBufferObject_RGBA32F::Unbind() const
{
    rtgi::SetViewport( mViewportWidth, mViewportHeight );

    glFramebufferTexture2DEXT( GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT,  GL_TEXTURE_2D, 0, 0 );
    glFramebufferTexture2DEXT( GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT1_EXT,  GL_TEXTURE_2D, 0, 0 );
    glFramebufferTexture2DEXT( GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT,   GL_TEXTURE_2D, 0, 0 );
    glFramebufferTexture2DEXT( GL_FRAMEBUFFER_EXT, GL_STENCIL_ATTACHMENT_EXT, GL_TEXTURE_2D, 0, 0 );

    glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, 0 );

    CheckErrors();
}

}

}
