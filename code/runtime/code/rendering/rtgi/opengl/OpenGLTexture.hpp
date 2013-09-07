#ifndef RENDERING_RTGI_OPENGL_TEXTURE_HPP
#define RENDERING_RTGI_OPENGL_TEXTURE_HPP

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

#include "rendering/rtgi/Texture.hpp"

namespace core
{
    class String;
}

namespace rendering
{

namespace rtgi
{

class PixelBuffer;

GLenum  GetOpenGLTexturePixelFormat        ( TexturePixelFormat              texturePixelFormat );
GLenum  GetOpenGLTextureInternalPixelFormat( TexturePixelFormat              texturePixelFormat );
GLenum  GetOpenGLDataType                  ( TexturePixelFormat              texturePixelFormat );
GLenum  GetOpenGLTextureTarget             ( TextureDimensions               textureDimensions );
GLuint  GetOpenGLNumComponents             ( TexturePixelFormat              texturePixelFormat );
GLuint  GetOpenGLNumBytesPerComponent      ( TexturePixelFormat              texturePixelFormat );
GLfloat GetOpenGLWrapMode                  ( TextureSamplerWrapMode          textureSamplerWrapMode );
GLfloat GetOpenGLMinFilter                 ( TextureSamplerInterpolationMode textureSamplerInterpolationMode );
GLfloat GetOpenGLMagFilter                 ( TextureSamplerInterpolationMode textureSamplerInterpolationMode );
GLenum  GetOpenGLGenerateMipMaps           ( bool                            textureGenerateMipMaps );

class OpenGLTexture : public Texture
{
public:
    OpenGLTexture( const TextureDataDesc& textureDesc );
    
    virtual ~OpenGLTexture();

    virtual void Bind( const TextureSamplerStateDesc& textureSamplerStateDesc ) const;
    virtual void Unbind()                                                       const;

    virtual void Update( void* data )                     const;
    virtual void Update( const PixelBuffer* pixelBuffer ) const;
    

    virtual void  GenerateMipmaps()                                                                            const;
    virtual void* GetMipLevel( int mipLevel )                                                                  const;
    virtual void  SetMipLevelRegion( void* data, int mipLevel, int beginX, int beginY, int width, int height ) const;

    GLuint           GetOpenGLTextureID()                  const;
    GLenum           GetOpenGLTextureTarget()              const;
    GLenum           GetOpenGLTexturePixelFormat()         const;
    GLenum           GetOpenGLTextureInternalPixelFormat() const;
    GLenum           GetOpenGLTextureDataType()            const;
    TextureDataDesc  GetTextureDataDesc()                  const;
    
private:
    TextureDataDesc              mTextureDataDesc;
    GLuint                       mOpenGLTextureID;
    GLenum                       mOpenGLPixelFormat;
    GLenum                       mOpenGLInternalPixelFormat;
    GLenum                       mOpenGLDataType;
    GLenum                       mOpenGLTextureTarget;
};

class OpenGLBufferTexture : public BufferTexture
{
public:
    OpenGLBufferTexture( const BufferTextureDataDesc& textureDesc );

    virtual ~OpenGLBufferTexture();

    virtual void Bind()   const;
    virtual void Unbind() const;

    GLuint                GetOpenGLBufferID()                   const;    
    GLuint                GetOpenGLTextureID()                  const;
    GLenum                GetOpenGLTextureTarget()              const;
    GLenum                GetOpenGLTexturePixelFormat()         const;
    GLenum                GetOpenGLTextureInternalPixelFormat() const;
    GLenum                GetOpenGLTextureDataType()            const;
    BufferTextureDataDesc GetBufferTextureDataDesc()            const;

private:
    BufferTextureDataDesc mBufferTextureDataDesc;
    GLuint                mOpenGLBufferID;
    GLuint                mOpenGLTextureID;
    GLenum                mOpenGLPixelFormat;
    GLenum                mOpenGLInternalPixelFormat;
    GLenum                mOpenGLDataType;
};

}

}

#endif