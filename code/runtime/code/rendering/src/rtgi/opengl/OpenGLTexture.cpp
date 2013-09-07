#include "rendering/rtgi/opengl/OpenGLTexture.hpp"

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

#include "core/String.hpp"
#include "core/Assert.hpp"

#include "rendering/rtgi/RTGI.hpp"

#include "rendering/rtgi/opengl/Extensions.hpp"
#include "rendering/rtgi/opengl/OpenGLPixelBuffer.hpp"

namespace rendering
{

namespace rtgi
{

GLenum GetOpenGLTexturePixelFormat( TexturePixelFormat texturePixelFormat )
{
    switch ( texturePixelFormat )
    {
    case TexturePixelFormat_R8_UI_DENORM:
    case TexturePixelFormat_R8_I_DENORM:
    case TexturePixelFormat_R16_UI_DENORM:
    case TexturePixelFormat_R16_I_DENORM:
    case TexturePixelFormat_R32_UI_DENORM:
    case TexturePixelFormat_R32_I_DENORM:
#if defined(PLATFORM_WIN32)
        return GL_RED_INTEGER;
#elif defined(PLATFORM_OSX)
        return GL_RED_INTEGER_EXT;
#endif    

    case TexturePixelFormat_A16_F_DENORM:
    case TexturePixelFormat_A32_F_DENORM:
        return GL_ALPHA;

    case TexturePixelFormat_R8_G8_UI_DENORM:
        return GL_RG_INTEGER;

    case TexturePixelFormat_R16_G16_B16_A16_UI_DENORM:
    case TexturePixelFormat_R16_G16_B16_A16_I_DENORM:
    case TexturePixelFormat_R32_G32_B32_A32_UI_DENORM:
    case TexturePixelFormat_R32_G32_B32_A32_I_DENORM:
#if defined(PLATFORM_WIN32) // Until mac osx supports more recent OpenGL
        return GL_RGBA_INTEGER;
#elif defined(PLATFORM_OSX)
        return GL_RGBA_INTEGER_EXT;
#endif

    case TexturePixelFormat_R16_G16_B16_A16_F_DENORM:
    case TexturePixelFormat_R32_G32_B32_A32_F_DENORM:
        return GL_RGBA;

    case TexturePixelFormat_A8_UI_NORM:
    case TexturePixelFormat_A8_I_NORM:
        return GL_ALPHA;

    case TexturePixelFormat_R8_G8_B8_UI_NORM:
        return GL_RGB;

    case TexturePixelFormat_R8_G8_B8_A8_UI_NORM:
    case TexturePixelFormat_R8_G8_B8_A8_I_NORM:
        return GL_RGBA;

    case TexturePixelFormat_L32_F_DENORM:
        return GL_LUMINANCE;

    default:
        Assert( 0 );
        return -1;
    }
}

GLenum GetOpenGLTextureInternalPixelFormat( TexturePixelFormat texturePixelFormat )
{
    switch ( texturePixelFormat )
    {
    case TexturePixelFormat_R8_UI_DENORM:              return GL_R8UI;
    case TexturePixelFormat_R8_I_DENORM:               return GL_R8I;
    case TexturePixelFormat_R16_UI_DENORM:             return GL_R16UI;
    case TexturePixelFormat_R16_I_DENORM:              return GL_R16I;
    case TexturePixelFormat_R32_UI_DENORM:             return GL_R32UI;
    case TexturePixelFormat_R32_I_DENORM:              return GL_R32I;
                                               
    case TexturePixelFormat_A16_F_DENORM:              return GL_ALPHA16F_ARB;
    case TexturePixelFormat_A32_F_DENORM:              return GL_ALPHA32F_ARB;
                                               
    case TexturePixelFormat_R8_G8_UI_DENORM:           return GL_RG8UI;
    
#if defined(PLATFORM_WIN32)
    case TexturePixelFormat_R16_G16_B16_A16_UI_DENORM: return GL_RGBA16UI;
    case TexturePixelFormat_R16_G16_B16_A16_I_DENORM:  return GL_RGBA16I;

    case TexturePixelFormat_R32_G32_B32_A32_UI_DENORM: return GL_RGBA32UI;
    case TexturePixelFormat_R32_G32_B32_A32_I_DENORM:  return GL_RGBA32I;

    case TexturePixelFormat_R16_G16_B16_A16_F_DENORM:  return GL_RGBA16F;
    case TexturePixelFormat_R32_G32_B32_A32_F_DENORM:  return GL_RGBA32F;
#elif defined(PLATFORM_OSX) // Until mac osx supports more recent OpenGL
    case TexturePixelFormat_R16_G16_B16_A16_UI_DENORM: return GL_RGBA16UI_EXT;
    case TexturePixelFormat_R16_G16_B16_A16_I_DENORM:  return GL_RGBA16I_EXT;
            
    case TexturePixelFormat_R32_G32_B32_A32_UI_DENORM: return GL_RGBA32UI_EXT;
    case TexturePixelFormat_R32_G32_B32_A32_I_DENORM:  return GL_RGBA32I_EXT;
            
    case TexturePixelFormat_R16_G16_B16_A16_F_DENORM:  return GL_RGBA16F_ARB;
    case TexturePixelFormat_R32_G32_B32_A32_F_DENORM:  return GL_RGBA32F_ARB;        
#endif

    case TexturePixelFormat_A8_UI_NORM:                return GL_ALPHA8;
#if defined(PLATFORM_WIN32)
    case TexturePixelFormat_A8_I_NORM:                 return GL_SIGNED_ALPHA8_NV;
#elif defined(PLATFORM_OSX)
    // No perfect replacement on OSX
    case TexturePixelFormat_A8_I_NORM:                 return GL_ALPHA8;
#endif

    case TexturePixelFormat_R8_G8_B8_UI_NORM:          return GL_RGB8;
    case TexturePixelFormat_R8_G8_B8_A8_UI_NORM:       return GL_RGBA8;
#if defined(PLATFORM_WIN32)
    case TexturePixelFormat_R8_G8_B8_A8_I_NORM:        return GL_SIGNED_RGBA8_NV;
#elif defined(PLATFORM_OSX)
    // No current replacement
#endif

    case TexturePixelFormat_L32_F_DENORM:              return GL_LUMINANCE32F_ARB;

    default:
        Assert( 0 );
        return -1;
    }
}

GLenum GetOpenGLDataType( TexturePixelFormat texturePixelFormat )
{
    switch ( texturePixelFormat )
    {
    case TexturePixelFormat_R8_UI_DENORM:              return GL_UNSIGNED_BYTE;
    case TexturePixelFormat_R8_I_DENORM:               return GL_BYTE;
    case TexturePixelFormat_R16_UI_DENORM:             return GL_UNSIGNED_SHORT;
    case TexturePixelFormat_R16_I_DENORM:              return GL_SHORT;
    case TexturePixelFormat_R32_UI_DENORM:             return GL_UNSIGNED_INT;
    case TexturePixelFormat_R32_I_DENORM:              return GL_INT;

    case TexturePixelFormat_A16_F_DENORM:              return GL_HALF_FLOAT;
    case TexturePixelFormat_A32_F_DENORM:              return GL_FLOAT;

    case TexturePixelFormat_R8_G8_UI_DENORM:           return GL_UNSIGNED_BYTE;

    case TexturePixelFormat_R16_G16_B16_A16_UI_DENORM: return GL_UNSIGNED_SHORT;
    case TexturePixelFormat_R16_G16_B16_A16_I_DENORM:  return GL_SHORT;

    case TexturePixelFormat_R32_G32_B32_A32_UI_DENORM: return GL_UNSIGNED_INT;
    case TexturePixelFormat_R32_G32_B32_A32_I_DENORM:  return GL_INT;

    case TexturePixelFormat_R16_G16_B16_A16_F_DENORM:  return GL_HALF_FLOAT;
    case TexturePixelFormat_R32_G32_B32_A32_F_DENORM:  return GL_FLOAT;

    case TexturePixelFormat_A8_UI_NORM:                return GL_UNSIGNED_BYTE;
    case TexturePixelFormat_A8_I_NORM:                 return GL_BYTE;
    case TexturePixelFormat_R8_G8_B8_UI_NORM:          return GL_UNSIGNED_BYTE;
    case TexturePixelFormat_R8_G8_B8_A8_UI_NORM:       return GL_UNSIGNED_BYTE;
    case TexturePixelFormat_R8_G8_B8_A8_I_NORM:        return GL_BYTE;

    case TexturePixelFormat_L32_F_DENORM:              return GL_FLOAT;

    default:
        Assert( 0 );
        return -1;
    }
}

GLenum GetOpenGLTextureTarget( TextureDimensions textureDimensions )
{
    switch ( textureDimensions )
    {
    case TextureDimensions_1D:
        return GL_TEXTURE_1D;
        break;

    case TextureDimensions_2D:
        return GL_TEXTURE_2D;
        break;

    case TextureDimensions_3D:
        return GL_TEXTURE_3D;
        break;

    default:
        Assert( 0 );
        return -1;
    }
}

GLuint GetOpenGLNumComponents( TexturePixelFormat texturePixelFormat )
{
    switch ( texturePixelFormat )
    {
    case TexturePixelFormat_R8_UI_DENORM:
    case TexturePixelFormat_R8_I_DENORM:
    case TexturePixelFormat_R16_UI_DENORM:
    case TexturePixelFormat_R16_I_DENORM:
    case TexturePixelFormat_R32_UI_DENORM:
    case TexturePixelFormat_R32_I_DENORM:
        return 1;

    case TexturePixelFormat_A16_F_DENORM:
    case TexturePixelFormat_A32_F_DENORM:
    case TexturePixelFormat_L32_F_DENORM:
        return 1;

    case TexturePixelFormat_R8_G8_UI_DENORM:
        return 2;

    case TexturePixelFormat_R16_G16_B16_A16_UI_DENORM:
    case TexturePixelFormat_R16_G16_B16_A16_I_DENORM:
    case TexturePixelFormat_R32_G32_B32_A32_UI_DENORM:
    case TexturePixelFormat_R32_G32_B32_A32_I_DENORM:
        return 4;

    case TexturePixelFormat_R16_G16_B16_A16_F_DENORM:
    case TexturePixelFormat_R32_G32_B32_A32_F_DENORM:
        return 4;

    case TexturePixelFormat_A8_UI_NORM:
        return 1;

    case TexturePixelFormat_A8_I_NORM:
        return 1;

    case TexturePixelFormat_R8_G8_B8_UI_NORM:
        return 3;

    case TexturePixelFormat_R8_G8_B8_A8_UI_NORM:
    case TexturePixelFormat_R8_G8_B8_A8_I_NORM:
        return 4;

    default:
        Assert( 0 );
        return -1;
    }
}

GLuint GetOpenGLNumBytesPerComponent( TexturePixelFormat texturePixelFormat )
{
    switch ( texturePixelFormat )
    {
    case TexturePixelFormat_R8_UI_DENORM:
    case TexturePixelFormat_R8_I_DENORM:
    case TexturePixelFormat_R8_G8_UI_DENORM:
    case TexturePixelFormat_A8_UI_NORM:
    case TexturePixelFormat_A8_I_NORM:
    case TexturePixelFormat_R8_G8_B8_UI_NORM:
    case TexturePixelFormat_R8_G8_B8_A8_UI_NORM:
    case TexturePixelFormat_R8_G8_B8_A8_I_NORM:
        return 1;

    case TexturePixelFormat_R16_UI_DENORM:
    case TexturePixelFormat_R16_I_DENORM:
    case TexturePixelFormat_A16_F_DENORM:
    case TexturePixelFormat_R16_G16_B16_A16_UI_DENORM:
    case TexturePixelFormat_R16_G16_B16_A16_I_DENORM:
    case TexturePixelFormat_R16_G16_B16_A16_F_DENORM:
        return 2;

    case TexturePixelFormat_R32_UI_DENORM:
    case TexturePixelFormat_R32_I_DENORM:
    case TexturePixelFormat_A32_F_DENORM:
    case TexturePixelFormat_R32_G32_B32_A32_UI_DENORM:
    case TexturePixelFormat_R32_G32_B32_A32_I_DENORM:
    case TexturePixelFormat_R32_G32_B32_A32_F_DENORM:
    case TexturePixelFormat_L32_F_DENORM:
        return 4;

    default:
        Assert( 0 );
        return -1;
    }
}

GLfloat GetOpenGLWrapMode( TextureSamplerWrapMode textureSamplerWrapMode )
{
    switch ( textureSamplerWrapMode )
    {
    case TextureSamplerWrapMode_Clamp:
        return GL_CLAMP;

    case TextureSamplerWrapMode_ClampToEdge:
        return GL_CLAMP_TO_EDGE;

    case TextureSamplerWrapMode_Repeat:
        return GL_REPEAT;

    default:
        Assert( 0 );
        return -1;
    }
}

GLfloat GetOpenGLMinFilter( TextureSamplerInterpolationMode textureSamplerInterpolationMode )
{
    switch ( textureSamplerInterpolationMode )
    {
    case TextureSamplerInterpolationMode_Nearest:
        return GL_NEAREST;
        break;

    case TextureSamplerInterpolationMode_Smooth:
        return GL_LINEAR;
        break;

    case TextureSamplerInterpolationMode_SmoothMipMaps:
        return GL_LINEAR_MIPMAP_LINEAR;
        break;

    default:
        Assert( 0 );
        return -1;
    }
}

GLfloat GetOpenGLMagFilter( TextureSamplerInterpolationMode textureSamplerInterpolationMode )
{
    switch ( textureSamplerInterpolationMode )
    {
    case TextureSamplerInterpolationMode_Nearest:
        return GL_NEAREST;
        break;

    case TextureSamplerInterpolationMode_Smooth:
        return GL_LINEAR;
        break;

    case TextureSamplerInterpolationMode_SmoothMipMaps:
        return GL_LINEAR;
        break;

    default:
        Assert( 0 );
        return -1;
    }
}

GLenum GetOpenGLGenerateMipMaps( bool textureGenerateMipMaps )
{
    if ( textureGenerateMipMaps )
    {
        return GL_TRUE;
    }
    else
    {
        return GL_FALSE;
    }
}

TextureDataDesc::TextureDataDesc() :
dimensions                  ( TextureDimensions_Invalid ),
pixelFormat                 ( TexturePixelFormat_Invalid ),
width                       ( -1 ),
height                      ( -1 ),
depth                       ( -1 ),
data                        ( NULL ),
generateMipMaps             ( false )
{
}

TextureSamplerStateDesc::TextureSamplerStateDesc() :
textureSamplerWrapMode         ( TextureSamplerWrapMode_Invalid ),
textureSamplerInterpolationMode( TextureSamplerInterpolationMode_Invalid )
{
}

BufferTextureDataDesc::BufferTextureDataDesc() :
numBytes   ( -1 ),
pixelFormat( TexturePixelFormat_Invalid ),
data       ( NULL )
{
}

OpenGLTexture::OpenGLTexture( const TextureDataDesc& textureDataDesc ) :
mTextureDataDesc          ( textureDataDesc ),
mOpenGLTextureID          ( -1 ),
mOpenGLPixelFormat        ( -1 ),
mOpenGLInternalPixelFormat( -1 ),
mOpenGLDataType           ( -1 ),
mOpenGLTextureTarget      ( -1 )
{
    mOpenGLPixelFormat          = rtgi::GetOpenGLTexturePixelFormat        ( textureDataDesc.pixelFormat );
    mOpenGLInternalPixelFormat  = rtgi::GetOpenGLTextureInternalPixelFormat( textureDataDesc.pixelFormat );
    mOpenGLDataType             = rtgi::GetOpenGLDataType                  ( textureDataDesc.pixelFormat );
    mOpenGLTextureTarget        = rtgi::GetOpenGLTextureTarget             ( textureDataDesc.dimensions );
    GLuint numComponents        = rtgi::GetOpenGLNumComponents             ( textureDataDesc.pixelFormat );
    GLuint numBytesPerComponent = rtgi::GetOpenGLNumBytesPerComponent      ( textureDataDesc.pixelFormat );
    GLuint generateMipMaps      = rtgi::GetOpenGLGenerateMipMaps           ( textureDataDesc.generateMipMaps );

    // allocate texture id
    glGenTextures( 1, &mOpenGLTextureID );

    // bind texture
    glBindTexture( mOpenGLTextureTarget, mOpenGLTextureID );

    // generate mip maps
    glTexParameteri( mOpenGLTextureTarget, GL_GENERATE_MIPMAP, generateMipMaps );

    // upload data
    switch ( textureDataDesc.dimensions )
    {
        case TextureDimensions_1D:
            glTexImage1D(
                mOpenGLTextureTarget,                 // target
                0,                                    // mip level
                mOpenGLInternalPixelFormat,           // internal format
                textureDataDesc.width,                // num texels width
                0,                                    // border
                mOpenGLPixelFormat,                   // format
                mOpenGLDataType,                      // type
                textureDataDesc.data );               // data
            break;

        case TextureDimensions_2D:
            glTexImage2D(
                mOpenGLTextureTarget,                 // target
                0,                                    // mip level
                mOpenGLInternalPixelFormat,           // internal format
                textureDataDesc.width,                // num texels width
                textureDataDesc.height,               // num texels height
                0,                                    // border
                mOpenGLPixelFormat,                   // format
                mOpenGLDataType,                      // type
                textureDataDesc.data );               // data
            break;

        case TextureDimensions_3D:
            glTexImage3D(
                mOpenGLTextureTarget,                 // target
                0,                                    // mip level
                mOpenGLInternalPixelFormat,           // internal format
                textureDataDesc.width,                // num texels width
                textureDataDesc.height,               // num texels height
                textureDataDesc.depth,                // num texels depth
                0,                                    // border
                mOpenGLPixelFormat,                   // format
                mOpenGLDataType,                      // type
                textureDataDesc.data );               // data
            break;

        default:
            Assert( 0 );
            break;
    }

    glTexParameteri( mOpenGLTextureTarget, GL_GENERATE_MIPMAP, GL_FALSE );

    CheckErrors();
}

OpenGLTexture::~OpenGLTexture()
{
    glDeleteTextures( 1, &mOpenGLTextureID );

    CheckErrors();
}

void OpenGLTexture::Bind( const TextureSamplerStateDesc& textureSamplerStateDesc ) const
{
    // enable texture target
    glEnable( mOpenGLTextureTarget );

    GLenum openGLWrapMode    = rtgi::GetOpenGLWrapMode ( textureSamplerStateDesc.textureSamplerWrapMode );
    GLenum openGLMinFilter   = rtgi::GetOpenGLMinFilter( textureSamplerStateDesc.textureSamplerInterpolationMode );
    GLenum openGLMagFilter   = rtgi::GetOpenGLMagFilter( textureSamplerStateDesc.textureSamplerInterpolationMode );

    // specify active texture unit
    glClientActiveTextureARB( GL_TEXTURE0_ARB );

    // bind texture object to texture unit
    glBindTexture( mOpenGLTextureTarget, mOpenGLTextureID );

    // opengl multiplies the current color by the texture color value, so set the current color to white
    glColor3f( 1, 1, 1 );

    // when this texture needs to be shrunk to fit on small polygons
    glTexParameteri( mOpenGLTextureTarget, GL_TEXTURE_MIN_FILTER, openGLMinFilter );
    // when this texture needs to be magnified to fit on a big polygon
    glTexParameteri( mOpenGLTextureTarget, GL_TEXTURE_MAG_FILTER, openGLMagFilter );
    // when the texture coordinates are out of bounds
    glTexParameteri( mOpenGLTextureTarget, GL_TEXTURE_WRAP_S, openGLWrapMode );
    // same as above for T axis
    glTexParameteri( mOpenGLTextureTarget, GL_TEXTURE_WRAP_T, openGLWrapMode );
    // same as above for R axis
    glTexParameteri( mOpenGLTextureTarget, GL_TEXTURE_WRAP_R, openGLWrapMode );
 
    CheckErrors();
}

void OpenGLTexture::Unbind() const
{
    glBindTexture( mOpenGLTextureTarget, 0 );
    glDisable( mOpenGLTextureTarget );

    CheckErrors();
}

GLuint OpenGLTexture::GetOpenGLTextureID() const
{
    return mOpenGLTextureID;
}

GLenum OpenGLTexture::GetOpenGLTextureTarget() const
{
    return mOpenGLTextureTarget;
}

GLenum OpenGLTexture::GetOpenGLTexturePixelFormat() const 
{
    return mOpenGLPixelFormat;
}

GLenum OpenGLTexture::GetOpenGLTextureInternalPixelFormat() const 
{
    return mOpenGLInternalPixelFormat;
}

GLenum OpenGLTexture::GetOpenGLTextureDataType() const 
{
    return mOpenGLDataType;
}

TextureDataDesc OpenGLTexture::GetTextureDataDesc() const
{
    return mTextureDataDesc;
}

void OpenGLTexture::Update( void* data ) const
{
    glBindTexture( mOpenGLTextureTarget, mOpenGLTextureID );

    // upload data
    switch ( mTextureDataDesc.dimensions )
    {
        case TextureDimensions_1D:
            glTexSubImage1D(
                mOpenGLTextureTarget,                 // target
                0,                                    // mip level
                0,                                    // x offset
                mTextureDataDesc.width,               // num texels width
                mOpenGLPixelFormat,                   // format
                mOpenGLDataType,                      // type
                data );                               // data

            break;

        case TextureDimensions_2D:
            glTexSubImage2D(
                mOpenGLTextureTarget,                 // target
                0,                                    // mip level
                0,                                    // x offset
                0,                                    // y offset
                mTextureDataDesc.width,               // num texels width
                mTextureDataDesc.height,              // num texels height
                mOpenGLPixelFormat,                   // format
                mOpenGLDataType,                      // type
                data );                               // data

            break;

        case TextureDimensions_3D:
            glTexSubImage3D(
                mOpenGLTextureTarget,                 // target
                0,                                    // mip level
                0,                                    // x offset
                0,                                    // y offset
                0,                                    // z offset
                mTextureDataDesc.width,               // num texels width
                mTextureDataDesc.height,              // num texels height
                mTextureDataDesc.depth,               // num texels depth
                mOpenGLPixelFormat,                   // format
                mOpenGLDataType,                      // type
                data );                               // data

            break;

        default:
            Assert( 0 );
            break;
    }

    glBindTexture( mOpenGLTextureTarget, 0 );

    CheckErrors();
}

void OpenGLTexture::Update( const PixelBuffer* pixelBuffer ) const
{
    const OpenGLPixelBuffer* openGLPixelBuffer = dynamic_cast< const OpenGLPixelBuffer* >( pixelBuffer );

    glBindBufferARB( GL_PIXEL_UNPACK_BUFFER_ARB, openGLPixelBuffer->GetOpenGLPixelBufferID() );

    Update( (void*)NULL );

    glBindBufferARB( GL_PIXEL_UNPACK_BUFFER_ARB, 0 );

    CheckErrors();
}

void OpenGLTexture::SetMipLevelRegion( void* data, int mipLevel, int beginX, int beginY, int width, int height ) const
{
    Assert( mTextureDataDesc.dimensions == TextureDimensions_2D );

    glBindTexture( mOpenGLTextureTarget, mOpenGLTextureID );

    glTexSubImage2D(
        mOpenGLTextureTarget,    // target
        mipLevel,                // mip level
        beginX,                  // x offset
        beginY,                  // y offset
        width,                   // num texels width
        height,                  // num texels height
        mOpenGLPixelFormat,      // format
        mOpenGLDataType,         // type
        data );                  // data

    glBindTexture( mOpenGLTextureTarget, 0 );

    CheckErrors();
}

void OpenGLTexture::GenerateMipmaps() const
{
    glBindTexture( mOpenGLTextureTarget, mOpenGLTextureID );
    glGenerateMipmapEXT( mOpenGLTextureTarget );
    glBindTexture( mOpenGLTextureTarget, 0 );

    CheckErrors();
}

void* OpenGLTexture::GetMipLevel( int mipLevel ) const
{
    unsigned char* levelData = 0;
    int width = 1;
    int height = 1;
    int depth = 1;
    int numBytesPerComponent = GetOpenGLNumBytesPerComponent( mTextureDataDesc.pixelFormat );

    glBindTexture( mOpenGLTextureTarget, mOpenGLTextureID );

    switch( mTextureDataDesc.dimensions )
    {
        case TextureDimensions_1D:
        {
            glGetTexLevelParameteriv( 
                mOpenGLTextureTarget,
                mipLevel,
                GL_TEXTURE_WIDTH,
                &width );

            break;
        }

        case TextureDimensions_2D:
        {
             glGetTexLevelParameteriv( 
                mOpenGLTextureTarget,
                mipLevel,
                GL_TEXTURE_WIDTH,
                &width );
            glGetTexLevelParameteriv( 
                mOpenGLTextureTarget,
                mipLevel,
                GL_TEXTURE_HEIGHT,
                &height );

            break;
        }

        case TextureDimensions_3D:
        {
            glGetTexLevelParameteriv( 
                mOpenGLTextureTarget,
                mipLevel,
                GL_TEXTURE_WIDTH,
                &width );
            glGetTexLevelParameteriv( 
                mOpenGLTextureTarget,
                mipLevel,
                GL_TEXTURE_HEIGHT,
                &height );
            glGetTexLevelParameteriv( 
                mOpenGLTextureTarget,
                mipLevel,
                GL_TEXTURE_DEPTH,
                &depth );

            break;
        }

        default:
        {
            Assert( 0 );
            break;
        }

    }

    levelData = new unsigned char[ width * height * depth * numBytesPerComponent ];

    glGetTexImage( mOpenGLTextureTarget, mipLevel, mOpenGLPixelFormat, mOpenGLDataType, levelData );

    glBindTexture( mOpenGLTextureTarget, 0 );

    CheckErrors();

    return reinterpret_cast<void*>( levelData );

}

OpenGLBufferTexture::OpenGLBufferTexture( const BufferTextureDataDesc& bufferTextureDataDesc ) :
mBufferTextureDataDesc    ( bufferTextureDataDesc ),
mOpenGLBufferID           ( -1 ),
mOpenGLTextureID          ( -1 ),
mOpenGLPixelFormat        ( -1 ),
mOpenGLInternalPixelFormat( -1 ),
mOpenGLDataType           ( -1 )
{
    mOpenGLPixelFormat          = rtgi::GetOpenGLTexturePixelFormat        ( bufferTextureDataDesc.pixelFormat );
    mOpenGLInternalPixelFormat  = rtgi::GetOpenGLTextureInternalPixelFormat( bufferTextureDataDesc.pixelFormat );
    mOpenGLDataType             = rtgi::GetOpenGLDataType                  ( bufferTextureDataDesc.pixelFormat );
    GLuint numComponents        = rtgi::GetOpenGLNumComponents             ( bufferTextureDataDesc.pixelFormat );
    GLuint numBytesPerComponent = rtgi::GetOpenGLNumBytesPerComponent      ( bufferTextureDataDesc.pixelFormat );

    unsigned int numBytes = -1;

    // allocate texture id
    glGenTextures( 1, &mOpenGLTextureID );

    // bind texture
    glBindTexture( GL_TEXTURE_BUFFER_EXT, mOpenGLTextureID );

    // allocate buffer id
    glGenBuffersARB( 1, &mOpenGLBufferID );

    // bind buffer
    glBindBufferARB( GL_TEXTURE_BUFFER_EXT, mOpenGLBufferID );

    // upload data
    glBufferDataARB( GL_TEXTURE_BUFFER_EXT, bufferTextureDataDesc.numBytes, bufferTextureDataDesc.data, GL_STATIC_DRAW_ARB );

    CheckErrors();
}

OpenGLBufferTexture::~OpenGLBufferTexture()
{
    glDeleteBuffersARB( 1, &mOpenGLBufferID );
    glDeleteTextures( 1, &mOpenGLTextureID );

    CheckErrors();
}

void OpenGLBufferTexture::Bind() const
{
    glBindTexture( GL_TEXTURE_BUFFER_EXT, mOpenGLTextureID );
    glBindBufferARB( GL_TEXTURE_BUFFER_EXT, mOpenGLBufferID );
    glTexBufferEXT( GL_TEXTURE_BUFFER_EXT, mOpenGLInternalPixelFormat, mOpenGLBufferID ); 

    CheckErrors();
}

void OpenGLBufferTexture::Unbind() const
{
    glTexBufferEXT( GL_TEXTURE_BUFFER_EXT, mOpenGLInternalPixelFormat, 0 ); 
    glBindBufferARB( GL_TEXTURE_BUFFER_EXT, 0 );
    glBindTexture( GL_TEXTURE_BUFFER_EXT, 0 );

    CheckErrors();
}

GLuint OpenGLBufferTexture::GetOpenGLTextureID() const
{
    return mOpenGLTextureID;
}

GLuint OpenGLBufferTexture::GetOpenGLBufferID() const
{
    return mOpenGLBufferID;
}

GLenum OpenGLBufferTexture::GetOpenGLTexturePixelFormat() const 
{
    return mOpenGLPixelFormat;
}

GLenum OpenGLBufferTexture::GetOpenGLTextureInternalPixelFormat() const 
{
    return mOpenGLInternalPixelFormat;
}

GLenum OpenGLBufferTexture::GetOpenGLTextureDataType() const 
{
    return mOpenGLDataType;
}

BufferTextureDataDesc OpenGLBufferTexture::GetBufferTextureDataDesc() const
{
    return mBufferTextureDataDesc;
}

}

}