#ifndef RENDERING_RTGI_TEXTURE_HPP
#define RENDERING_RTGI_TEXTURE_HPP

#include "content/Asset.hpp"

namespace rendering
{

namespace rtgi
{

class PixelBuffer;

enum TextureSamplerWrapMode
{
    TextureSamplerWrapMode_Clamp,
    TextureSamplerWrapMode_ClampToEdge,
    TextureSamplerWrapMode_Repeat,

    TextureSamplerWrapMode_Invalid
};

enum TextureSamplerInterpolationMode
{
    TextureSamplerInterpolationMode_Nearest,
    TextureSamplerInterpolationMode_Smooth,
    TextureSamplerInterpolationMode_SmoothMipMaps,

    TextureSamplerInterpolationMode_Invalid
};

enum TexturePixelFormat
{
    TexturePixelFormat_R8_UI_DENORM,            
    TexturePixelFormat_R8_I_DENORM,             
    TexturePixelFormat_R16_UI_DENORM,           
    TexturePixelFormat_R16_I_DENORM,              
    TexturePixelFormat_R32_UI_DENORM,           
    TexturePixelFormat_R32_I_DENORM,            
                                           
    TexturePixelFormat_A16_F_DENORM,            
    TexturePixelFormat_A32_F_DENORM,            
                                           
    TexturePixelFormat_R8_G8_UI_DENORM,         
                                           
    TexturePixelFormat_R16_G16_B16_A16_UI_DENORM,
    TexturePixelFormat_R16_G16_B16_A16_I_DENORM,

    TexturePixelFormat_R32_G32_B32_A32_UI_DENORM,
    TexturePixelFormat_R32_G32_B32_A32_I_DENORM,

    TexturePixelFormat_R16_G16_B16_A16_F_DENORM,
    TexturePixelFormat_R32_G32_B32_A32_F_DENORM,

    TexturePixelFormat_A8_UI_NORM,
    TexturePixelFormat_A8_I_NORM,
    TexturePixelFormat_R8_G8_B8_UI_NORM,        
    TexturePixelFormat_R8_G8_B8_A8_UI_NORM,     
    TexturePixelFormat_R8_G8_B8_A8_I_NORM,

    TexturePixelFormat_L32_F_DENORM,

    TexturePixelFormat_Invalid
};

enum TextureDimensions
{
    TextureDimensions_1D,
    TextureDimensions_2D,
    TextureDimensions_3D,

    TextureDimensions_Invalid
};

struct TextureDataDesc
{
    TextureDimensions  dimensions;
    TexturePixelFormat pixelFormat;
    unsigned int       width;
    unsigned int       height;
    unsigned int       depth;
    void*              data;
    bool               generateMipMaps;

    TextureDataDesc();
};

struct TextureSamplerStateDesc
{
    TextureSamplerWrapMode          textureSamplerWrapMode;
    TextureSamplerInterpolationMode textureSamplerInterpolationMode;

    TextureSamplerStateDesc();
};

class Texture : public content::Asset
{
public:
    virtual void Bind( const TextureSamplerStateDesc& textureSamplerStateDesc ) const = 0;
    virtual void Unbind()                                                       const = 0;

    virtual void Update( void* data )                     const = 0;
    virtual void Update( const PixelBuffer* pixelBuffer ) const = 0;

    virtual void  GenerateMipmaps()                                                                            const = 0;
    virtual void* GetMipLevel( int mipLevel )                                                                  const = 0;
    virtual void  SetMipLevelRegion( void* data, int mipLevel, int beginX, int beginY, int width, int height ) const = 0;

protected:
    Texture() {};
    virtual ~Texture() {};
};

struct BufferTextureDataDesc
{
    TexturePixelFormat pixelFormat;
    unsigned int       numBytes;
    void*              data;

    BufferTextureDataDesc();
};

class BufferTexture : public content::Asset
{
public:
    virtual void Bind()   const = 0;
    virtual void Unbind() const = 0;

protected:
    BufferTexture() {};
    virtual ~BufferTexture() {};
};

}

}

#endif