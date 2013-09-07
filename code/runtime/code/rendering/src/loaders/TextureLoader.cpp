#include "rendering/loaders/TextureLoader.hpp"

#include <math.h>

#include <BMPLoader.h>

#include <FCollada.h> 
#include <FCDocument/FCDocument.h>
#include <FCDocument/FCDLibrary.h>
#include <FUtils/FUDaeEnum.h>

#include <FCDocument/FCDEffect.h>
#include <FCDocument/FCDEffectProfile.h>
#include <FCDocument/FCDEffectProfileFX.h>
#include <FCDocument/FCDEffectCode.h>
#include <FCDocument/FCDEffectParameter.h>

#if defined(PLATFORM_WIN32)// This header is automatically included on Mac / Linux
#include <FCDocument/FCDEffectParameter.hpp>
#endif

#include <FCDocument/FCDEffectParameterSurface.h>
#include <FCDocument/FCDEffectParameterSampler.h>
#include <FCDocument/FCDMaterial.h>
#include <FCDocument/FCDImage.h>

#include "content/Inventory.hpp"
#include "content/ParameterManager.hpp"

#include "rendering/rtgi/RTGI.hpp"
#include "rendering/rtgi/Texture.hpp"

#include "rendering/Material.hpp"

namespace rendering
{

void TextureLoader::Load( content::Inventory* inventory, FCDocument* document )
{
    // get effect library
    FCDEffectLibrary* effectLibrary = document->GetEffectLibrary();

    // for each effect
    for ( size_t i = 0; i < effectLibrary->GetEntityCount(); i++ )
    {
        FCDEffect*          fcdEffect     = effectLibrary->GetEntity( i );
        FCDEffectProfile*   profileCG     = fcdEffect->FindProfile( FUDaeProfileType::CG );
        FCDEffectProfile*   profileCommon = fcdEffect->FindProfile( FUDaeProfileType::COMMON );

        // if the CG profile is non-null, then we assume that textures will be stored as parameters at the material level
        // rather than at the effect level.  thus we only check the parameters at the effect if the cg profile is NULL.
        if ( profileCG == NULL && profileCommon != NULL )
        {
            for ( size_t j = 0; j < profileCommon->GetEffectParameterCount(); j++ )
            {
                FCDEffectParameter*        effectParameter     = profileCommon->GetEffectParameter( j );
                FCDEffectParameter::Type   type                = effectParameter->GetType();
                core::String               effectParameterName = effectParameter->GetReference().c_str();

                if ( type == FCDEffectParameter::SAMPLER )
                {
                    FCDEffectParameterSampler* effectParameterSampler = dynamic_cast< FCDEffectParameterSampler* >( effectParameter );
                    FCDEffectParameterSurface* effectParameterSurface = effectParameterSampler->GetSurface();

                    Assert( effectParameterSurface->GetImageCount() == 1 );

                    FCDImage*    effectParameterImage = effectParameterSurface->GetImage( 0 );
#if defined(PLATFORM_WIN32)
                    core::String fileName             = effectParameterImage->GetFilename();
#elif defined(PLATFORM_OSX)
                    core::String fileName( effectParameterImage->GetFilename().c_str() );
#endif

                    BMPClass bmp;
                    BMPError errorCode = BMPLoad( fileName.ToStdString(), bmp );
                    Assert( errorCode == BMPNOERROR );

                    rtgi::TextureDataDesc textureDataDesc;

                    textureDataDesc.dimensions                   = rtgi::TextureDimensions_2D;
                    textureDataDesc.pixelFormat                  = rtgi::TexturePixelFormat_R8_G8_B8_UI_NORM;
                    textureDataDesc.data                         = bmp.bytes;
                    textureDataDesc.width                        = bmp.width;
                    textureDataDesc.height                       = bmp.height;
                    textureDataDesc.generateMipMaps              = true;

                    rtgi::Texture* texture = rtgi::CreateTexture( textureDataDesc );
                    
                    SetDebugName( texture, effectParameterName );

                    Insert( inventory, effectParameterName, texture );
                }
            }
        }
    }

    // now iterate through each material
    FCDMaterialLibrary* materialLibrary = document->GetMaterialLibrary();

    // for each material
    for ( size_t i = 0; i < materialLibrary->GetEntityCount(); i++ )
    {
        // get material
        FCDMaterial*                 fcdMaterial  = materialLibrary->GetEntity( i );
        FCDEffect*                   fcdEffect    = fcdMaterial->GetEffect();
        core::String                 materialName = fcdMaterial->GetName().c_str();

        // iterate through the parameters and get the samplers, since we can get the raw image filenames by tracing backwards from the samplers
        for ( size_t j = 0; j < fcdMaterial->GetEffectParameterCount(); j++ )
        {
            FCDEffectParameter*      effectParameter     = fcdMaterial->GetEffectParameter( j );
            FCDEffectParameter::Type type                = effectParameter->GetType();
            core::String             effectParameterName = effectParameter->GetReference().c_str();

            // get the type and value.  if we found ui annotations then make a slider
            if ( type == FCDEffectParameter::SAMPLER )
            {
                FCDEffectParameterSampler* effectParameterSampler = dynamic_cast< FCDEffectParameterSampler* >( effectParameter );
                FCDEffectParameterSurface* effectParameterSurface = effectParameterSampler->GetSurface();

                Assert( effectParameterSurface->GetImageCount() == 1 );

                FCDImage*    effectParameterImage = effectParameterSurface->GetImage( 0 );
#if defined(PLATFORM_WIN32)
                core::String fileName             = effectParameterImage->GetFilename();
                core::String semanticName         = effectParameterSampler->GetSemantic();
#elif defined(PLATFORM_OSX)
                core::String fileName( effectParameterImage->GetFilename().c_str() );
                core::String semanticName( effectParameterSampler->GetSemantic().c_str() );
#endif

                BMPClass bmp;
                BMPError errorCode = BMPLoad( fileName.ToStdString(), bmp );
                Assert( errorCode == BMPNOERROR );

                rtgi::TextureDataDesc textureDataDesc;

                textureDataDesc.dimensions                   = rtgi::TextureDimensions_2D;
                textureDataDesc.pixelFormat                  = rtgi::TexturePixelFormat_R8_G8_B8_UI_NORM;
                textureDataDesc.data                         = bmp.bytes;
                textureDataDesc.width                        = bmp.width;
                textureDataDesc.height                       = bmp.height;
                textureDataDesc.generateMipMaps              = true;

                rtgi::Texture* texture = rtgi::CreateTexture( textureDataDesc );

                SetDebugName( texture, effectParameterName );

                Insert( inventory, effectParameterName, texture );
            }
        }
    }
}

}