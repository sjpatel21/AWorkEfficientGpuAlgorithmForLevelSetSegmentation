#include "rendering/loaders/EffectLoader.hpp"

#include <math.h>

#include <FCollada.h> 
#include <FCDocument/FCDocument.h>
#include <FCDocument/FCDLibrary.h>
#include <FUtils/FUDaeEnum.h>

#include <FCDocument/FCDEffect.h>
#include <FCDocument/FCDEffectProfile.h>
#include <FCDocument/FCDEffectProfileFX.h>
#include <FCDocument/FCDEffectCode.h>
#include <FCDocument/FCDEffectParameter.h>
#if defined(PLATFORM_WIN32) // This header is automatically included on Mac / Linux
#include <FCDocument/FCDEffectParameter.hpp>
#endif
#include <FCDocument/FCDEffectParameterSurface.h>
#include <FCDocument/FCDEffectParameterSampler.h>
#include <FCDocument/FCDImage.h>

#include "content/Inventory.hpp"
#include "content/ParameterManager.hpp"

#include "rendering/rtgi/RTGI.hpp"
#include "rendering/rtgi/Effect.hpp"

#include "rendering/Material.hpp"

namespace rendering
{

void EffectLoader::Load( content::Inventory* inventory, FCDocument* document )
{
    // get effect library
    FCDEffectLibrary* effectLibrary = document->GetEffectLibrary();

    // for each effect
    for ( size_t i = 0; i < effectLibrary->GetEntityCount(); i++ )
    {
        FCDEffect*          fcdEffect     = effectLibrary->GetEntity( i );
        FCDEffectProfile*   profileCG     = fcdEffect->FindProfile( FUDaeProfileType::CG );
        FCDEffectProfile*   profileCommon = fcdEffect->FindProfile( FUDaeProfileType::COMMON );
        core::String        effectName    = fcdEffect->GetName().c_str();
        core::String        effectFile    = "";

        if ( profileCG != NULL )
        {
            FCDEffectProfileFX* profileFX  = dynamic_cast< FCDEffectProfileFX* >( profileCG );

            Assert( profileFX != NULL );
            Assert( profileFX->GetCodeCount() == 1 );

            FCDEffectCode* code = profileFX->GetCode( 0 );
            effectFile          = code->GetFilename().c_str();

            rtgi::Effect* effect = rtgi::CreateEffect( effectFile );

            SetDebugName( effect, effectName );

            Insert( inventory, effectName, effect );
        }
   }
}

}