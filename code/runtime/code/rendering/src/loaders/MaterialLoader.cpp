#include "rendering/loaders/MaterialLoader.hpp"

#include <math.h>

#include <FCollada.h> 
#include <FCDocument/FCDocument.h>
#include <FCDocument/FCDLibrary.h>
#include <FCDocument/FCDMaterial.h>
#include <FUtils/FUDaeEnum.h>

#include <FCDocument/FCDEffect.h>
#include <FCDocument/FCDEffectStandard.h>
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
#include <FCDocument/FCDTexture.h>

#include "content/Inventory.hpp"
#include "content/ParameterManager.hpp"

#include "rendering/rtgi/RTGI.hpp"
#include "rendering/rtgi/Effect.hpp"

#include "rendering/Material.hpp"
#include "rendering/Context.hpp"

namespace rendering
{

void MaterialLoader::Load( content::Inventory* inventory, FCDocument* document )
{
    // get the material library
    FCDMaterialLibrary* materialLibrary = document->GetMaterialLibrary();

    // for each material
    for ( size_t i = 0; i < materialLibrary->GetEntityCount(); i++ )
    {
        // get material
        FCDMaterial*                 fcdMaterial  = materialLibrary->GetEntity( i );
        FCDEffect*                   fcdEffect    = fcdMaterial->GetEffect();
        core::String                 materialName = fcdMaterial->GetName().c_str();

        // create runtime material
        Material* material = new Material();

        SetFinalized( material, false );

        // to wire the material to the effect, we need to determine if the effect is a common profile
        // effect (in which case we wire the material to the default phong effect), or a CG profile
        // custom effect defined in a .cgfx file.
        FCDEffectProfile*   profileCG     = fcdEffect->FindProfile( FUDaeProfileType::CG );
        FCDEffectProfile*   profileCommon = fcdEffect->FindProfile( FUDaeProfileType::COMMON );
        core::String        effectName;

        if ( profileCG != NULL )
        {
            effectName = fcdEffect->GetName().c_str();

            // iterate through the parameters and add the parameters to our runtime material
            for ( size_t j = 0; j < fcdMaterial->GetEffectParameterCount(); j++ )
            {
                FCDEffectParameter*      effectParameter     = fcdMaterial->GetEffectParameter( j );
                FCDEffectParameter::Type type                = effectParameter->GetType();
                core::String             effectParameterName = effectParameter->GetReference().c_str();

                bool foundUIMinValue = false;
                bool foundUIMaxValue = false;
                float uiMinValue, uiMaxValue;

                // look for UI annotations so we can make slider
                for ( size_t k = 0; k < effectParameter->GetAnnotationCount(); k++ )
                {
                    FCDEffectParameterAnnotation* annotation = effectParameter->GetAnnotation( k );

                    if ( core::String( annotation->name->c_str() ) == "UIMinValue" || core::String( annotation->name->c_str() ) == "UIMin" )
                    {
                        foundUIMinValue = true;
                        uiMinValue      = atof( annotation->value->c_str() );
                    }

                    if ( core::String( annotation->name->c_str() ) == "UIMaxValue" || core::String( annotation->name->c_str() ) == "UIMax" )
                    {
                        foundUIMaxValue = true;
                        uiMaxValue      = atof( annotation->value->c_str() );
                    }
                }

                // get the type and value.  if we found ui annotations then make a slider
                switch ( type )
                {
                case FCDEffectParameter::INTEGER:
                    {
                        FCDEffectParameterInt* effectParameterInt = dynamic_cast< FCDEffectParameterInt* >( effectParameter );
                        float                  value              = static_cast< float >( effectParameterInt->GetValue() );

                        material->InsertNewMaterialParameter( effectParameterName, value );

                        if ( foundUIMinValue && foundUIMaxValue )
                        {
                            content::ParameterManager::InsertNewParameter(
                                "materials", Material::GetFullyQualifiedMaterialParameterName( materialName, effectParameterName ), value, uiMinValue, uiMaxValue );
                        }
                    }
                    break;

                case FCDEffectParameter::FLOAT:
                    {
                        FCDEffectParameterFloat* effectParameterFloat = dynamic_cast< FCDEffectParameterFloat* >( effectParameter );
                        float                    value                = effectParameterFloat->GetValue();

                        material->InsertNewMaterialParameter( effectParameterName, value );

                        if ( foundUIMinValue && foundUIMaxValue )
                        {
                            content::ParameterManager::InsertNewParameter(
                                "materials", Material::GetFullyQualifiedMaterialParameterName( materialName, effectParameterName ), value, uiMinValue, uiMaxValue );
                        }
                    }

                    break;

                case FCDEffectParameter::FLOAT3:
                    {
                        FCDEffectParameterFloat3* effectParameterFloat3 = dynamic_cast< FCDEffectParameterFloat3* >( effectParameter );
                        FMVector3 fmValue = effectParameterFloat3->GetValue();

                        math::Vector3 value;

                        value[ math::X ] = fmValue.x;
                        value[ math::Y ] = fmValue.y;
                        value[ math::Z ] = fmValue.z;

                        material->InsertNewMaterialParameter( effectParameterName, value );

                        if ( foundUIMinValue && foundUIMaxValue )
                        {
                            content::ParameterManager::InsertNewParameter(
                                "materials", Material::GetFullyQualifiedMaterialParameterName( materialName, effectParameterName ), value, uiMinValue, uiMaxValue );
                        }
                    }
                    break;

                case FCDEffectParameter::VECTOR:
                    {
                        FCDEffectParameterVector* effectParameterVector = dynamic_cast< FCDEffectParameterVector* >( effectParameter );
                        FMVector4 fmValue = effectParameterVector->GetValue();

                        math::Vector4 value;

                        value[ math::X ] = fmValue.x;
                        value[ math::Y ] = fmValue.y;
                        value[ math::Z ] = fmValue.z;
                        value[ math::H ] = fmValue.w;

                        material->InsertNewMaterialParameter( effectParameterName, value );

                        if ( foundUIMinValue && foundUIMaxValue )
                        {
                            content::ParameterManager::InsertNewParameter(
                                "materials", Material::GetFullyQualifiedMaterialParameterName( materialName, effectParameterName ), value, uiMinValue, uiMaxValue );
                        }
                    }
                    break;

                case FCDEffectParameter::MATRIX:
                    {
                        Assert( 0 );
                    }
                    break;

                case FCDEffectParameter::SAMPLER:
                    {
                        Assert( 0 );
                    }
                    break;

                case FCDEffectParameter::SURFACE:
                    {
                        Assert( 0 );
                    }
                    break;

                default:
                    {
                        Assert( 0 );
                    }
                    break;

                }
            }

            if ( inventory->Contains( effectName ) )
            {
                content::Ref< rtgi::Effect > effect = inventory->Find< rtgi::Effect >( effectName );

                SetDebugName( material, materialName );

                material->SetName( materialName );
                material->SetEffect( effect );

                SetFinalized( material, true );

                Insert( inventory, materialName, material );
            }
            else
            {
                // force delete of material
                material->AddRef();
                material->Release();
            }
        }
        else if ( profileCommon != NULL )
        {
            effectName = fcdEffect->GetName().c_str();

            FCDEffectStandard* effectStandard = dynamic_cast< FCDEffectStandard* >( profileCommon );
            Assert( effectStandard != NULL );

            const FMVector4& ambientColor  = effectStandard->GetAmbientColor();
            const FMVector4& diffuseColor  = effectStandard->GetDiffuseColor();
            const FMVector4& specularColor = effectStandard->GetSpecularColor();
            float            shininess     = effectStandard->GetShininess();

            size_t diffuseTextureCount = effectStandard->GetTextureCount( FUDaeTextureChannel::DIFFUSE );

            Assert( diffuseTextureCount <= 1 );
            Assert( effectStandard->GetTextureCount( FUDaeTextureChannel::AMBIENT )        == 0 );
            Assert( effectStandard->GetTextureCount( FUDaeTextureChannel::BUMP )           == 0 );
            Assert( effectStandard->GetTextureCount( FUDaeTextureChannel::COUNT )          == 0 );
            Assert( effectStandard->GetTextureCount( FUDaeTextureChannel::DISPLACEMENT )   == 0 );
            Assert( effectStandard->GetTextureCount( FUDaeTextureChannel::EMISSION )       == 0 );
            Assert( effectStandard->GetTextureCount( FUDaeTextureChannel::FILTER )         == 0 );
            Assert( effectStandard->GetTextureCount( FUDaeTextureChannel::REFLECTION )     == 0 );
            Assert( effectStandard->GetTextureCount( FUDaeTextureChannel::REFRACTION )     == 0 );
            Assert( effectStandard->GetTextureCount( FUDaeTextureChannel::SHININESS )      == 0 );
            Assert( effectStandard->GetTextureCount( FUDaeTextureChannel::SPECULAR )       == 0 );
            Assert( effectStandard->GetTextureCount( FUDaeTextureChannel::SPECULAR_LEVEL ) == 0 );
            Assert( effectStandard->GetTextureCount( FUDaeTextureChannel::TRANSPARENT )    == 0 );
            Assert( effectStandard->GetTextureCount( FUDaeTextureChannel::UNKNOWN )        == 0 );
            
            if ( diffuseTextureCount == 0 )
            {
                material->InsertNewMaterialParameter( "useDiffuseMap",     0.0f );
                material->InsertNewMaterialParameter( "diffuseMapSampler", Context::GetDebugTexture() );
                material->InsertNewMaterialParameter( "diffuseColor",      math::Vector3( diffuseColor.x, diffuseColor.y, diffuseColor.z ) );
            }
            else
            {
                FCDTexture*                   diffuseTexture     = effectStandard->GetTexture( FUDaeTextureChannel::DIFFUSE, 0 );
                FCDEffectParameterSampler*    diffuseSampler     = diffuseTexture->GetSampler();
                core::String                  diffuseSamplerName = diffuseSampler->GetReference().c_str();
                content::Ref< rtgi::Texture > texture            = inventory->Find< rtgi::Texture >( diffuseSamplerName );

                material->InsertNewMaterialParameter( "useDiffuseMap",     1.0f );
                material->InsertNewMaterialParameter( "diffuseMapSampler", texture.GetRawPointer() );
                material->InsertNewMaterialParameter( "diffuseColor",      math::Vector3( 1.0f, 0.0f, 1.0f ) );
            }

            material->InsertNewMaterialParameter( "ambientColor",      math::Vector3( ambientColor.x, ambientColor.y, ambientColor.z ) );
            material->InsertNewMaterialParameter( "specularColor",     math::Vector3( specularColor.x, specularColor.y, specularColor.z ) );
            material->InsertNewMaterialParameter( "shininess",         shininess );

            content::Ref< rtgi::Effect > effect = Context::GetPhongEffect();

            SetDebugName( material, materialName );

            material->SetName( materialName );
            material->SetEffect( effect );

            SetFinalized( material, true );

            Insert( inventory, materialName, material );
        }
        else
        {
            Assert( 0 );
        }
    }
}

}