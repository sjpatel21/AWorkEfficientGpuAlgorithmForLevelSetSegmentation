#include "rendering/Material.hpp"

#include "math/Utility.hpp"
#include "math/Matrix44.hpp"

#include "content/ParameterManager.hpp"

#include "rendering/rtgi/Effect.hpp"
#include "rendering/rtgi/Texture.hpp"
#include "rendering/rtgi/Color.hpp"
#include "rendering/rtgi/VertexBuffer.hpp"

#include "rendering/Context.hpp"
#include "rendering/Camera.hpp"

namespace rendering
{

Material::Material()
{
}

Material::~Material()
{
}

void Material::Bind()
{
    SetEffectParameters();
}

void Material::Bind( container::Map< core::String, rtgi::VertexDataSourceDesc > vertexDataSources )
{
    mEffect->BindVertexDataSources( vertexDataSources );

    SetEffectParameters();
}

void Material::Unbind()
{
    mEffect->UnbindVertexDataSources();
}

bool Material::BindPass()
{
    return mEffect->BindPass();
}

void Material::UnbindPass()
{
    mEffect->UnbindPass();
}

void Material::SetEffect( content::Ref< rtgi::Effect > effect )
{
    Assert( !IsFinalized() );

    mEffect = effect;
}

void Material::SetName( const core::String& name )
{
    Assert( !IsFinalized() );

    mName = name;
}

void Material::SetMaterialParameter( const core::String& materialParameterName, const rtgi::Texture* value )
{
    Assert( mParameters.Contains( materialParameterName ) );

    MaterialParameter existingValue;

    existingValue = mParameters.Value( materialParameterName );

    Assert( existingValue.type == MaterialParameterType_Texture );

    existingValue.dataTexture = value;

    mParameters.Insert( materialParameterName, existingValue );    
}

void Material::SetMaterialParameter( const core::String& materialParameterName, const math::Matrix44& value )
{
    Assert( mParameters.Contains( materialParameterName ) );

    MaterialParameter existingValue;

    existingValue = mParameters.Value( materialParameterName );

    Assert( existingValue.type == MaterialParameterType_Matrix44 );

    existingValue.dataMatrix44 = value;

    mParameters.Insert( materialParameterName, existingValue );    
}

void Material::SetMaterialParameter( const core::String& materialParameterName, const math::Vector4& value )
{
    Assert( mParameters.Contains( materialParameterName ) );

    MaterialParameter existingValue;

    existingValue = mParameters.Value( materialParameterName );

    Assert( existingValue.type == MaterialParameterType_Vector4 );

    existingValue.dataVector4 = value;

    mParameters.Insert( materialParameterName, existingValue );    
}

void Material::SetMaterialParameter( const core::String& materialParameterName, const math::Vector3& value )
{
    Assert( mParameters.Contains( materialParameterName ) );

    MaterialParameter existingValue;

    existingValue = mParameters.Value( materialParameterName );

    Assert( existingValue.type == MaterialParameterType_Vector3 );

    existingValue.dataVector3 = value;

    mParameters.Insert( materialParameterName, existingValue );    
}

void Material::SetMaterialParameter( const core::String& materialParameterName, float value )
{
    Assert( mParameters.Contains( materialParameterName ) );

    MaterialParameter existingValue;

    existingValue = mParameters.Value( materialParameterName );

    Assert( existingValue.type == MaterialParameterType_Float );

    existingValue.dataFloat = value;

    mParameters.Insert( materialParameterName, existingValue );    
}

core::String Material::GetFullyQualifiedMaterialParameterName( const core::String& materialName, const core::String& materialParameterName )
{
    return materialName + "-" + materialParameterName;
}

void Material::InsertNewMaterialParameter( const core::String& materialParameterName, const rtgi::Texture* value )
{
    MaterialParameter newValue;

    newValue.type        = MaterialParameterType_Texture;
    newValue.dataTexture = value;

    mParameters.Insert( materialParameterName, newValue );
}

void Material::InsertNewMaterialParameter( const core::String& materialParameterName, const math::Matrix44& value )
{
    MaterialParameter newValue;

    newValue.type         = MaterialParameterType_Matrix44;
    newValue.dataMatrix44 = value;

    mParameters.Insert( materialParameterName, newValue );
}

void Material::InsertNewMaterialParameter( const core::String& materialParameterName, const math::Vector4& value )
{
    MaterialParameter newValue;

    newValue.type        = MaterialParameterType_Vector4;
    newValue.dataVector4 = value;

    mParameters.Insert( materialParameterName, newValue );
}

void Material::InsertNewMaterialParameter( const core::String& materialParameterName, const math::Vector3& value )
{
    MaterialParameter newValue;

    newValue.type        = MaterialParameterType_Vector3;
    newValue.dataVector3 = value;

    mParameters.Insert( materialParameterName, newValue );
}

void Material::InsertNewMaterialParameter( const core::String& materialParameterName, float value )
{
    MaterialParameter newValue;

    newValue.type      = MaterialParameterType_Float;
    newValue.dataFloat = value;

    mParameters.Insert( materialParameterName, newValue );
}

void Material::SetEffectParameters()
{
    mEffect->BeginSetEffectParameters();

    //
    // get UI updates if available
    //
    foreach_key_value ( core::String materialParameterName, MaterialParameter materialParameter, mParameters )
    {
        core::String fullyQualifiedMaterialName = GetFullyQualifiedMaterialParameterName( mName, materialParameterName );

        if ( content::ParameterManager::Contains( "materials", fullyQualifiedMaterialName ) )
        {
            content::ParameterType type = content::ParameterManager::GetParameterType( "materials", fullyQualifiedMaterialName );

            switch( type )
            {
            case content::ParameterType_Float:
                SetMaterialParameter( materialParameterName, content::ParameterManager::GetParameter< float >( "materials", fullyQualifiedMaterialName ) );
                break;

            case content::ParameterType_Vector3:
                SetMaterialParameter( materialParameterName, content::ParameterManager::GetParameter< math::Vector3 >( "materials", fullyQualifiedMaterialName ) );
                break;

            case content::ParameterType_Vector4:
                SetMaterialParameter( materialParameterName, content::ParameterManager::GetParameter< math::Vector4 >( "materials", fullyQualifiedMaterialName ) );
                break;

            case content::ParameterType_Matrix44:
                SetMaterialParameter( materialParameterName, content::ParameterManager::GetParameter< math::Matrix44 >( "materials", fullyQualifiedMaterialName ) );
                break;
            }
        }
    }

    //
    // if the effect requires any engine parameters, then set the engine parameters here
    //
    container::List< core::String > parameterNames = Context::GetEffectParameterNames();

    foreach ( core::String effectParameterName, parameterNames )
    {
        if ( mEffect->ContainsEffectParameter( effectParameterName ) )
        {
            switch( Context::GetEffectParameterType( effectParameterName ) )
            {
            case rtgi::EffectParameterType_Texture:
                mEffect->SetEffectParameter( effectParameterName, Context::GetEffectParameter< const rtgi::Texture* >( effectParameterName ) );
                break;

            case rtgi::EffectParameterType_Matrix44:
                mEffect->SetEffectParameter( effectParameterName, Context::GetEffectParameter< math::Matrix44 >( effectParameterName ) );
                break;

            case rtgi::EffectParameterType_Vector4:
                mEffect->SetEffectParameter( effectParameterName, Context::GetEffectParameter< math::Vector4 >( effectParameterName ) );
                break;

            case rtgi::EffectParameterType_Vector3:
                mEffect->SetEffectParameter( effectParameterName, Context::GetEffectParameter< math::Vector3 >( effectParameterName ) );
                break;

            case rtgi::EffectParameterType_Float:
                mEffect->SetEffectParameter( effectParameterName, Context::GetEffectParameter< float >( effectParameterName ) );
                break;

            default:
                Assert( 0 );
            }
        }
    }

    //
    // now set the parameters specific to this material
    //
    foreach_key_value ( core::String materialParameterName, MaterialParameter materialParameter, mParameters )
    {
        switch( materialParameter.type )
        {
        case  rtgi::EffectParameterType_Texture:
            mEffect->SetEffectParameter( materialParameterName, mParameters.Value( materialParameterName ).dataTexture );
            break;

        case rtgi::EffectParameterType_Matrix44:
            mEffect->SetEffectParameter( materialParameterName, mParameters.Value( materialParameterName ).dataMatrix44 );
            break;

        case rtgi::EffectParameterType_Vector4:
            mEffect->SetEffectParameter( materialParameterName, mParameters.Value( materialParameterName ).dataVector4 );
            break;

        case rtgi::EffectParameterType_Vector3:
            mEffect->SetEffectParameter( materialParameterName, mParameters.Value( materialParameterName ).dataVector3 );
            break;

        case rtgi::EffectParameterType_Float:
            mEffect->SetEffectParameter( materialParameterName, mParameters.Value( materialParameterName ).dataFloat );
            break;

        default:
            Assert( 0 );
        }
    }

    mEffect->EndSetEffectParameters();
}
}