#include "content/ParameterManager.hpp"

#include <boost/filesystem.hpp>

#include "core/String.hpp"
#include "core/Assert.hpp"

#include "container/List.hpp"
#include "container/InsertOrderedMap.hpp"

#include "content/Document.hpp"
#include "content/Element.hpp"
#include "content/Config.hpp"

namespace content
{

bool                                                                                                    sInitialized = false;
container::InsertOrderedMap< core::String, container::InsertOrderedMap< core::String, ParameterDesc > > sSystemParameterKeyValueMap;

void ParameterManager::Initialize()
{
    sInitialized = true;
}

void ParameterManager::Terminate()
{
    sInitialized = false;
}

void ParameterManager::LoadParameters( const core::String& parametersFile )
{
    Assert( sInitialized );

    boost::filesystem::path parametersFilePath( parametersFile.ToStdString() );
    core::String            absoluteParametersFile( parametersFilePath.native_file_string() );

    ReleaseAssert( boost::filesystem::exists( parametersFilePath ) );

    Document parameterDatabaseFile( parametersFile );

    Element rootElement              = parameterDatabaseFile.GetRootElement();
    Element parameterDatabaseElement = rootElement.GetCurrentChildElement( "ParameterManager" );
    Element systemsElement           = parameterDatabaseElement.GetCurrentChildElement( "Systems" );
    Element systemElement            = systemsElement.GetCurrentChildElement( "System" );

    while( !systemElement.IsNull() )
    {
        core::String systemName = systemElement.GetAttributeValue< core::String >( "name" );

        Element parametersElement     = systemElement.GetCurrentChildElement( "Parameters" );
        Element parameterElement      = parametersElement.GetCurrentChildElement( "Parameter" );

        while( !parameterElement.IsNull() )
        {
            core::String  parameterName       = parameterElement.GetAttributeValue< core::String >( "name" );
            core::String  parameterTypeString = parameterElement.GetAttributeValue< core::String >( "type" );
            float         parameterMinValue   = parameterElement.GetAttributeValue< float >( "minValue" );
            float         parameterMaxValue   = parameterElement.GetAttributeValue< float >( "maxValue" );

            if ( parameterTypeString == "ParameterType_Float" )
            {
                Element       parameterValueElement = parameterElement.GetCurrentChildElement( "ValueFloat" );
                ParameterType parameterType      = ParameterType_Float;
                float         parameterValue     = parameterValueElement.GetAttributeValue< float >( "value" );

                InsertNewParameter( systemName, parameterName, parameterValue, parameterMinValue, parameterMaxValue );
            }
            else if ( parameterTypeString == "ParameterType_Vector3" )
            {
                Element       parameterValueElement = parameterElement.GetCurrentChildElement( "ValueVector3" );
                ParameterType parameterType      = ParameterType_Vector3;
                math::Vector3 parameterValue;

                parameterValue[ math::X ] = parameterValueElement.GetAttributeValue< float >( "x" );
                parameterValue[ math::Y ] = parameterValueElement.GetAttributeValue< float >( "y" );
                parameterValue[ math::Z ] = parameterValueElement.GetAttributeValue< float >( "z" );

                InsertNewParameter( systemName, parameterName, parameterValue, parameterMinValue, parameterMaxValue );
            }
            else if ( parameterTypeString == "ParameterType_Vector4" )
            {
                Element       parameterValueElement = parameterElement.GetCurrentChildElement( "ValueVector4" );
                ParameterType parameterType      = ParameterType_Vector4;
                math::Vector4 parameterValue;

                parameterValue[ math::X ] = parameterValueElement.GetAttributeValue< float >( "x" );
                parameterValue[ math::Y ] = parameterValueElement.GetAttributeValue< float >( "y" );
                parameterValue[ math::Z ] = parameterValueElement.GetAttributeValue< float >( "z" );
                parameterValue[ math::H ] = parameterValueElement.GetAttributeValue< float >( "h" );

                InsertNewParameter( systemName, parameterName, parameterValue, parameterMinValue, parameterMaxValue );
            }
            else
            {
                Assert( 0 );
            }

            parameterElement = parametersElement.GetCurrentChildElement( "Parameter" );
        }

        systemElement = systemsElement.GetCurrentChildElement( "System" );
    }
}

void ParameterManager::SaveParameters( const core::String& parametersFile )
{
    Assert( sInitialized );

    boost::filesystem::path parametersFilePath( parametersFile.ToStdString() );
    core::String            absoluteParametersFile( parametersFilePath.native_file_string() );

    ReleaseAssert( boost::filesystem::exists( parametersFilePath ) );

    Document parameterDatabaseFile( parametersFile, DocumentOpenMode_ReadWrite );

    Element rootElement              = parameterDatabaseFile.GetRootElement();
    Element parameterDatabaseElement = rootElement.GetCurrentChildElement( "ParameterManager" );
    Element systemsElement           = parameterDatabaseElement.GetCurrentChildElement( "Systems" );
    Element systemElement            = systemsElement.GetCurrentChildElement( "System" );

    while( !systemElement.IsNull() )
    {
        core::String systemName = systemElement.GetAttributeValue< core::String >( "name" );

        Element parametersElement     = systemElement.GetCurrentChildElement( "Parameters" );
        Element parameterElement      = parametersElement.GetCurrentChildElement( "Parameter" );

        while( !parameterElement.IsNull() )
        {
            core::String parameterName = parameterElement.GetAttributeValue< core::String >( "name" );

            switch( GetParameterType( systemName, parameterName ) )
            {
            case ParameterType_Float:
                {
                    Element parameterValueElement = parameterElement.GetCurrentChildElement( "ValueFloat" );
                    parameterValueElement.SetAttributeValue( "value", GetParameter< float >( systemName, parameterName ) );
                }
                break;

            case ParameterType_Vector3:
                {
                    Element parameterValueElement = parameterElement.GetCurrentChildElement( "ValueVector3" );
                    parameterValueElement.SetAttributeValue( "x", GetParameter< math::Vector3 >( systemName, parameterName )[ math::X ] );
                    parameterValueElement.SetAttributeValue( "y", GetParameter< math::Vector3 >( systemName, parameterName )[ math::Y ] );
                    parameterValueElement.SetAttributeValue( "z", GetParameter< math::Vector3 >( systemName, parameterName )[ math::Z ] );
                }
                break;

            case ParameterType_Vector4:
                {
                    Element parameterValueElement = parameterElement.GetCurrentChildElement( "ValueVector4" );
                    parameterValueElement.SetAttributeValue( "x", GetParameter< math::Vector4 >( systemName, parameterName )[ math::X ] );
                    parameterValueElement.SetAttributeValue( "y", GetParameter< math::Vector4 >( systemName, parameterName )[ math::Y ] );
                    parameterValueElement.SetAttributeValue( "z", GetParameter< math::Vector4 >( systemName, parameterName )[ math::Z ] );
                    parameterValueElement.SetAttributeValue( "h", GetParameter< math::Vector4 >( systemName, parameterName )[ math::H ] );
                }
                break;

            default:
                Assert( 0 );
            }

            parameterElement = parametersElement.GetCurrentChildElement( "Parameter" );
        }

        systemElement = systemsElement.GetCurrentChildElement( "System" );
    }

    parameterDatabaseFile.Save();
}

void ParameterManager::SetParameter( const core::String& system, const core::String& key, const math::Matrix44& value )
{
    Assert( Contains( system, key ) );

    ParameterDesc existingValue;

    Value( system, key, existingValue );

    Assert( existingValue.type == ParameterType_Matrix44 );

    existingValue.dataMatrix44 = value;

    Insert( system, key, existingValue );    
}

void ParameterManager::SetParameter( const core::String& system, const core::String& key, const math::Vector4& value )
{
    Assert( Contains( system, key ) );

    ParameterDesc existingValue;

    Value( system, key, existingValue );

    Assert( existingValue.type == ParameterType_Vector4 );

    existingValue.dataVector4 = value;

    Insert( system, key, existingValue );    
}

void ParameterManager::SetParameter( const core::String& system, const core::String& key, const math::Vector3& value )
{
    Assert( Contains( system, key ) );

    ParameterDesc existingValue;

    Value( system, key, existingValue );

    Assert( existingValue.type == ParameterType_Vector3 );

    existingValue.dataVector3 = value;

    Insert( system, key, existingValue );    
}

void ParameterManager::SetParameter( const core::String& system, const core::String& key, float value )
{
    Assert( Contains( system, key ) );

    ParameterDesc existingValue;

    Value( system, key, existingValue );

    Assert( existingValue.type == ParameterType_Float );

    existingValue.dataFloat = value;

    Insert( system, key, existingValue );    
}

void ParameterManager::InsertNewParameter( const core::String& system, const core::String& key, float value, float min, float max )
{
    ParameterDesc desc( value, min, max );

    Insert( system, key, desc );
}

void ParameterManager::InsertNewParameter( const core::String& system, const core::String& key, const math::Vector3& value, float min, float max )
{
    ParameterDesc desc( value, min, max );

    Insert( system, key, desc );
}

void ParameterManager::InsertNewParameter( const core::String& system, const core::String& key, const math::Vector4& value, float min, float max )
{
    ParameterDesc desc( value, min, max );

    Insert( system, key, desc );
}

void ParameterManager::InsertNewParameter( const core::String& system, const core::String& key, const math::Matrix44& value, float min, float max )
{
    ParameterDesc desc( value, min, max );

    Insert( system, key, desc );
}

ParameterType ParameterManager::GetParameterType( const core::String& system, const core::String& key )
{
    Assert( Contains( system, key ) );

    ParameterDesc existingValue;

    Value( system, key, existingValue );

    return existingValue.type;
}

float ParameterManager::GetMinimumValue( const core::String& system, const core::String& key )
{
    Assert( Contains( system, key ) );

    ParameterDesc existingValue;

    Value( system, key, existingValue );

    return existingValue.minimumValue;
}

float ParameterManager::GetMaximumValue( const core::String& system, const core::String& key )
{
    Assert( Contains( system, key ) );

    ParameterDesc existingValue;

    Value( system, key, existingValue );

    return existingValue.maximumValue;
}

bool ParameterManager::Contains( const core::String& system, const core::String& key )
{
    if ( !sSystemParameterKeyValueMap.Contains( system ) )
    {
        return false;
    }
    else if ( !sSystemParameterKeyValueMap.Value( system ).Contains( key ) )
    {
        return false;
    }
    else
    {
        return true;
    }
}

container::List< core::String >::ConstIterator ParameterManager::GetSystemsBegin()
{
    return sSystemParameterKeyValueMap.Keys().ConstBegin();
}

container::List< core::String >::ConstIterator ParameterManager::GetSystemsEnd()
{
    return sSystemParameterKeyValueMap.Keys().ConstEnd();
}

container::List< core::String >::ConstIterator ParameterManager::GetParametersBegin( const core::String& system )
{
    return sSystemParameterKeyValueMap.Value( system ).Keys().ConstBegin();
}

container::List< core::String >::ConstIterator ParameterManager::GetParametersEnd( const core::String& system )
{
    return sSystemParameterKeyValueMap.Value( system ).Keys().ConstEnd();
}

void ParameterManager::RemoveSystem( const core::String& system )
{
    sSystemParameterKeyValueMap.Remove( "materials" );
}

void ParameterManager::Insert( const core::String& system, const core::String& key, const ParameterDesc& desc )
{    
    if ( !sSystemParameterKeyValueMap.Contains( system ) )
    {
        sSystemParameterKeyValueMap.Insert( system, container::InsertOrderedMap< core::String, ParameterDesc >() );
    }

    container::InsertOrderedMap< core::String, ParameterDesc > keyValueMapForSystem = sSystemParameterKeyValueMap.Value( system );
    keyValueMapForSystem.Insert( key, desc );

    sSystemParameterKeyValueMap.Insert( system, keyValueMapForSystem );
}

void ParameterManager::Value( const core::String& system, const core::String& key, ParameterDesc& desc )
{    
    Assert( sSystemParameterKeyValueMap.Contains( system ) && sSystemParameterKeyValueMap.Value( system ).Contains( key ) );

    desc = sSystemParameterKeyValueMap.Value( system ).Value( key );
}

}