#ifndef CONTENT_PARAMETER_MANAGER_HPP
#define CONTENT_PARAMETER_MANAGER_HPP

#include "math/Vector3.hpp"
#include "math/Vector4.hpp"
#include "math/Matrix44.hpp"

#include "container/InsertOrderedMap.hpp"
#include "container/Map.hpp"
#include "container/List.hpp"

#include "content/ParameterType.hpp"
#include "content/ParameterDesc.hpp"

namespace core
{
    class String;
}

namespace content
{

class ParameterManager
{
public:
    static void Initialize();
    static void Terminate();

    static void LoadParameters( const core::String& parametersFile );
    static void SaveParameters( const core::String& parametersFile );

    static void InsertNewParameter( const core::String& system, const core::String& key, const math::Matrix44& value, float min, float max );
    static void InsertNewParameter( const core::String& system, const core::String& key, const math::Vector4& value, float min, float max );
    static void InsertNewParameter( const core::String& system, const core::String& key, const math::Vector3& value, float min, float max );
    static void InsertNewParameter( const core::String& system, const core::String& key, float                value, float min, float max );

    static void SetParameter( const core::String& system, const core::String& key, const math::Matrix44& value );
    static void SetParameter( const core::String& system, const core::String& key, const math::Vector4& value );
    static void SetParameter( const core::String& system, const core::String& key, const math::Vector3& value );
    static void SetParameter( const core::String& system, const core::String& key, float                value );

    template< typename T > inline static T                GetParameter( const core::String& system, const core::String& key );
    
#if defined(PLATFORM_WIN32)
    template <>            static float                   GetParameter( const core::String& system, const core::String& key );
    template <>            static math::Vector3           GetParameter( const core::String& system, const core::String& key );
    template <>            static math::Vector4           GetParameter( const core::String& system, const core::String& key );
    template <>            static math::Matrix44          GetParameter( const core::String& system, const core::String& key );
#endif
    
    static ParameterType                                  GetParameterType( const core::String& system, const core::String& string );
    static float                                          GetMinimumValue ( const core::String& system, const core::String& string );
    static float                                          GetMaximumValue ( const core::String& system, const core::String& string );

    static container::List< core::String >::ConstIterator GetSystemsBegin   ();
    static container::List< core::String >::ConstIterator GetSystemsEnd     ();
    static container::List< core::String >::ConstIterator GetParametersBegin( const core::String& system );
    static container::List< core::String >::ConstIterator GetParametersEnd  ( const core::String& system );

    static void RemoveSystem( const core::String& system );

    static bool Contains( const core::String& system, const core::String& string );

private:
    static void Insert( const core::String& system, const core::String& key, const ParameterDesc& desc );
    static void Value ( const core::String& system, const core::String& key, ParameterDesc& desc );
};

//
// All valid template specializations are defined below, if not specialized then assert.
//
template< typename T > inline T ParameterManager::GetParameter( const core::String& system, const core::String& key )
{
    Assert( false );

    T t;
    return t;
};
    
template <> inline float ParameterManager::GetParameter( const core::String& system, const core::String& key )
{
    Assert( Contains( system, key ) );

    ParameterDesc existingValue;
        
    Value( system, key, existingValue );
        
    Assert( existingValue.type == ParameterType_Float );
        
    return existingValue.dataFloat;
}
    
template <> inline math::Vector3 ParameterManager::GetParameter( const core::String& system, const core::String& key )
{
    Assert( Contains( system, key ) );
        
    ParameterDesc existingValue;
        
    Value( system, key, existingValue );
        
    Assert( existingValue.type == ParameterType_Vector3 );
        
    return existingValue.dataVector3;
}
    
template <> inline math::Vector4 ParameterManager::GetParameter( const core::String& system, const core::String& key )
{
    Assert( Contains( system, key ) );
        
    ParameterDesc existingValue;
        
    Value( system, key, existingValue );
        
    Assert( existingValue.type == ParameterType_Vector4 );
        
    return existingValue.dataVector4;
}
    
template <> inline math::Matrix44 ParameterManager::GetParameter( const core::String& system, const core::String& key )
{
    Assert( Contains( system, key ) );
        
    ParameterDesc existingValue;
        
    Value( system, key, existingValue );
    
    Assert( existingValue.type == ParameterType_Matrix44 );
        
    return existingValue.dataMatrix44;
}

}

#endif