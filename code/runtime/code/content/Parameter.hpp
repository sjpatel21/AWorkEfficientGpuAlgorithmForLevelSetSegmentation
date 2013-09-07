#ifndef CONTENT_PARAMETER_HPP
#define CONTENT_PARAMETER_HPP

#include "core/String.hpp"

#include "content/ParameterManager.hpp"

namespace content
{

template< typename V > class Parameter
{
public:
    Parameter( const core::String& system, const core::String& key );
    ~Parameter();

    V             GetValue() const;
    void          SetValue( V value );

private:
    core::String mSystem;
    core::String mKey;
};

template< typename V > Parameter< V >::Parameter( const core::String& system, const core::String& key ) :
mSystem( system ),
mKey   ( key )
{
}

template< typename V > Parameter< V >::~Parameter()
{
}

template< typename V > V Parameter< V >::GetValue() const
{
    return ParameterManager::GetParameter< V >( mSystem, mKey );
};

template< typename V > void Parameter< V >::SetValue( V value )
{
    ParameterManager::SetParameter( mSystem, mKey, value );
};

}

#endif