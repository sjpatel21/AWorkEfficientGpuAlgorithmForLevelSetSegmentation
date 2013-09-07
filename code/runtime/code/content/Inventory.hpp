#ifndef CONTENT_INVENTORY_HPP
#define CONTENT_INVENTORY_HPP

#include "core/String.hpp"
#include "core/RefCounted.hpp"

#include "container/Map.hpp"
#include "container/List.hpp"

#include "content/Config.hpp"

namespace content
{

class Asset;
template < typename T > class Ref;

class Inventory : public core::RefCounted 
{

// we want the loader classes to be able to add to the inventory, but no other classes.
friend class Loader;

// we want the load manager to clear the inventory, but no other classes
friend class LoadManager;

template < typename T > friend class Ref;
public:
    Inventory();
    ~Inventory();

    // note - this method is defined in Ref.hpp to resolve a circular dependency
    template < typename T > Ref< T >                    Find   ( const core::String& assetName );
    template < typename T > container::List< Ref< T > > FindAll();

    bool Contains( const core::String& assetName ) const;

private:
    template < typename T > T* FindPrivate( const core::String& assetName ) const;
    void                       Insert     ( const core::String& assetName, Asset* asset );
    void                       Clear      ();

    container::Map< core::String, Asset* > mNameAssetMap;
};

template < typename T > T* Inventory::FindPrivate( const core::String& assetName ) const
{
    Assert( mNameAssetMap.Contains( assetName ) );

    core::RefCounted* asset = mNameAssetMap.Value( assetName );

    Assert( dynamic_cast< T* >( asset ) != NULL );

    return static_cast< T* >( asset );
}

}

#endif