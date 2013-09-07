#ifndef CONTENT_REF_HPP
#define CONTENT_REF_HPP

#include <boost/shared_ptr.hpp>

#include "core/String.hpp"
#include "core/RefCounted.hpp"

#include "content/Config.hpp"
#include "content/Asset.hpp"
#include "content/Inventory.hpp"

#define RefIsNull( ref ) content::RefIsNullHelper( ref )

namespace content
{

class Inventory;

template< typename T > class Ref
{

template < typename T2 > friend bool RefIsNullHelper( const Ref< T2 >& ref );

public:
    Ref();
    Ref( Inventory* inventory, const core::String& assetName );
    Ref( const Ref& other );
    ~Ref();

    Ref< T >& operator  = ( const Ref< T >& other );
    const T&  operator  * () const;
    T* const  operator -> () const;

    T* const               GetRawPointer() const;
    boost::shared_ptr< T > GetSharedPtr () { 
#if defined(PLATFORM_WIN32)
        return GetRawPointer()->ToSharedPtr< T >(); 
#elif defined(PLATFORM_OSX)
        core::RefCounted* assetRefCounted = dynamic_cast< core::RefCounted * >( GetRawPointer() );
        if( assetRefCounted != NULL )
            return assetRefCounted->ToSharedPtr< T >();
                
        return boost::shared_ptr< T >();
#endif
    }

private:
    bool IsNull() const
    {
#ifdef ENABLE_ASSET_RELOADING
        return mInventory == NULL;
#endif

#ifndef ENABLE_ASSET_RELOADING
        return mAsset == NULL;
#endif
    };

#ifdef ENABLE_ASSET_RELOADING
    core::String mAssetName;
    Inventory*   mInventory;
#endif

#ifndef ENABLE_ASSET_RELOADING
    T* mAsset;
#endif

};

template < typename T > bool RefIsNullHelper( const content::Ref< T >& ref )
{
    return ref.IsNull();
}

//
// since content::Ref functionality depends on content::Inventory,
// we have to put the content::Ref functionality after the content::Inventory
// declaration in this file to avoid a circular dependency
//
#ifdef ENABLE_ASSET_RELOADING

template< typename T > Ref< T >::Ref() :
mAssetName( "DEFAULT_ASSET_NAME" ),
mInventory( NULL )
{
};

template< typename T > Ref< T >::Ref( Inventory* inventory, const core::String& assetName ) : 
mAssetName( assetName ),
mInventory( NULL )
{
    Assert( inventory != NULL );

    AssignRef( mInventory, inventory );
};

template< typename T > Ref< T >::Ref( const Ref& other ) :
mAssetName( other.mAssetName ),
mInventory( other.mInventory )
{
    //
    // since Ref< T > acts like a pointer, the usual AssignRef semantics aren't quite what 
    // we want.  we expect the inventory to be identical in most cases, but we want to addref
    // anyway since another Ref< T > object means another reference to the inventory.  this is
    // why we do an addref after the assignref.
    //
    bool identicalInventories = ( mInventory == other.mInventory );

    AssignRef( mInventory, other.mInventory );

    if ( mInventory != NULL && identicalInventories )
    {
        mInventory->AddRef();
    }
};

template< typename T > Ref< T >::~Ref()
{
    AssignRef( mInventory, NULL );
};

template< typename T > typename Ref< T >& Ref< T >::operator = ( const Ref< T >& other )
{
    if ( mAssetName != other.mAssetName || mInventory != other.mInventory )
    {
        mAssetName = other.mAssetName;


        //
        // since Ref< T > acts like a pointer, the usual AssignRef semantics aren't quite what 
        // we want.  we expect the inventory to be identical in most cases, but we want to addref
        // anyway since another Ref< T > object means another reference to the inventory.  this is
        // why we do an addref after the assignref.
        //
        bool identicalInventories = ( mInventory == other.mInventory );

        AssignRef( mInventory, other.mInventory );

        if ( mInventory != NULL && identicalInventories )
        {
            mInventory->AddRef();
        }
    }

    return *this;
}

template< typename T > const T& Ref< T >::operator * () const
{
    return *( mInventory->FindPrivate< T >( mAssetName ) );
};

template< typename T > T* const Ref< T >::operator -> () const
{
    return mInventory->FindPrivate< T >( mAssetName );
};

template< typename T > T* const Ref< T >::GetRawPointer() const
{
    return mInventory->FindPrivate< T >( mAssetName );
};

#endif



#ifndef ENABLE_ASSET_RELOADING

template< typename T > Ref< T >::Ref() :
mAsset( NULL )
{
};

template< typename T > Ref< T >::Ref( Inventory* inventory, const core::String& assetName ) : 
mAsset( NULL )
{
    Assert( inventory != NULL );

    mAsset = inventory->FindPrivate< T >( assetName );
    mAsset->AddRef();
};

template< typename T > Ref< T >::Ref( const Ref& other ) :
mAsset( NULL )
{
    AssignRef( mAsset, other.mAsset );
};

template< typename T > Ref< T >::~Ref()
{
    AssignRef( mAsset, NULL );
};

#if defined(PLATFORM_WIN32)
template< typename T > typename Ref< T >& Ref< T >::operator = ( const Ref< T >& other )
#elif defined(PLATFORM_OSX)
template< typename T > Ref< T >& Ref< T >::operator = ( const Ref< T >& other )
#endif    
{
    if ( mAsset != other.mAsset )
    {
        AssignRef( mAsset, other.mAsset );
    }

    return *this;
}


template< typename T > const T& Ref< T >::operator * () const
{
    Assert( mAsset != NULL );

    return *( mAsset );
};

template< typename T > T* const Ref< T >::operator -> () const
{
    Assert( mAsset != NULL );

    return mAsset;
};

template< typename T > T* const Ref< T >::GetRawPointer() const
{
    Assert( mAsset != NULL );

    return mAsset;
};

#endif


//
// this method is in Ref.h instead of Inventory.h to resolve a circular dependency issue.
// otherwise Inventory.h will have to include Ref.h, which is bad because Ref.h already needs
// to include Inventory.h.  This is the only Inventory method that needs Ref.h, and anyone
// calling this function is going to need to include Ref.h anyway.
//
template < typename T > Ref< T > Inventory::Find( const core::String& assetName )
{
    Assert( mNameAssetMap.Contains( assetName ) );

    return Ref< T >( this, assetName );
}

template < typename T > container::List< Ref< T > > Inventory::FindAll()
{
    container::List< Ref< T > > listOfMatchingObjects;

    foreach_key_value( const core::String& assetName, Asset* asset, mNameAssetMap )
    {
        if ( dynamic_cast< T* >( asset ) != NULL )
        {
            listOfMatchingObjects.Append( Ref< T >( this, assetName ) );
        }
    }

    return listOfMatchingObjects;
}

}

#endif