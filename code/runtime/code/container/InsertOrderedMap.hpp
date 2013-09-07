#ifndef CONTAINER_INSERT_ORDERED_MAP_HPP
#define CONTAINER_INSERT_ORDERED_MAP_HPP

#include <QtCore/QMap>

#include "container/ForEach.hpp"
#include "container/List.hpp"

namespace container
{

template< class K, class V > class InsertOrderedMap
{
public:

    class ConstIterator
    {    
        friend class InsertOrderedMap;

    public:
        const V&                                          operator *  () const;
        const V* const                                    operator -> () const;
        bool                                              operator == ( typename InsertOrderedMap< K, V >::ConstIterator& other ) const;
        bool                                              operator != ( typename InsertOrderedMap< K, V >::ConstIterator& other ) const;
        typename InsertOrderedMap< K, V >::ConstIterator& operator ++ ( int ); // the int means post-increment (e.g. i++)
        const K&                                          Key()   const;
        const V&                                          Value() const; 

    private:
#if defined(PLATFORM_WIN32)
        const typename QMap < K, V >*      mMap;
        mutable typename V                         mValue;
        mutable typename K                         mKey;
#elif defined(PLATFORM_OSX)
        const QMap < K, V >*               mMap;
        mutable V                         mValue;
        mutable K                         mKey;
#endif



        typename QList< K >::ConstIterator mConstIterator;
    };

    bool                                             Contains  ( const K& key )                  const;
    void                                             Insert    ( const K& key, const V& value );
    void                                             Remove    ( const K& key );
    const V                                          Value     ( const K& key )                  const;
    void                                             Clear     ();
    typename InsertOrderedMap< K, V >::ConstIterator ConstBegin()                                const;
    typename InsertOrderedMap< K, V >::ConstIterator ConstEnd  ()                                const;
    int                                              Size      ()                                const;
    container::List< K >                             Keys      ()                                const;

private:
    QList< K >    mList;
    QMap < K, V > mMap;

};

template< class K, class V > const V& InsertOrderedMap< K, V >::ConstIterator::operator * () const
{
    mValue = mMap->value( *mConstIterator );
    return mValue;
};

template< class K, class V > const V* const InsertOrderedMap< K, V >::ConstIterator::operator -> () const
{
    return &( *mConstIterator );
};

template< class K, class V > bool InsertOrderedMap< K, V >::ConstIterator::operator == ( typename InsertOrderedMap< K, V >::ConstIterator& other ) const
{
    return mConstIterator == other.mConstIterator;
};

template< class K, class V > bool InsertOrderedMap< K, V >::ConstIterator::operator != ( typename InsertOrderedMap< K, V >::ConstIterator& other ) const
{
    return mConstIterator != other.mConstIterator;
};

// the int means post-increment (e.g. i++)
template< class K, class V > typename InsertOrderedMap< K, V >::ConstIterator& InsertOrderedMap< K, V >::ConstIterator::operator ++ ( int )
{
    mConstIterator++;
    return *this;
};

template< class K, class V > const K& InsertOrderedMap< K, V >::ConstIterator::Key() const
{
    mKey = mMap->key( *mConstIterator );
    return mKey;
};

template< class K, class V > const V& InsertOrderedMap< K, V >::ConstIterator::Value() const
{
    mValue = mMap->value( *mConstIterator );
    return mValue;
};

template< class K, class V > bool InsertOrderedMap< K, V >::Contains( const K& key ) const
{
    Assert( mList.contains( key ) == mMap.contains( key ) );

    return mMap.contains( key );
};

template< class K, class V > void InsertOrderedMap< K, V >::Insert( const K& key, const V& value )
{
    mMap.insert( key, value );

    if ( !mList.contains( key ) )
    {
        mList.append( key );
    }
};

template< class K, class V > void InsertOrderedMap< K, V >::Remove( const K& key )
{
    mMap.remove( key );

    mList.removeAll( key );
};

template< class K, class V > const V InsertOrderedMap< K, V >::Value( const K& key ) const
{
    return mMap.value( key );
};

template< class K, class V > void InsertOrderedMap< K, V >::Clear()
{
    mList.clear();
    mMap.clear();
};

template< class K, class V > typename InsertOrderedMap< K, V >::ConstIterator InsertOrderedMap< K, V >::ConstBegin() const
{
    ConstIterator constIterator;
    constIterator.mConstIterator = mList.constBegin();
    constIterator.mMap           = &mMap;

    return constIterator;
};

template< class K, class V > typename InsertOrderedMap< K, V >::ConstIterator InsertOrderedMap< K, V >::ConstEnd() const
{
    ConstIterator constIterator;
    constIterator.mConstIterator = mList.constEnd();
    constIterator.mMap           = &mMap;

    return constIterator;
};

template< class K, class V > int InsertOrderedMap< K, V >::Size() const
{
    Assert( mMap.size() == mList.size() );

    return mMap.size();
};

template< class K, class V > List< K > InsertOrderedMap< K, V >::Keys() const
{
    return container::List< K >( mList );
};

}

#endif