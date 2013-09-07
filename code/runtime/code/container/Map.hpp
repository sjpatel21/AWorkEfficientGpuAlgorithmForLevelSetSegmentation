#ifndef CONTAINER_MAP_HPP
#define CONTAINER_MAP_HPP

#include <QtCore/QMap>

#include "container/ForEach.hpp"
#include "container/List.hpp"

namespace container
{

template< class K, class V > class Map
{
public:

    class ConstIterator
    {    
    friend class Map;

    public:
        const V&                             operator *  () const;
        const V* const                       operator -> () const;
        bool                                 operator == ( typename Map< K, V >::ConstIterator& other ) const;
        bool                                 operator != ( typename Map< K, V >::ConstIterator& other ) const;
        typename Map< K, V >::ConstIterator& operator ++ ( int ); // the int means post-increment (e.g. i++)
        const K&                             Key();
        const V&                             Value(); 

    private:
        typename QMap< K, V >::ConstIterator mConstIterator;
    };

    bool                                Contains  ( const K& key )                  const;
    void                                Insert    ( const K& key, const V& value );
    void                                Remove    ( const K& key );
    const V                             Value     ( const K& key )                  const;
    void                                Clear     ();
    typename Map< K, V >::ConstIterator ConstBegin()                                const;
    typename Map< K, V >::ConstIterator ConstEnd  ()                                const;
    int                                 Size      ()                                const;
    container::List< K >                Keys      ()                                const;

private:
    QMap< K, V > mMap;
};



template< class K, class V > const V& Map< K, V >::ConstIterator::operator * () const
{
    return *mConstIterator;
};

template< class K, class V > const V* const Map< K, V >::ConstIterator::operator -> () const
{
    return &( *mConstIterator );
};

template< class K, class V > bool Map< K, V >::ConstIterator::operator == ( typename Map< K, V >::ConstIterator& other ) const
{
    return mConstIterator == other.mConstIterator;
};

template< class K, class V > bool Map< K, V >::ConstIterator::operator != ( typename Map< K, V >::ConstIterator& other ) const
{
    return mConstIterator != other.mConstIterator;
};

// the int means post-increment (e.g. i++)
template< class K, class V > typename Map< K, V >::ConstIterator& Map< K, V >::ConstIterator::operator ++ ( int )
{
    mConstIterator++;
    return *this;
};

template< class K, class V > const K& Map< K, V >::ConstIterator::Key()
{
    return mConstIterator.key();
};

template< class K, class V > const V& Map< K, V >::ConstIterator::Value()
{
    return mConstIterator.value();
};




template< class K, class V > bool Map< K, V >::Contains( const K& key ) const
{
    return mMap.contains( key );
};

template< class K, class V > void Map< K, V >::Insert( const K& key, const V& value )
{
    mMap.insert( key, value );
};

template< class K, class V > void Map< K, V >::Remove( const K& key )
{
    mMap.remove( key );
};

template< class K, class V > const V Map< K, V >::Value( const K& key ) const
{
    return mMap.value( key );
};

template< class K, class V > void Map< K, V >::Clear()
{
    mMap.clear();
};

template< class K, class V > typename Map< K, V >::ConstIterator Map< K, V >::ConstBegin() const
{
    ConstIterator constIterator;
    constIterator.mConstIterator = mMap.constBegin();

    return constIterator;
};

template< class K, class V > typename Map< K, V >::ConstIterator Map< K, V >::ConstEnd() const
{
    ConstIterator constIterator;
    constIterator.mConstIterator = mMap.constEnd();

    return constIterator;
};

template< class K, class V > int Map< K, V >::Size() const
{
    return mMap.size();
};

template< class K, class V > container::List< K > Map< K, V >::Keys() const
{
    return container::List< K >( mMap.keys() );
};

}

#endif