#ifndef CONTAINER_Array_HPP
#define CONTAINER_Array_HPP

#include <QtCore/QVector>

#include "container/ForEach.hpp"

namespace container
{

template< typename T > class Array
{
public:

    class ConstIterator
    {
    friend class Array;

    public:
        const T&                            operator *  () const;
        const T* const                      operator -> () const;
        bool                                operator == ( typename Array< T >::ConstIterator& other ) const;
        bool                                operator != ( typename Array< T >::ConstIterator& other ) const;
        typename Array< T >::ConstIterator& operator ++ ( int ); // the int means post-increment (e.g. i++)

    private:
        typename QVector< T >::ConstIterator mConstIterator;
    };

    Array();
    Array( QVector< T > QVector );
    ~Array();

    Array< T >& operator =  ( const Array< T >& other );
    bool        operator == ( const Array< T >& other ) const;

    int                                Size        () const;
    int                                IndexOf     ( const T& item, int from = 0 ) const;
    bool                               Contains    ( const T& item  ) const;
    void                               Append      ( const T& item  );
    void                               Clear       ();
    void                               Remove      ( int      index );
    void                               RemoveAll   ( const T& item  );
    void                               RemoveFirst ( const T& item  );
    void                               RemoveLast  ( const T& item  );
    const T                            At          ( int      index ) const;
    void                               Reserve     ( int      capacity );

    typename Array< T >::ConstIterator ConstBegin  () const;
    typename Array< T >::ConstIterator ConstEnd    () const;

private:
    QVector< T > mArray;
};


template< typename T > const T& Array< T >::ConstIterator::operator * () const
{
    return *mConstIterator;
};

template< typename T > const T* const Array< T >::ConstIterator::operator -> () const
{
    return &( *mConstIterator );
};

template< typename T > bool Array< T >::ConstIterator::operator == ( typename Array< T >::ConstIterator& other ) const
{
    return mConstIterator == other.mConstIterator;
};

template< typename T > bool Array< T >::ConstIterator::operator != ( typename Array< T >::ConstIterator& other ) const
{
    return mConstIterator != other.mConstIterator;
};

// the int in the argument Array means post-increment (e.g. i++)
template< typename T > typename Array< T >::ConstIterator& Array< T >::ConstIterator::operator ++ ( int )
{
    mConstIterator++;
    return *this;
};


template< typename T > Array< T >::Array()
{
}

template< typename T > Array< T >::Array( QVector< T > QVector ) :
mArray( QVector )
{
}

template< typename T > Array< T >::~Array()
{
}

template< typename T > Array< T >& Array< T >::operator = ( const Array< T >& other )
{
    mArray = other.mArray;
    return *this;
};

template< typename T > bool Array< T >::operator == ( const Array< T >& other ) const
{
    return mArray == other.mArray;
};

template< typename T > int Array< T >::Size() const
{
    return mArray.size();
}

template< typename T > int Array< T >::IndexOf( const T& item, int from ) const
{
    return mArray.indexOf( item, from );
}

template< typename T > bool Array< T >::Contains( const T& item ) const
{
    return mArray.contains( item );
}

template< typename T > void Array< T >::Append( const T& item )
{
    mArray.append( item );
}

template< typename T > void Array< T >::Clear()
{
    mArray.clear();
}

template< typename T > void Array< T >::Remove( int index )
{
    mArray.removeAt( index );
}

template< typename T > void Array< T >::RemoveAll( const T& item )
{
    mArray.removeAll( item );
}

template< typename T > void Array< T >::RemoveFirst( const T& item )
{
    mArray.removeFirst( item );
}

template< typename T > void Array< T >::RemoveLast( const T& item )
{
    mArray.removeLast( item );
}

template< typename T > const T Array< T >::At( int index ) const
{
    return mArray.at( index );
}

template< typename T > void Array< T >::Reserve( int capacity )
{
    mArray.reserve( capacity );
}

template< typename T > typename Array< T >::ConstIterator Array< T >::ConstBegin() const
{
    ConstIterator constIterator;
    constIterator.mConstIterator = mArray.constBegin();

    return constIterator;
}

template< typename T > typename Array< T >::ConstIterator Array< T >::ConstEnd() const
{
    ConstIterator constIterator;
    constIterator.mConstIterator = mArray.constEnd();

    return constIterator;
}

}

#endif