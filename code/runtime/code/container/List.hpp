#ifndef CONTAINER_LIST_HPP
#define CONTAINER_LIST_HPP

#include <QtCore/QList>

#include "container/ForEach.hpp"

namespace container
{

template< typename T > class List
{
public:

    class ConstIterator
    {
    friend class List;

    public:
        const T&                           operator *  () const;
        const T* const                     operator -> () const;
        bool                               operator == ( typename List< T >::ConstIterator& other ) const;
        bool                               operator != ( typename List< T >::ConstIterator& other ) const;
        typename List< T >::ConstIterator& operator ++ ( int ); // the int means post-increment (e.g. i++)

    private:
        typename QList< T >::ConstIterator mConstIterator;
    };

    List();
    List( QList< T > qList );
    ~List();

    List< T >& operator =  ( const List< T >& other );
    bool       operator == ( const List< T >& other ) const;

    int                               Size        () const;
    int                               IndexOf     ( const T& item, int from = 0 ) const;
    bool                              Contains    ( const T& item  ) const;
    void                              Append      ( const T& item  );
    void                              Prepend     ( const T& item  );
    void                              Clear       ();
    void                              Replace     ( int index, const T& item );
    void                              Remove      ( int      index );
    void                              RemoveAll   ( const T& item  );
    void                              RemoveFirst ( const T& item  );
    void                              RemoveLast  ( const T& item  );
    const T                           At          ( int      index ) const;
    typename List< T >::ConstIterator ConstBegin  () const;
    typename List< T >::ConstIterator ConstEnd    () const;

private:
    QList< T > mList;
};


template< typename T > const T& List< T >::ConstIterator::operator * () const
{
    return *mConstIterator;
};

template< typename T > const T* const List< T >::ConstIterator::operator -> () const
{
    return &( *mConstIterator );
};

template< typename T > bool List< T >::ConstIterator::operator == ( typename List< T >::ConstIterator& other ) const
{
    return mConstIterator == other.mConstIterator;
};

template< typename T > bool List< T >::ConstIterator::operator != ( typename List< T >::ConstIterator& other ) const
{
    return mConstIterator != other.mConstIterator;
};

// the int in the argument list means post-increment (e.g. i++)
template< typename T > typename List< T >::ConstIterator& List< T >::ConstIterator::operator ++ ( int )
{
    mConstIterator++;
    return *this;
};


template< typename T > List< T >::List()
{
}

template< typename T > List< T >::List( QList< T > qList ) :
mList( qList )
{
}

template< typename T > List< T >::~List()
{
}

template< typename T > List< T >& List< T >::operator = ( const List< T >& other )
{
    mList = other.mList;
    return *this;
};

template< typename T > bool List< T >::operator == ( const List< T >& other ) const
{
    return mList == other.mList;
};

template< typename T > int List< T >::Size() const
{
    return mList.size();
}

template< typename T > int List< T >::IndexOf( const T& item, int from ) const
{
    return mList.indexOf( item, from );
}

template< typename T > bool List< T >::Contains( const T& item ) const
{
    return mList.contains( item );
}

template< typename T > void List< T >::Append( const T& item )
{
    mList.append( item );
}

template< typename T > void List< T >::Prepend( const T& item )
{
    mList.prepend( item );
}

template< typename T > void List< T >::Replace( int index, const T& item )
{
    mList.replace( index, item );
}

template< typename T > void List< T >::Clear()
{
    mList.clear();
}

template< typename T > void List< T >::Remove( int index )
{
    mList.removeAt( index );
}

template< typename T > void List< T >::RemoveAll( const T& item )
{
    mList.removeAll( item );
}

template< typename T > void List< T >::RemoveFirst( const T& item )
{
    mList.removeFirst( item );
}

template< typename T > void List< T >::RemoveLast( const T& item )
{
    mList.removeLast( item );
}

template< typename T > const T List< T >::At( int index ) const
{
    return mList.at( index );
}

template< typename T > typename List< T >::ConstIterator List< T >::ConstBegin() const
{
    ConstIterator constIterator;
    constIterator.mConstIterator = mList.constBegin();

    return constIterator;
}

template< typename T > typename List< T >::ConstIterator List< T >::ConstEnd() const
{
    ConstIterator constIterator;
    constIterator.mConstIterator = mList.constEnd();

    return constIterator;
}

}

#endif