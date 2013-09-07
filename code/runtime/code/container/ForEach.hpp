#ifndef CONTAINER_FOR_EACH_HPP
#define CONTAINER_FOR_EACH_HPP

class IForEachContainer {};

template < typename T > class ForEachContainer : public IForEachContainer
{
public:
    ForEachContainer( const T& container );

    bool Finished() const;
    inline void Iterate();

    bool FinishedMiddle() const;
    void IterateMiddle();

    bool FinishedInner() const;
    void IterateInner();

    typename T::ConstIterator GetIterator();

private:
    int                               mMiddleLoopCount;
    int                               mInnerLoopCount;
    mutable typename T::ConstIterator mIterator;
    mutable typename T::ConstIterator mEnd;
};

// helper methods for "foreach" macro
template < typename T > inline ForEachContainer< T >  CreateForEachContainer( const T& t );
template < typename T > inline ForEachContainer< T >* GetForEachContainer   ( IForEachContainer* base, T* );
template < typename T > inline T*                     TypeSafeDummy         ( const T& );


template < typename T > ForEachContainer< T >::ForEachContainer( const T& container ) :
    mInnerLoopCount ( 0 ),
    mMiddleLoopCount( 0 ),
    mIterator       ( container.ConstBegin() ),
    mEnd            ( container.ConstEnd()   )
{
};

template < typename T > bool ForEachContainer< T >::Finished() const
{
    return mIterator == mEnd;
};

template < typename T > void ForEachContainer< T >::Iterate()
{
    mIterator++;
    mMiddleLoopCount = 0;
    mInnerLoopCount  = 0;
};

template < typename T > bool ForEachContainer< T >::FinishedMiddle() const
{
    return mMiddleLoopCount > 0;
};

template < typename T > void ForEachContainer< T >::IterateMiddle()
{
    mMiddleLoopCount++;
    mInnerLoopCount = 0;
};

template < typename T > bool ForEachContainer< T >::FinishedInner() const
{
    return mInnerLoopCount > 0;
};

template < typename T > void ForEachContainer< T >::IterateInner()
{
    mInnerLoopCount++;
};

template < typename T > typename T::ConstIterator ForEachContainer< T >::GetIterator()
{
    return mIterator;
};


// used to create an indirection so the for loop doesn't have to know the container type
template < typename T > inline ForEachContainer< T > CreateForEachContainer( const T& t )
{
    return ForEachContainer< T >( t );
}

// gets a pointer to the derived container from the base container
template < typename T > inline ForEachContainer< T >* GetForEachContainer( const IForEachContainer* base, T* )
{
    return static_cast< ForEachContainer< T >* >( const_cast<IForEachContainer*>(base) );
}

// always returns a pointer of type T that equals null
template < typename T > inline T* TypeSafeDummy( const T& )
{
    return NULL;
}

#ifdef foreach
#undef foreach
#endif

#define foreach(loopVariable,container)                                                                              \
    for (                                                                                                            \
        const IForEachContainer& forEachContainer = CreateForEachContainer( container );                                   \
        !( GetForEachContainer( &forEachContainer, TypeSafeDummy( container ) )->Finished() );                       \
         ( GetForEachContainer( &forEachContainer, TypeSafeDummy( container ) )->Iterate() ) )                       \
        for (                                                                                                        \
            loopVariable = *( GetForEachContainer( &forEachContainer, TypeSafeDummy( container ) )->GetIterator() ); \
            !( GetForEachContainer( &forEachContainer, TypeSafeDummy( container ) )->FinishedInner() );              \
             ( GetForEachContainer( &forEachContainer, TypeSafeDummy( container ) )->IterateInner() ) )

#define foreach_key_value(loopKey,loopValue,container)                                                                         \
    for (                                                                                                                      \
        const IForEachContainer& forEachContainer = CreateForEachContainer( container );                                             \
        !( GetForEachContainer( &forEachContainer, TypeSafeDummy( container ) )->Finished() );                                 \
         ( GetForEachContainer( &forEachContainer, TypeSafeDummy( container ) )->Iterate() ) )                                 \
        for ( loopKey = ( GetForEachContainer( &forEachContainer, TypeSafeDummy( container ) )->GetIterator().Key() );         \
            !( GetForEachContainer( &forEachContainer, TypeSafeDummy( container ) )->FinishedMiddle() );                       \
             ( GetForEachContainer( &forEachContainer, TypeSafeDummy( container ) )->IterateMiddle() ) )                       \
            for ( loopValue = ( GetForEachContainer( &forEachContainer, TypeSafeDummy( container ) )->GetIterator().Value() ); \
                !( GetForEachContainer( &forEachContainer, TypeSafeDummy( container ) )->FinishedInner() );                    \
                 ( GetForEachContainer( &forEachContainer, TypeSafeDummy( container ) )->IterateInner() ) )

#endif

#define boost_foreach BOOST_FOREACH