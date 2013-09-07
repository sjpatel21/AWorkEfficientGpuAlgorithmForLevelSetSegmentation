#ifndef CORE_FUNCTOR_HPP
#define CORE_FUNCTOR_HPP

#ifdef PLATFORM_WIN32
#define NOMINMAX
#include <windows.h>
#endif

#include <boost/shared_ptr.hpp>

#include "core/RefCounted.hpp"


#define SET_CALLBACK_METHOD_HELPER SetCallbackMethodHelper
#define GET_CALLBACK_METHOD_HELPER GetCallbackMethodHelper
#define CALLBACK_METHOD_HELPER     CallbackMethodHelper

#define CORE_SET_CALLBACK_METHOD(setCallbackMethod)                                                \
    ;                                                                                              \
    inline void setCallbackMethod##SET_CALLBACK_METHOD_HELPER ( boost::shared_ptr< core::FunctorContainer > functorContainer ) \
    {                                                                                              \
        setCallbackMethod ( functorContainer->mFunctor );                                          \
    };

#define CORE_GET_CALLBACK_METHOD(getCallbackMethod)                                                                                       \
    ;                                                                                                                                     \
    inline boost::shared_ptr< core::FunctorContainer > getCallbackMethod##GET_CALLBACK_METHOD_HELPER () const                             \
    {                                                                                                                                     \
        boost::shared_ptr< core::FunctorContainer > functorContainer =                                                                    \
        boost::shared_ptr< core::FunctorContainer >( new core::FunctorContainer() );                                                      \
        AssignRef( functorContainer->mFunctor, getCallbackMethod () );                                                                    \
        return functorContainer;                                                                                                          \
    };

#if defined(PLATFORM_WIN32)

#define CORE_CALLBACK_METHOD(thisType,callbackMethod)                                            \
    ;                                                                                            \
    inline boost::shared_ptr< core::FunctorContainer > callbackMethod##CALLBACK_METHOD_HELPER () \
    {                                                                                            \
        boost::shared_ptr< core::FunctorContainer > functorContainer =                           \
            boost::shared_ptr< core::FunctorContainer >( new core::FunctorContainer() );         \
        functorContainer->mFunctor = new core::Functor< thisType >( this, & callbackMethod );    \
        functorContainer->mFunctor->AddRef();                                                    \
        return functorContainer;                                                                 \
    };

#elif defined(PLATFORM_OSX)

#define CORE_CALLBACK_METHOD(thisType,callbackMethod, qualifiedCallbackMethod)                                            \
    ;                                                                                            \
    inline boost::shared_ptr< core::FunctorContainer > callbackMethod##CALLBACK_METHOD_HELPER () \
    {                                                                                            \
        boost::shared_ptr< core::FunctorContainer > functorContainer =                           \
        boost::shared_ptr< core::FunctorContainer >( new core::FunctorContainer() );         \
        functorContainer->mFunctor = new core::Functor< thisType >( this, & qualifiedCallbackMethod );    \
        functorContainer->mFunctor->AddRef();                                                    \
        return functorContainer;                                                                 \
};

#endif

namespace core
{

// Abstract IFunctor base class
class IFunctor : public RefCounted
{
public:
    // this function will call the object::function() to which the functor points, with the specified user data
    virtual void Call( void* userData ) const = 0;

    // this function will return the object to which the functor points
    virtual void* GetObject() const = 0;

protected:
    // this function is virtual so deleting AbstractFunctor pointers will call the correct destructor.
    // this function is protected because AbstractFunctors are reference counted.  No one client should
    // control an AbstractFunctor's lifetime.
    virtual ~IFunctor() {};
};

// Concrete Functor class
template < typename T > class Functor : public IFunctor
{
public:
    // create a functor that points to a specific object, and a specific member function of that object
    Functor( T* object, void ( T::*function )( void* ) );
    ~Functor() {};

    // this function will call the object::function() to which the functor points, with the specified user data
    void Call( void* userData ) const;

    // this function will return the object to which the functor points
    void* GetObject() const;

private:
    T*   mObject;                        // pointer to object of class T
    void ( T::*mFunction )( void* );     // pointer to one of T's functions
};

// create a functor that points to a specific object, and a specific member function of that object
template < typename T > Functor< T >::Functor( T* object, void ( T::*function )( void* ) ) :
mObject  ( object ),
mFunction( function )
{
};

// this function will call the object::function() to which the functor points, with the specified user data
template < typename T > void Functor< T >::Call( void* userData ) const
{
    ( *mObject.*mFunction )( userData );
};

// this function will return the object to which the functor points
template < typename T > void* Functor< T >::GetObject() const
{
    return mObject;
};


class FunctorContainer
{
public:
    inline FunctorContainer()  : mFunctor( NULL ) {};
    inline ~FunctorContainer() { AssignRef( mFunctor, NULL ); };

    IFunctor* mFunctor;
};

}

#endif