#ifndef CORE_REFCOUNTED_HPP
#define CORE_REFCOUNTED_HPP

#include <boost/shared_ptr.hpp>
#include <boost/mem_fn.hpp>

#include "core/Assert.hpp"

namespace core
{

#define AssignRef( left, right )                                                     \
    core::AssignRefHelper( reinterpret_cast< core::RefCounted** >( & left ), right ) \

class RefCounted
{
public:
    RefCounted();
    virtual ~RefCounted();

    void AddRef();
    void Release();

    // some systems will only work if an object has already been addref'd at least once.
    // this method enables these systems to assert on failure cases.
    int GetReferenceCount() const;

    // interoperability with boost
    template < typename T > boost::shared_ptr< T > ToSharedPtr();

private:
    int mReferenceCount;
};

void AssignRefHelper( RefCounted** left, RefCounted* right );

template < typename T > boost::shared_ptr< T > RefCounted::ToSharedPtr()
{
    Assert( dynamic_cast< T* >( this ) != NULL );

    AddRef();
    return boost::shared_ptr< T >( static_cast< T* >( this ), boost::mem_fn( &RefCounted::Release ) );    
}

}
#endif