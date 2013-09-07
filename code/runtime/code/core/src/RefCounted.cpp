#include "core/RefCounted.hpp"

#include "core/Assert.hpp"
#include "core/MemoryMonitor.hpp"

namespace core
{

RefCounted::RefCounted() :
mReferenceCount( 0 )
{
    MemoryMonitor::RegisterObject( this );
}

RefCounted::~RefCounted()
{
    MemoryMonitor::UnregisterObject( this );
}

void RefCounted::AddRef()
{
    ++mReferenceCount;
}

void RefCounted::Release()
{
    Assert( mReferenceCount > 0 );

    --mReferenceCount;
    
    if ( 0 == mReferenceCount )
    {
        delete this;
    }
}

bool IsReferenceCountedSlow( RefCounted* object )
{
    return ( dynamic_cast< RefCounted* >( object ) != NULL );
}

void AssignRefHelper( RefCounted** left, RefCounted* right )
{
    Assert( left != NULL );
    Assert( *left == NULL || IsReferenceCountedSlow( *left ) );
    Assert( right == NULL || IsReferenceCountedSlow( right ) );

    if ( *left == right )
    {
        return;
    }

    if ( *left != NULL )
    {
        ( *left )->Release();
    }

    *left = right;

    if ( left != NULL )
    {
        if ( ( *left ) != NULL )
        {
            ( *left )->AddRef();
        }
    }
}

int RefCounted::GetReferenceCount() const
{
    return mReferenceCount;
}

}