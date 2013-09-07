#include "core/MemoryMonitor.hpp"

#include <QtCore/QList>

#include "core/Config.hpp"

#include "core/RefCounted.hpp"
#include "core/Assert.hpp"

namespace core
{

static QList< const RefCounted* > sRefCountedObjects;
static bool                       sInitialized = false;

void MemoryMonitor::Initialize()
{
    sInitialized = true;
}

void MemoryMonitor::Terminate()
{
    sInitialized = false;
    Assert( sRefCountedObjects.count() == 0 );
}

void MemoryMonitor::RegisterObject( const RefCounted* object )
{
    if ( sInitialized )
    {
        Assert( !sRefCountedObjects.contains( object ) );

        sRefCountedObjects.append( object );
    }
}

void MemoryMonitor::UnregisterObject( const RefCounted* object )
{
    if ( sInitialized )
    {
        sRefCountedObjects.removeAll( object );
    }
}

}