#ifndef CORE_MEMORY_HPP
#define CORE_MEMORY_HPP

namespace core
{

class RefCounted;

class MemoryMonitor
{

public:
    static void Initialize();
    static void Terminate();

    static void RegisterObject  ( const RefCounted* object );
    static void UnregisterObject( const RefCounted* object );
};

}

#endif