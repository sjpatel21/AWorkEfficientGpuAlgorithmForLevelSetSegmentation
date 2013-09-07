#ifndef PYTHON_SCRIPT_MANAGER_HPP
#define PYTHON_SCRIPT_MANAGER_HPP

#include "python/BoostPythonPrefix.hpp"
#include <boost/python.hpp>
#include "python/BoostPythonSuffix.hpp"

#include "core/String.hpp"
#include "core/RefCounted.hpp"

namespace core
{
    class String;
}

namespace python
{

struct ScriptDesc
{
    core::String moduleName;
    std::time_t fileModificationCode;
    double      reloadTimerSeconds;

    ScriptDesc();
};

class ScriptManager
{
public:
    static void Initialize();
    static void Terminate();

    static void Load  ( const core::String& module, const core::String& fileName );
    static void Unload( const core::String& module, const core::String& fileName );

    static void Update( double timeDeltaSeconds );

    static void                         Call( const core::String& module, const core::String& functionName );
    static void                         Call( const core::String& module, const core::String& functionName, double doubleToPass );
    template < typename T > static void Call( const core::String& module, const core::String& functionName, T*     objectToPass );

private:
    static void                  RegisterScript  ( const core::String& module, const core::String& fileName );
    static void                  UnregisterScript( const core::String& module, const core::String& fileName );

    static boost::python::object GetModuleNamespace( const core::String& module );
    static boost::python::object GetFunctionFromNamespace( const core::String& module, const core::String& functionName );

    static void                  ErrorHandler();
};

template < typename T > void ScriptManager::Call( const core::String& module, const core::String& functionName, T* objectToPass )
{
    // if the reference count of the passed in object is zero, then python will set it to 1 when it gets passed to the
    // function, and then back to 0 when the function is finished, which will delete the object.  in general, this means
    // that you can't call this method from a constructor.
    Assert( objectToPass != NULL );
    Assert( objectToPass->GetReferenceCount() > 0 );

    try
    {
        boost::python::object function = GetFunctionFromNamespace( module, functionName );
#if defined(PLATFORM_WIN32)
        function( objectToPass->ToSharedPtr< T >() );
#elif defined(PLATFORM_OSX)
        core::RefCounted* objectRefCounted = dynamic_cast< core::RefCounted * >( objectToPass );
        if( objectRefCounted != NULL )
            function( objectRefCounted->ToSharedPtr< T >() );
#endif
    }
    catch( boost::python::error_already_set const& )
    {
        ErrorHandler();
    }
}

}

#endif