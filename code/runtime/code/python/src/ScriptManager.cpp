#include "python/ScriptManager.hpp"

#include <boost/filesystem.hpp>

#include "core/String.hpp"
#include "core/Assert.hpp"
#include "core/Printf.hpp"
#include "core/Time.hpp"

#include "container/Map.hpp"

#include "python/Config.hpp"
#include "python/Macros.hpp"

PYTHON_FORWARD_DECLARE_MODULE( core )
PYTHON_FORWARD_DECLARE_MODULE( math )
PYTHON_FORWARD_DECLARE_MODULE( content )
PYTHON_FORWARD_DECLARE_MODULE( rendering )
PYTHON_FORWARD_DECLARE_MODULE( python )

namespace python
{

static container::Map< core::String, python::ScriptDesc > sFileNameScriptDescMap;
static const double RELOAD_DELAY  = 0.2;
static const double NOT_RELOADING = -1.0;

ScriptDesc::ScriptDesc() :
moduleName          ( ""   ),
fileModificationCode( -1   ),
reloadTimerSeconds  ( NOT_RELOADING )
{
}

void ScriptManager::Initialize()
{
    Py_Initialize();

    PYTHON_INITIALIZE_MODULE( core )
    PYTHON_INITIALIZE_MODULE( math )
    PYTHON_INITIALIZE_MODULE( content )
    PYTHON_INITIALIZE_MODULE( rendering )
    PYTHON_INITIALIZE_MODULE( python )

    //
    // boostrap python code to redirect printing through core
    //
    core::String initializationString = "";

    initializationString += "import sys         \n";
    initializationString += "sys.stderr = core  \n";
    initializationString += "sys.stdout = core  \n";

    boost::python::api::object mainNamespace = GetModuleNamespace( "__main__" );
    boost::python::exec( initializationString.ToAscii(), mainNamespace, mainNamespace );
}

void ScriptManager::Terminate()
{
    PYTHON_TERMINATE_MODULE( python )
    PYTHON_TERMINATE_MODULE( rendering )
    PYTHON_TERMINATE_MODULE( content )
    PYTHON_TERMINATE_MODULE( math )
    PYTHON_TERMINATE_MODULE( core )

    Py_Finalize();
}

void ScriptManager::Load( const core::String& module, const core::String& fileName )
{
    try
    {
        boost::python::object mainNamespace   = GetModuleNamespace( "__main__" );
        boost::python::object moduleNamespace = GetModuleNamespace( module );
        boost::python::exec_file( fileName.ToAscii(), mainNamespace, moduleNamespace );
    }
    catch( boost::python::error_already_set const& )
    {
        ErrorHandler();
    }

    RegisterScript( module, fileName );
}

void ScriptManager::Unload( const core::String& module, const core::String& fileName )
{
    UnregisterScript( module, fileName );
}

void ScriptManager::Update( double timeDeltaSeconds )
{
#ifdef ENABLE_SCRIPT_RELOADING
    foreach_key_value ( const core::String& fileName, ScriptDesc scriptDesc, sFileNameScriptDescMap )
    {
        boost::filesystem::path filePath( fileName.ToStdString() );

        std::time_t modificationCode = boost::filesystem::last_write_time( filePath );
        
        if ( modificationCode != scriptDesc.fileModificationCode && scriptDesc.reloadTimerSeconds == NOT_RELOADING )
        {
            scriptDesc.reloadTimerSeconds = RELOAD_DELAY;
        }

        if ( scriptDesc.reloadTimerSeconds != NOT_RELOADING )
        {
            scriptDesc.reloadTimerSeconds -= timeDeltaSeconds;

            if ( scriptDesc.reloadTimerSeconds < 0.0 )
            {
                Load( scriptDesc.moduleName, fileName );
            }
        }
    }
#endif
}

void ScriptManager::Call( const core::String& module, const core::String& functionName, double doubleToPass )
{
    try
    {
        boost::python::object function = GetFunctionFromNamespace( module, functionName );
        function( doubleToPass );
    }
    catch( boost::python::error_already_set const& )
    {
        ErrorHandler();
    }
}

void ScriptManager::Call( const core::String& module, const core::String& functionName )
{
    try
    {
        boost::python::object function = GetFunctionFromNamespace( module, functionName );
        function();
    }
    catch( boost::python::error_already_set const& )
    {
        ErrorHandler();
    }
}

void ScriptManager::RegisterScript( const core::String& module, const core::String& fileName )
{
    boost::filesystem::path filePath( fileName.ToStdString() );
    std::time_t modificationCode = boost::filesystem::last_write_time( filePath );

    ScriptDesc desc;

    desc.moduleName           = module;
    desc.fileModificationCode = modificationCode;

    sFileNameScriptDescMap.Insert( fileName, desc );
}

void ScriptManager::UnregisterScript( const core::String& module, const core::String& fileName )
{
    Assert( sFileNameScriptDescMap.Contains( fileName ) );
    Assert( sFileNameScriptDescMap.Value( fileName ).moduleName == module );

    sFileNameScriptDescMap.Remove( fileName );
}

void ScriptManager::ErrorHandler()
{
    core::Printf( "\n" );
    core::Printf( "\n" );
    core::Printf( "\n" );
    core::Printf( "\n" );

    // print all other errors to stderr
    PyErr_Print();

    core::Printf( "\n" );
    core::Printf( "\n" );
    core::Printf( "\n" );
    core::Printf( "\n" );

    Assert( 0 );
}

boost::python::object ScriptManager::GetModuleNamespace( const core::String& module )
{
    boost::python::object moduleObject = boost::python::import( module.ToAscii() );
    return moduleObject.attr( "__dict__" );
}

boost::python::object ScriptManager::GetFunctionFromNamespace( const core::String& module, const core::String& functionName )
{
    boost::python::object moduleNamespace = GetModuleNamespace( module );
    return moduleNamespace[ functionName.ToAscii() ];
}

}