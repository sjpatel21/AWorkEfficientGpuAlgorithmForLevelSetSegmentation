#include "python/Macros.hpp"
#include "python/ScriptManager.hpp"

#include "core/String.hpp"

PYTHON_MODULE_BEGIN( python )

    PYTHON_CLASS_BEGIN_NON_COPYABLE( "ScriptManager", python::ScriptManager )
        PYTHON_CLASS_METHOD_STATIC( "Load",           python::ScriptManager::Load )
        PYTHON_CLASS_METHOD_STATIC( "Unload",         python::ScriptManager::Unload )
    PYTHON_CLASS_END

PYTHON_MODULE_END
