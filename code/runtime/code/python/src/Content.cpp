#include "python/Macros.hpp"

#include "core/String.hpp"
#include "core/RefCounted.hpp"

#include "content/LoadManager.hpp"
#include "content/ParameterManager.hpp"
#include "content/Inventory.hpp"
#include "content/Ref.hpp"

#include "rendering/Camera.hpp"
#include "rendering/Scene.hpp"

namespace python
{

PYTHON_DECLARE_CLASS_FACTORY( CreateInventory, content::Inventory )

PYTHON_MODULE_BEGIN( content )

    PYTHON_CLASS_FACTORY( "CreateInventory", CreateInventory, content::Inventory )

    PYTHON_CLASS_BEGIN( "RefRenderingScene", content::Ref< rendering::Scene > )
        PYTHON_CLASS_METHOD_CONSTRUCTOR()
        PYTHON_CLASS_METHOD_CONSTRUCTOR( content::Inventory*, const core::String& )
    PYTHON_CLASS_END

    PYTHON_CLASS_BEGIN( "RefRenderingCamera", content::Ref< rendering::Camera > )
        PYTHON_CLASS_METHOD_CONSTRUCTOR()
        PYTHON_CLASS_METHOD_CONSTRUCTOR( content::Inventory*, const core::String& )
        PYTHON_CLASS_METHOD_GET_RAW_POINTER( "GetCamera", rendering::Camera )
    PYTHON_CLASS_END

    PYTHON_CLASS_BEGIN( "Inventory", content::Inventory )
        PYTHON_CLASS_METHOD( "FindRenderingScene",  content::Inventory::Find< rendering::Scene  > )
        PYTHON_CLASS_METHOD( "FindRenderingCamera", content::Inventory::Find< rendering::Camera > )
    PYTHON_CLASS_END

    PYTHON_CLASS_BEGIN_NON_COPYABLE( "ParameterManager", content::ParameterManager )
        PYTHON_CLASS_METHOD_STATIC( "Initialize",     content::ParameterManager::Initialize )
        PYTHON_CLASS_METHOD_STATIC( "Terminate",      content::ParameterManager::Terminate )
        PYTHON_CLASS_METHOD_STATIC( "LoadParameters", content::ParameterManager::LoadParameters )
        PYTHON_CLASS_METHOD_STATIC( "SaveParameters", content::ParameterManager::SaveParameters )
    PYTHON_CLASS_END

    PYTHON_CLASS_BEGIN_NON_COPYABLE( "LoadManager", content::LoadManager )
        PYTHON_CLASS_METHOD_STATIC( "Load",   content::LoadManager::Load )
        PYTHON_CLASS_METHOD_STATIC( "Unload", content::LoadManager::Unload )
    PYTHON_CLASS_END

PYTHON_MODULE_END

}