#ifndef CONTENT_LOAD_MANAGER_HPP
#define CONTENT_LOAD_MANAGER_HPP

#include <boost/filesystem.hpp>

#include "core/NameSpaceID.hpp"

class FCDocument;

namespace core
{
    class String;
    class IFunctor;
}

namespace content
{

class Loader;
class Inventory;

class LoadManager
{
public:
    static void Initialize();
    static void Terminate();

    static void Update();

    static void InstallLoader( Loader* loader );

    static void Load( const core::String& fileName, Inventory* inventory );
    static void Unload( Inventory* inventory );

    static void SetAssetChangedAfterReloadCallback ( core::IFunctor* assetChangedBeforeReloadCallback );
    static void SetAssetChangedBeforeReloadCallback( core::IFunctor* assetChangedBeforeReloadCallback );

private:
    static void UninstallLoaders();

    static void RegisterInventory  ( const boost::filesystem::path& filePath, Inventory* inventory );
    static void UnregisterInventory( Inventory* inventory );

    static void LoadHelper( const boost::filesystem::path& filePath, Inventory* inventory );
    static void UpdateTimeStamp( const boost::filesystem::path& filePath );

    // leave these private so no one can try to instantiate a load manager
    LoadManager();
    ~LoadManager();
};

}

#endif