#include "content/LoadManager.hpp"

#include <boost/filesystem.hpp>

#include <FCollada.h>
#include <FCDocument/FCDocument.h>
#include <FUtils/FUError.h>

#include "core/String.hpp"
#include "core/RefCounted.hpp"
#include "core/Functor.hpp"

#include "container/List.hpp"
#include "container/Map.hpp"

#include "content/Loader.hpp"
#include "content/Inventory.hpp"
#include "content/ParameterManager.hpp"
#include "content/Config.hpp"

namespace content
{

struct AssetFileDesc
{
    std::time_t                   fileModificationCode;
    container::List< Inventory* > inventories;
};

static container::List< Loader* >                    sLoaderList;
static container::Map< core::String, AssetFileDesc > sFileNameAssetDescMap;
static core::IFunctor*                               sAssetChangedBeforeReloadCallback = NULL;
static core::IFunctor*                               sAssetChangedAfterReloadCallback  = NULL;

void LoadManager::Initialize()
{
    FCollada::Initialize();
}

void LoadManager::Terminate()
{
    AssignRef( sAssetChangedBeforeReloadCallback, NULL );
    AssignRef( sAssetChangedAfterReloadCallback,  NULL );

    UninstallLoaders();

    FCollada::Release();
}

void LoadManager::Update()
{
#ifdef ENABLE_ASSET_RELOADING

    bool fileChanged = false;

    //
    // decide if any files need reloading
    //
    foreach_key_value ( const core::String& fileName, AssetFileDesc assetFileDesc, sFileNameAssetDescMap )
    {
        std::time_t modificationCode = boost::filesystem::last_write_time( boost::filesystem::path( fileName.ToStdString() ) );

        if ( modificationCode != assetFileDesc.fileModificationCode )
        {
            foreach ( Inventory* inventory, assetFileDesc.inventories )
            {
                fileChanged = true;
            }
        }
    }

    //
    // if so, then call the before-reload callback
    //
    if ( fileChanged && sAssetChangedBeforeReloadCallback != NULL )
    {
        sAssetChangedBeforeReloadCallback->Call( NULL );
    }

    //
    // now actually reload
    //
    foreach_key_value ( const core::String& fileName, AssetFileDesc assetFileDesc, sFileNameAssetDescMap )
    {
        boost::filesystem::path filePath( fileName.ToStdString() );
        std::time_t             modificationCode = boost::filesystem::last_write_time( filePath );

        if ( modificationCode != assetFileDesc.fileModificationCode )
        {
            UpdateTimeStamp( filePath );

            foreach ( Inventory* inventory, assetFileDesc.inventories )
            {
                inventory->Clear();
                LoadHelper( filePath, inventory );
            }
        }
    }
    
    //
    // now call the after-reload callback
    //
    if ( fileChanged && sAssetChangedAfterReloadCallback != NULL )
    {
        sAssetChangedAfterReloadCallback->Call( NULL );
    }
#endif
}

void LoadManager::InstallLoader( Loader* loader )
{
    loader->AddRef();
    sLoaderList.Append( loader );
}

void LoadManager::Load( const core::String& fileName, Inventory* inventory )
{
    boost::filesystem::path filePath( fileName.ToStdString() );

    LoadHelper( filePath, inventory );
    RegisterInventory( filePath, inventory );
}

void LoadManager::Unload( Inventory* inventory )
{
    // give each loader a chance to examine the file
    foreach ( Loader* loader, sLoaderList )
    {
        loader->Unload( inventory );
    }

    UnregisterInventory( inventory );

    inventory->Clear();
}

void LoadManager::UninstallLoaders()
{
    foreach ( Loader* loader, sLoaderList )
    {
        loader->Release();
    }

    sLoaderList.Clear();
}

void LoadManager::SetAssetChangedBeforeReloadCallback( core::IFunctor* assetChangedBeforeReloadCallback )
{
    AssignRef( sAssetChangedBeforeReloadCallback, assetChangedBeforeReloadCallback );
}

void LoadManager::SetAssetChangedAfterReloadCallback( core::IFunctor* assetChangedAfterReloadCallback )
{
    AssignRef( sAssetChangedAfterReloadCallback, assetChangedAfterReloadCallback );
}

void LoadManager::RegisterInventory( const boost::filesystem::path& filePath, Inventory* inventory )
{
    core::String fileName = filePath.native_file_string();

    if ( sFileNameAssetDescMap.Contains( fileName ) )
    {
        AssetFileDesc desc = sFileNameAssetDescMap.Value( fileName );

        desc.inventories.Append( inventory );
        desc.fileModificationCode = boost::filesystem::last_write_time( filePath );

        sFileNameAssetDescMap.Insert( fileName, desc );
    }
    else
    {
        AssetFileDesc desc;

        desc.inventories.Append( inventory );
        desc.fileModificationCode = boost::filesystem::last_write_time( filePath );

        sFileNameAssetDescMap.Insert( fileName, desc );
    }
}

void LoadManager::UnregisterInventory( Inventory* inventory )
{
    container::List< core::String > fileNamesToRemove;

    foreach_key_value ( const core::String& fileName, AssetFileDesc assetFileDesc, sFileNameAssetDescMap )
    {
        assetFileDesc.inventories.RemoveAll( inventory );

        if ( assetFileDesc.inventories.Size() == 0 )
        {
            if ( !fileNamesToRemove.Contains( fileName ) )
            {
                fileNamesToRemove.Append( fileName );
            }
        }
    }

    foreach ( const core::String& fileName, fileNamesToRemove )
    {
        sFileNameAssetDescMap.Remove( fileName );
    }
}

void LoadManager::LoadHelper( const boost::filesystem::path& filePath, Inventory* inventory )
{
    FUErrorSimpleHandler errorHandler( FUError::WARNING_LEVEL );

    // load file
    FCDocument* document = FCollada::NewTopDocument();

    bool success = FCollada::LoadDocumentFromFile( document, filePath.native_file_string().c_str() );
    Assert( success );

    core::String errorString = errorHandler.GetErrorString();

    Assert( errorHandler.IsSuccessful() );

    // give each loader a chance to examine the file
    foreach ( Loader* loader, sLoaderList )
    {
        loader->Load( inventory, document );
    }

    document->Release();
    document = NULL;
}

void LoadManager::UpdateTimeStamp( const boost::filesystem::path& filePath )
{
    core::String fileName = filePath.native_file_string();

    Assert( sFileNameAssetDescMap.Contains( fileName ) );

    AssetFileDesc desc = sFileNameAssetDescMap.Value( fileName );

    desc.fileModificationCode = boost::filesystem::last_write_time( filePath );

    sFileNameAssetDescMap.Insert( fileName, desc );
}

}