#include "content/Inventory.hpp"

#include <FCollada.h> 
#include <FCDocument/FCDocument.h>

#include "core/Assert.hpp"
#include "core/String.hpp"

#include "container/Map.hpp"

#include "content/Asset.hpp"

namespace content
{

Inventory::Inventory()
{
}

Inventory::~Inventory()
{
    Assert( mNameAssetMap.Size() == 0 );
}

void Inventory::Insert( const core::String& assetName, Asset* asset )
{
    Assert( asset->IsFinalized() );
    Assert( !mNameAssetMap.Contains( assetName ) );

#ifndef ENABLE_ASSET_RELOADING
    Assert( !mNameAssetMap.Contains( assetName ) );
#endif

    asset->AddRef();
    mNameAssetMap.Insert( assetName, asset );
}

void Inventory::Clear()
{
    foreach ( Asset* asset, mNameAssetMap )
    {
        asset->Release();
    }

    mNameAssetMap.Clear();
}

bool Inventory::Contains( const core::String& assetName ) const
{
    return mNameAssetMap.Contains( assetName );
}

}