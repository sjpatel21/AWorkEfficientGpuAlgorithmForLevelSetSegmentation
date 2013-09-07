#include "content/Loader.hpp"

#include <FCollada.h>
#include <FCDocument/FCDocument.h>

#include "content/Inventory.hpp"

namespace content
{

Loader::Loader()
{
}

Loader::~Loader()
{
}

void Loader::Load( Inventory*, FCDocument* )
{
}

void Loader::Unload( Inventory* )
{
}

// Helper functions for loaders.  Several classes are friends with Loader, but since Loader the
// friend relationship is not inherited by derived loader classes, we provide these loader helper classes. 
void Loader::Insert( Inventory* inventory, const core::String& assetName, Asset* asset )
{
    inventory->Insert( assetName, asset );
}

void Loader::SetDebugName( Asset* asset, const core::String& debugName )
{
    asset->SetDebugName( debugName );
}

void Loader::SetFinalized( Asset* asset, bool finalized )
{
    asset->SetFinalized( finalized );
}

}