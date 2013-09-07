#ifndef CONTENT_LOADER_HPP
#define CONTENT_LOADER_HPP

#include "core/RefCounted.hpp"

#include "content/Ref.hpp"

class FCDocument;

namespace content
{

class Inventory;

class Loader : public core::RefCounted
{
public:
    Loader();

    virtual void Load  ( Inventory* inventory, FCDocument* document );
    virtual void Unload( Inventory* inventory );

protected:
    virtual ~Loader();

    // Helper functions for loaders.  Several classes are friends with Loader, but since Loader the
    // friend relationship is not inherited by derived loader classes, we provide these loader helper classes. 
    void                        Insert      ( Inventory* inventory, const core::String& assetName, Asset* asset );
    void                        SetDebugName( Asset* asset, const core::String& debugName );
    void                        SetFinalized( Asset* asset, bool finalized );
    template< typename T > void SetFinalized( content::Ref< T > asset, bool finalized );
};

template< typename T > void Loader::SetFinalized( content::Ref< T > asset, bool finalized )
{
    asset->SetFinalized( finalized );
};

}

#endif