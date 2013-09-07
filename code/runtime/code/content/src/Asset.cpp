#include "content/Asset.hpp"

#include "content/LoadManager.hpp"

namespace content
{

Asset::Asset() :
mDebugName( "DEFAULT_DEBUG_NAME" ),
mFinalized( true )
{
}

Asset::~Asset()
{
}

bool Asset::IsFinalized() const
{
    return mFinalized;
}

// useful for debugging
void Asset::SetDebugName( const core::String& debugName )
{
    mDebugName = debugName;
};

// useful for disallowing mutator methods after initialization
void Asset::SetFinalized( bool finalized )
{
    mFinalized = finalized;
};

}