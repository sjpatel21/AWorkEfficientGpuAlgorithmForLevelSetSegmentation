#ifndef CONTENT_ASSET_HPP
#define CONTENT_ASSET_HPP

#include "core/String.hpp"
#include "core/RefCounted.hpp"

namespace content
{

class Asset : public core::RefCounted
{
friend class Loader;

public:
    Asset();

    bool IsFinalized() const;

protected:
    ~Asset();

    // useful for debugging
    void SetDebugName( const core::String& debugName );

    // useful for disallowing mutator methods after initialization
    void SetFinalized( bool finalized );

    core::String mDebugName;
    bool         mFinalized;
};

}

#endif