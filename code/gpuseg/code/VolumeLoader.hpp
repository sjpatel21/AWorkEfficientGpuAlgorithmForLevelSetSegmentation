#ifndef VOLUME_LOADER_HPP
#define VOLUME_LOADER_HPP

#include <boost/filesystem.hpp>

#include "VolumeDesc.hpp"

struct VolumeFileDesc;

class VolumeLoader
{
public:
    static VolumeDesc LoadVolume  ( const VolumeFileDesc& volumeFileDesc );
    static void       UnloadVolume( VolumeDesc&           volumeDesc );

private:
    static VolumeDesc LoadVolumeDicom( const boost::filesystem::path& dicomPath, const VolumeFileDesc& volumeFileDesc );
    static VolumeDesc LoadVolumeRaw  ( const boost::filesystem::path& rawPath,   const VolumeFileDesc& volumeFileDesc );

    VolumeLoader();
    ~VolumeLoader();
};

#endif