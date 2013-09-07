#ifndef VOLUME_SAVER_HPP
#define VOLUME_SAVER_HPP

#include <boost/filesystem.hpp>

#include "VolumeDesc.hpp"

struct VolumeFileDesc;

class VolumeSaver
{
public:
    static void SaveVolume( const core::String& fileName, const VolumeDesc& volumeDesc );

private:
    static void SaveVolumeAsDicom( const boost::filesystem::path& dicomPath, const VolumeDesc& volumeDesc );

    VolumeSaver();
    ~VolumeSaver();
};




#endif // VOLUME_SAVER_HPP
