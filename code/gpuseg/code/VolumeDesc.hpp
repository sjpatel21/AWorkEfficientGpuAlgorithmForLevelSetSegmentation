#ifndef VOLUME_DESC_HPP
#define VOLUME_DESC_HPP

#include <boost/filesystem.hpp>

#include "core/String.hpp"

#include "math/Vector3.hpp"

#include "container/List.hpp"

struct VolumeDesc
{
    math::Vector3                              upDirection;
                                    
    double                                     voxelHeightMillimeters;
    double                                     voxelWidthMillimeters;
    double                                     voxelDepthMillimeters;
                                    
    void*                                      volumeData;
                                    
    unsigned int                               numVoxelsX;
    unsigned int                               numVoxelsY;
    unsigned int                               numVoxelsZ;
                                    
    float                                      zAnisotropy;
    float                                      minValue;
    float                                      maxValue;
                                    
    unsigned char                              numBytesPerVoxel;
                                    
    bool                                       isSigned;

    container::List< boost::filesystem::path > filePaths;

    VolumeDesc();
    VolumeDesc( const VolumeDesc& other );
};

#endif