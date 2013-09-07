#ifndef VOLUME_FILE_DESC_HPP
#define VOLUME_FILE_DESC_HPP

#include "core/String.hpp"

#include "math/Vector3.hpp"

struct VolumeFileDesc
{
    math::Vector3 upDirection;

    core::String  fileName;

    unsigned int  numVoxelsX;
    unsigned int  numVoxelsY;
    unsigned int  numVoxelsZ;

    float         zAnisotropy;

    unsigned char numBytesPerVoxel;

    bool          isSigned;

    VolumeFileDesc();
};

#endif