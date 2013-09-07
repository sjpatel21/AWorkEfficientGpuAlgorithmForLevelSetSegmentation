#include "VolumeFileDesc.hpp"

VolumeFileDesc::VolumeFileDesc() :
    upDirection     ( 0, 0, 1 ),
    fileName        ( "" ),
    numVoxelsX      ( 0 ),
    numVoxelsY      ( 0 ),
    numVoxelsZ      ( 0 ),
    zAnisotropy     ( 1.0f ),
    numBytesPerVoxel( 0 ),
    isSigned        ( false )
{
};