#include "VolumeDesc.hpp"

VolumeDesc::VolumeDesc() :
    upDirection           ( 0, 0, 1 ),
    voxelHeightMillimeters( 0.0 ),
    voxelWidthMillimeters ( 0.0 ),
    voxelDepthMillimeters ( 0.0 ),
    volumeData            ( NULL ),
    numVoxelsY            ( 0 ),
    numVoxelsZ            ( 0 ),
    zAnisotropy           ( 1.0f ),
    minValue              ( 0.0f ),
    maxValue              ( 0.0f ),
    numBytesPerVoxel      ( 0 ),
    isSigned              ( false )
{
};

VolumeDesc::VolumeDesc( const VolumeDesc& other ) :
upDirection           ( other.upDirection ),
voxelHeightMillimeters( other.voxelHeightMillimeters ),
voxelWidthMillimeters ( other.voxelWidthMillimeters ),
voxelDepthMillimeters ( other.voxelDepthMillimeters ),
volumeData            ( other.volumeData ),
numVoxelsX            ( other.numVoxelsX ),
numVoxelsY            ( other.numVoxelsY ),
numVoxelsZ            ( other.numVoxelsZ ),
zAnisotropy           ( other.zAnisotropy ),
minValue              ( other.minValue ),
maxValue              ( other.maxValue ),
numBytesPerVoxel      ( other.numBytesPerVoxel ),
isSigned              ( other.isSigned ),
filePaths             ( other.filePaths )
{
};