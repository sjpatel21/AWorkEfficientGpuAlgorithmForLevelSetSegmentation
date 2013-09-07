#include "lefohn/LefohnTestData.hpp"
#include "lefohn/LefohnSegmentationSimulator.hpp"
#include "lefohn/ArrayIndexing.hpp"
#include "lefohn/LefohnCoordinates.hpp"

#include "core/String.hpp"

#include "assert.h"

namespace lefohn
{

VolumeTestCase::VolumeTestCase(int volumeDimension, int tileSize) :
    volume                (NULL),
    mVolumeDimension    (volumeDimension),
    mTileSize            (tileSize)
{
    volume = new float[volumeDimension*volumeDimension*volumeDimension];
    memset(volume, 0, sizeof(float) * volumeDimension * volumeDimension * volumeDimension);
}

void VolumeTestCase::AssertActivePhysicalTileNumber( int numberOfVirtualTiles, lefohn::SegmentationSimulator* simulator )
{
    assert( numberOfVirtualTiles == GetSimulatorInversePageTableSize(simulator));
}

lefohn::VirtualTile* VolumeTestCase::GetSimulatorVirtualTiles( lefohn::SegmentationSimulator* simulator )
{
    return simulator->mVirtualMemoryTiles;
}

lefohn::PhysicalTile** VolumeTestCase::GetSimulatorPhysicalTiles( lefohn::SegmentationSimulator* simulator )
{
    return simulator->mInversePageTable->mPhysicalTiles;
}

lefohn::InversePageTable* VolumeTestCase::GetSimulatorInversePageTable( lefohn::SegmentationSimulator* simulator )
{
    return simulator->mInversePageTable;
}

int VolumeTestCase::GetSimulatorInversePageTableSize(lefohn::SegmentationSimulator* simulator)
{
    lefohn::InversePageTable* inverseTable = simulator->mInversePageTable;
    return inverseTable->CalculateNumberPhysicalTilesStored();
}

lefohn::PageTable* VolumeTestCase::GetSimulatorPageTable( lefohn::SegmentationSimulator* simulator )
{
    return simulator->mPageTable;
}

lefohn::PhysicalCoordinate VolumeTestCase::GetVirtualTilePhysicalAddress(lefohn::VirtualTile* virtualTile)
{
    return virtualTile->mPhysicalPageAddress;
}

NoneActiveVolumeTestCase::NoneActiveVolumeTestCase( float volumeValue, int volumeDimension, int tileSize ) :
    VolumeTestCase(volumeDimension, tileSize)
{
    for(int i = 0; i < mVolumeDimension*mVolumeDimension*mVolumeDimension; ++i)
        volume[i] = volumeValue;
}

void NoneActiveVolumeTestCase::AssertActiveTiles( lefohn::SegmentationSimulator* simulator )
{
    lefohn::PhysicalTile** physicalTiles = GetSimulatorPhysicalTiles(simulator);
    for(int i = 0; i < mTileSize*mTileSize*mVolumeDimension; ++i)
    {
        lefohn::VirtualTile* curVirtualTile = &GetSimulatorVirtualTiles(simulator)[i];
        lefohn::PhysicalTile* curPhysicalTile = physicalTiles[i];
        assert(curVirtualTile->active == false);
        assert(curPhysicalTile == NULL);
    }
}

AllActiveAlternatingVolumeTestCase::AllActiveAlternatingVolumeTestCase( int volumeDimension, int tileSize ) :
    VolumeTestCase( volumeDimension, tileSize )
{
    float val1 = 1.9999999f;
    float val2 = 2.0000000f;
    // Fill with alternating values.
    for(int i = 0; i < mVolumeDimension*mVolumeDimension*mVolumeDimension; ++i)
    {
        volume[i] = i % 2 == 1 ? val2 : val1;
    }
}

void AllActiveAlternatingVolumeTestCase::AssertActiveTiles( lefohn::SegmentationSimulator* simulator )
{
    lefohn::InversePageTable* inversePageTable = GetSimulatorInversePageTable(simulator);
    lefohn::VirtualTile* curVirtualTile = NULL;
    lefohn::PhysicalTile* curPhysicalTile = NULL;

    for(int i = 0; i < mTileSize*mTileSize*mVolumeDimension; ++i)
    {
        curVirtualTile = &GetSimulatorVirtualTiles(simulator)[i];
        assert(curVirtualTile->active == true);

        curPhysicalTile = inversePageTable->GetPhysicalTile(GetVirtualTilePhysicalAddress(curVirtualTile));
        assert( curPhysicalTile->GetVirtualAddress() != lefohn::VirtualCoordinate::INVALID_COORDINATE );
    }
}

BoundaryCubeVolumeTestCase::BoundaryCubeVolumeTestCase( int volumeDimension, int tileSize ) :
VolumeTestCase( volumeDimension, tileSize )
{
    // Draw a square along edges of each layer.
    for(int z = 0; z < volumeDimension; ++z)
    {
        for(int y = 0; y < volumeDimension; ++y)
        {
            for(int x = 0; x < volumeDimension; ++x)
            {
                // Fill sides only
                if( y != 0 && y != volumeDimension - 1 )
                {
                    if( x == 0 || x == volumeDimension - 1)
                    {
                        volume[Get1DIndexFrom3DIndex(z, y, x, volumeDimension, volumeDimension)] = 1.0f;
                    }
                }
                else    // Fill top and bottom
                {
                    volume[Get1DIndexFrom3DIndex(z, y, x, volumeDimension, volumeDimension)] = 1.0f;
                }

            }
        }
    }
}

void BoundaryCubeVolumeTestCase::AssertActiveTiles( lefohn::SegmentationSimulator* simulator )
{
    lefohn::InversePageTable* inversePageTable = GetSimulatorInversePageTable(simulator);

    for(int z = 0; z < mVolumeDimension; ++z)
    {
        for(int y = 0; y < mTileSize; ++y)
        {
            for(int x = 0; x < mTileSize; ++x)
            {
                lefohn::VirtualTile* curVirtualTile = &GetSimulatorVirtualTiles(simulator)[Get1DIndexFrom3DIndex(z,y,x,mTileSize,mTileSize)];
                lefohn::PhysicalTile* curPhysicalTile = inversePageTable->GetPhysicalTile(GetVirtualTilePhysicalAddress(curVirtualTile));
                
                // Check sides
                if( y != 0 && y != mTileSize - 1 )
                {
                    if( x == 0 || x == mTileSize - 1)
                    {
                        assert(curVirtualTile->active == true);
                        assert( curPhysicalTile->GetVirtualAddress() != lefohn::VirtualCoordinate::INVALID_COORDINATE );
                    }
                    else
                    {
                        assert(curVirtualTile->active == false);
                        assert( curPhysicalTile == NULL );
                    }
                }
                else    // Check top/bottom
                {
                    assert(curVirtualTile->active == true);
                    assert(curPhysicalTile->GetVirtualAddress() != lefohn::VirtualCoordinate::INVALID_COORDINATE );
                }
            }
        }
    }
}

SolidCubeOnTileBoundariesTestCase::SolidCubeOnTileBoundariesTestCase(lefohn::VirtualCoordinate upperLeftVirtualCoordinate,
                                                                     int volumeDimension, int tileSize) :
VolumeTestCase(volumeDimension, tileSize),
mUpperLeftTileX    (upperLeftVirtualCoordinate[math::X]),
mUpperLeftTileY    (upperLeftVirtualCoordinate[math::Y]),
mUpperLeftTileZ    (upperLeftVirtualCoordinate[math::Z]),
mSize            (tileSize)
{
    assert(mUpperLeftTileZ >= 0);
    assert(mUpperLeftTileY >= 0);
    assert(mUpperLeftTileX >= 0);
    assert(mUpperLeftTileZ+mSize <= volumeDimension);
    assert(mUpperLeftTileY*tileSize+mSize <= volumeDimension);
    assert(mUpperLeftTileX*tileSize+mSize <= volumeDimension);

    // Create the solidCube to fill a volume of tiles exactly.
    for(int z = mUpperLeftTileZ; z < mUpperLeftTileZ+mSize; ++z)
    {
        for(int y = mUpperLeftTileY*mSize; y < mUpperLeftTileY*mSize+mSize; ++y)
        {
            for(int x = mUpperLeftTileX*mSize; x < mUpperLeftTileX*mSize+mSize; ++x)
            {
                volume[Get1DIndexFrom3DIndex(z, y, x, mVolumeDimension, mVolumeDimension)] = 1.0f;
            }
        }
    }
}

void SolidCubeOnTileBoundariesTestCase::AssertActiveTiles(lefohn::SegmentationSimulator *simulator)
{
    const int numTiles = mVolumeDimension / mTileSize;

    lefohn::InversePageTable* inversePageTable = GetSimulatorInversePageTable(simulator);

    for(int z = 0; z < mVolumeDimension; ++z)
    {
        for(int y = 0; y < numTiles; ++y)
        {
            for(int x = 0; x < numTiles; ++x)
            {
                int offset = Get1DIndexFrom3DIndex(z, y, x, numTiles, numTiles);

                lefohn::VirtualTile* curVirtualTile = &GetSimulatorVirtualTiles(simulator)[offset];
                lefohn::PhysicalTile* curPhysicalTile = inversePageTable->GetPhysicalTile(GetVirtualTilePhysicalAddress(curVirtualTile));

                if( x >= mUpperLeftTileX - 1 && x <= mUpperLeftTileX + 1 &&
                    y >= mUpperLeftTileY - 1 && y <= mUpperLeftTileY + 1 &&
                    z >= mUpperLeftTileZ - 1 && z <= mUpperLeftTileZ + mSize )
                {
                    assert(curVirtualTile->active == true);
                    assert( curPhysicalTile->GetVirtualAddress() != lefohn::VirtualCoordinate::INVALID_COORDINATE );
                }
                else
                {
                    assert( curVirtualTile->active == false);
                    assert( curPhysicalTile == NULL );
                }
            }
        }
    }
}

}
