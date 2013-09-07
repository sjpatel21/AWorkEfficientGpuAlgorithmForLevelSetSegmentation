#ifndef LEFOHN_TEST_DATA_HPP
#define LEFOHN_TEST_DATA_HPP

/***
This defines a test case system for volumes passed into the SegmentationSimulator.
***/

namespace lefohn
{
    class SegmentationSimulator;
    class VirtualTile;
    class PhysicalTile;
    class InversePageTable;
    class PageTable;
    class PhysicalCoordinate;
    class VirtualCoordinate;
}

namespace math
{
    class Vector3;
}

/***
VolumeTestCase -- The interface for all volume tests.


AssertActiveTiles(simulator) -- asserts all active virtual and physical tiles are correct for the
        volume type.

assertActivePhysicalTileNumber( numberVirtualTiles, simulator) -- asserts the number of active
        virtual tiles is equal to the number of active physical tiles.

***/

namespace lefohn
{

class VolumeTestCase
{
public:
    VolumeTestCase( int volumeDimension, int tileSize );
    virtual ~VolumeTestCase(){if(volume){delete [] volume;}}

    virtual void AssertActiveTiles( lefohn::SegmentationSimulator* simulator ) = 0;
    virtual void AssertActivePhysicalTileNumber( int numberOfVirtualTiles, lefohn::SegmentationSimulator* simulator );

    float* volume;

protected:

    // The following functions allow the derived test cases private access to any lefohn type.
    lefohn::VirtualTile* GetSimulatorVirtualTiles( lefohn::SegmentationSimulator* simulator );
    lefohn::PhysicalTile** GetSimulatorPhysicalTiles( lefohn::SegmentationSimulator* simulator );
    lefohn::InversePageTable* GetSimulatorInversePageTable( lefohn::SegmentationSimulator* simulator );
    lefohn::PageTable* GetSimulatorPageTable( lefohn::SegmentationSimulator* simulator );

    lefohn::PhysicalCoordinate GetVirtualTilePhysicalAddress(lefohn::VirtualTile* virtualTile);
    int GetSimulatorInversePageTableSize(lefohn::SegmentationSimulator* simulator);
    int mVolumeDimension;
    int mTileSize;
};

/***
NoneActiveVolumeTestCase -- tests to see if no tiles are activated when the entire volume is filled with
        one value.
***/
class NoneActiveVolumeTestCase : public VolumeTestCase
{
public:
    NoneActiveVolumeTestCase( float volumeValue, int volumeDimension, int tileSize );
    virtual ~NoneActiveVolumeTestCase(){};

    virtual void AssertActiveTiles( lefohn::SegmentationSimulator* simulator );
};

/***
AllActiveAlternatingVolumeTestCase -- tests to see if every tile is activated if the volume is initialized
        with alternating values which are very close to each other in value.
***/
class AllActiveAlternatingVolumeTestCase : public VolumeTestCase
{
public:
    AllActiveAlternatingVolumeTestCase(int volumeDimension, int tileSize);
    virtual ~AllActiveAlternatingVolumeTestCase(){};

    virtual void AssertActiveTiles( lefohn::SegmentationSimulator* simulator );
};

/***
BoundaryCubeVolumeTestCase -- tests to see if a cube drawn in the volume along the edges of the volume
        only will set the appropriate tiles active.
***/
class BoundaryCubeVolumeTestCase : public VolumeTestCase
{
public:
    BoundaryCubeVolumeTestCase(int volumeDimension, int tileSize);
    virtual ~BoundaryCubeVolumeTestCase(){};

    virtual void AssertActiveTiles( lefohn::SegmentationSimulator* simulator );
};

/***
SolidCubeOnTileBoundariesTestCase -- tests to see if the correct active tiles are those tiles directly
        touching any of the tiles inside a solid cube of size 1x1x16 tiles (the tiles containing the solid cube
        volume must also be active).
***/
class SolidCubeOnTileBoundariesTestCase : public VolumeTestCase
{
public:
    SolidCubeOnTileBoundariesTestCase(lefohn::VirtualCoordinate, 
                                      int volumeDimension, int tileSize);
    virtual ~SolidCubeOnTileBoundariesTestCase(){};

    virtual void AssertActiveTiles( lefohn::SegmentationSimulator* simulator );

private:
    int mUpperLeftTileX, mUpperLeftTileY, mUpperLeftTileZ;
    int mSize;
};

}

#endif
