#include "lefohn/LefohnPageTable.hpp"
#include "lefohn/LefohnSegmentationSimulator.hpp"
#include "lefohn/ArrayIndexing.hpp"
#include "lefohn/LefohnCoordinates.hpp"

#include "core/String.hpp"

namespace lefohn
{

    const int PageTable::PAGE_SIZE = SegmentationSimulator::TILE_SIZE;

PageTable::PageTable( VirtualTile* inititialVirtualTiles ) :
    mVirtualTiles        ( NULL ),
    mStaticVirtualTiles ( NULL )

{
    const int dataSize = SegmentationSimulator::DATA_SIZE;
    const int numberTilesPerRowColumn = SegmentationSimulator::NUM_TILES_PER_ROW_COLUMN_OF_DATA;
    const int numberTilesPerRowColumnSquared = SegmentationSimulator::NUM_TILES_PER_ROW_COLUMN_OF_DATA_SQUARED;

    mVirtualTiles = new VirtualTile*[ dataSize * numberTilesPerRowColumnSquared ];
    mStaticVirtualTiles = new VirtualTile[2];
    mStaticVirtualTiles[0].active = true;
    mStaticVirtualTiles[0].mPhysicalPageAddress = PhysicalCoordinate(0,0);
    mStaticVirtualTiles[1].active = true;
    mStaticVirtualTiles[1].mPhysicalPageAddress = PhysicalCoordinate(1,0);

    // Initialize all active/inactive tiles.
    for( int z = 0; z < dataSize; ++z )
    {
        for( int y = 0; y < numberTilesPerRowColumn; ++y )
        {
            for( int x = 0; x < numberTilesPerRowColumn; ++x )
            {
                int offset = Get1DIndexFrom3DIndex(z, y, x, numberTilesPerRowColumn, numberTilesPerRowColumn );
                VirtualTile* externalTile = &inititialVirtualTiles[offset];
                
                if( externalTile->active )
                {
                    mVirtualTiles[offset] = externalTile;
                }
                else    // Just set it to the first virtual tile, we will probably handle
                {        // inside/outside later.
                    mVirtualTiles[offset] = &mStaticVirtualTiles[0];
                }

            }
        }
    }

}

PageTable::~PageTable()
{
    if( mVirtualTiles )
    {
        // We only handle cleanup of the array, not what the array points to.
        delete [] mVirtualTiles;
        mVirtualTiles = NULL;
    }
    if( mStaticVirtualTiles )
    {
        delete [] mStaticVirtualTiles;
        mStaticVirtualTiles = NULL;
    }
}

PhysicalCoordinate PageTable::GetPhysicalAddress(VirtualTile* virtualTile)
{
    return virtualTile->mPhysicalPageAddress;
}

PhysicalCoordinate PageTable::GetPhysicalAddress( const VirtualCoordinate& virtualAddress )
{
    const int numTilesPerRowColumn = SegmentationSimulator::NUM_TILES_PER_ROW_COLUMN_OF_DATA;
    int x = virtualAddress[math::X];
    int y = virtualAddress[math::Y];
    int z = virtualAddress[math::Z];
    int offset = Get1DIndexFrom3DIndex(z, y, x, numTilesPerRowColumn, numTilesPerRowColumn );

    return mVirtualTiles[offset]->mPhysicalPageAddress;
}

InversePageTable::InversePageTable() :
    mPhysicalTiles                ( NULL ),
    mCurrentPhysicalLocation    ( 0, 0 )
{
    const int dataSize = SegmentationSimulator::DATA_SIZE;
    const int numberTilesPerRowColumn = SegmentationSimulator::NUM_TILES_PER_ROW_COLUMN_OF_DATA;
    const int numberTilesPerRowColumnSquared = SegmentationSimulator::NUM_TILES_PER_ROW_COLUMN_OF_DATA_SQUARED;
    mPhysicalTiles = new PhysicalTile*[ dataSize * numberTilesPerRowColumnSquared ];
    memset( mPhysicalTiles, 0, sizeof(PhysicalTile*) * dataSize * numberTilesPerRowColumnSquared );
}

InversePageTable::~InversePageTable()
{
    if( mPhysicalTiles )
    {
        // Delete the physical tiles this points to.
        // This was added because I don't yet know where physical tiles are stored
        // other than here, so to make sure there's no leaks while we build the system
        // the inverse page table will clean up.
        const int numPhysicalTiles = SegmentationSimulator::DATA_SIZE *
            SegmentationSimulator::NUM_TILES_PER_ROW_COLUMN_OF_DATA_SQUARED;
        for(int i = 0; i < numPhysicalTiles; ++i)
        {
            PhysicalTile* curTile = mPhysicalTiles[i];
            if(curTile)
            {
                delete curTile;
                curTile = NULL;
            }
            else    // All data MUST be contigious!
                break;
        }
        delete [] mPhysicalTiles;
        mPhysicalTiles = NULL;
    }
}

// Would like to throw an exception but let's hack some error code...
PhysicalCoordinate InversePageTable::Insert(PhysicalTile* physicalTile, const VirtualCoordinate& virtualAddress )
{
    const int numTilesPerRowColumn = SegmentationSimulator::NUM_TILES_PER_ROW_COLUMN;

    int x = mCurrentPhysicalLocation[math::X];
    int y = mCurrentPhysicalLocation[math::Y];

    if( x < numTilesPerRowColumn && x >= 0 && y >= 0)
    {

        mPhysicalTiles[Get1DIndexFrom2DIndex(y, x, numTilesPerRowColumn)] = physicalTile;
        physicalTile->mPhysicalPageAddress = mCurrentPhysicalLocation;
        physicalTile->mVirtualPageNumber = virtualAddress;
        IncrementCurrentPhysicalLocation();
        return physicalTile->mPhysicalPageAddress;
    }

    return PhysicalCoordinate(-1,-1);
}

PhysicalTile* InversePageTable::GetPhysicalTile( const PhysicalCoordinate& physicalAddress )
{

    const int numTilesPerRowColumn = SegmentationSimulator::NUM_TILES_PER_ROW_COLUMN;

    PhysicalTile* returnTile = NULL;

    int x = physicalAddress[math::X];
    int y = physicalAddress[math::Y];

    if( x < numTilesPerRowColumn && x >=0 && y >= 0 )
    {
        returnTile = mPhysicalTiles[Get1DIndexFrom2DIndex(y, x, numTilesPerRowColumn)];
    }

    return returnTile;
}

int InversePageTable::CalculateNumberPhysicalTilesStored()
{
    int x = static_cast<int>(mCurrentPhysicalLocation[math::X]);
    int y = static_cast<int>(mCurrentPhysicalLocation[math::Y]);
    return y * SegmentationSimulator::NUM_TILES_PER_ROW_COLUMN + x;
}

void InversePageTable::IncrementCurrentPhysicalLocation()
{
    mCurrentPhysicalLocation[math::X] += 1.0f;

    if( mCurrentPhysicalLocation[math::X] >= SegmentationSimulator::NUM_TILES_PER_ROW_COLUMN )
    {
        mCurrentPhysicalLocation[math::X] = 0.0f;
        mCurrentPhysicalLocation[math::Y] += 1.0f;
    }
}

VirtualTile::VirtualTile() :
    active                ( false ),
        mPhysicalPageAddress ( PhysicalCoordinate::INVALID_COORDINATE ),
        mVirtualPageNumber    ( VirtualCoordinate:: INVALID_COORDINATE )
{

}

VirtualTile::~VirtualTile()
{

}

PhysicalTile::PhysicalTile() :
mPhysicalPageAddress        ( PhysicalCoordinate::INVALID_COORDINATE ),
    mVirtualPageNumber        ( VirtualCoordinate::INVALID_COORDINATE )
{

}

PhysicalTile::~PhysicalTile()
{

}

}
