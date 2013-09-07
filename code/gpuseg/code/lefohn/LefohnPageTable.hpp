#ifndef LEFOHN_PAGE_TABLE_HPP
#define LEFOHN_PAGE_TABLE_HPP

#include "LefohnCoordinates.hpp"

namespace lefohn
{

class PageTable;
class InversePageTable;
class VirtualTile;
class PhysicalTile;

class PageTable
{
public:
    PageTable( VirtualTile* initialVirtualTiles );
    ~PageTable();

    // Returns the physical address corresponding to the virtual address.
    PhysicalCoordinate GetPhysicalAddress( const VirtualCoordinate& virtualAddress );
    PhysicalCoordinate GetPhysicalAddress( VirtualTile* virtualTile);

    static const int PAGE_SIZE;

private:
    friend class VolumeTestCase;
    VirtualTile** mVirtualTiles;
    VirtualTile*    mStaticVirtualTiles;

};

class InversePageTable
{
public:
    InversePageTable();
    ~InversePageTable();

    // Returns the physical address of the tile
    // Returns PhysicalCoordinate(-1,-1) if the table is full.
    PhysicalCoordinate Insert(PhysicalTile* physicalTile, const VirtualCoordinate& virtualAddress);

    // Returns a pointer to a PhysicalTile in the InversePageTable
    // Returns NULL if no PhysicalTile exists at physicalAddress
    PhysicalTile* GetPhysicalTile( const PhysicalCoordinate& physicalAddress );

    int CalculateNumberPhysicalTilesStored();

private:
    friend class VolumeTestCase;
    void IncrementCurrentPhysicalLocation();    

    PhysicalTile** mPhysicalTiles;
    PhysicalCoordinate mCurrentPhysicalLocation;

};

class VirtualTile
{
public:
    VirtualTile();
    ~VirtualTile();

    void SetVirtualPageNumber( const VirtualCoordinate& virtualPageNumber ){mVirtualPageNumber = virtualPageNumber;}
    void SetPhysicalAddress( const PhysicalCoordinate& physicalAddress ){mPhysicalPageAddress = physicalAddress;}

    bool active;

private:
    friend class PageTable;
    friend class VolumeTestCase;
    PhysicalCoordinate mPhysicalPageAddress;    // Actually a 2D vector.
    VirtualCoordinate mVirtualPageNumber;

};

class PhysicalTile
{
public:
    PhysicalTile();
    ~PhysicalTile();

    PhysicalCoordinate GetPhysicalAddress() const {return mPhysicalPageAddress;}
    VirtualCoordinate GetVirtualAddress() const {return mVirtualPageNumber;}

private:
    friend class InversePageTable;
    friend class VolumeTestCase;
    PhysicalCoordinate mPhysicalPageAddress;    // Actually a 2D vector.
    VirtualCoordinate mVirtualPageNumber;
};

}

#endif