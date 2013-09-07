#ifndef SRC_CUDA_COORDINATES_CU
#define SRC_CUDA_COORDINATES_CU

__device__ int ComputeIndexThread1DBlock1DTo1D()
{
    const dim3&  threadBlockDimensions        = blockDim;
    const uint3& threadBlockIndexWithinGrid   = blockIdx;
    const uint3& threadIndexWithinThreadBlock = threadIdx;

    return __umul24( threadBlockIndexWithinGrid.x, threadBlockDimensions.x ) + threadIndexWithinThreadBlock.x;
}

__device__ dim3 ComputeIndexThread3DBlock2DTo3D( dim3 volumeDimensions, int logicalNumThreadBlocksX, float invNumThreadBlocksX )
{
    const dim3&  threadBlockDimensions        = blockDim;
    const uint3& threadBlockIndexWithinGrid   = blockIdx;
    const uint3& threadIndexWithinThreadBlock = threadIdx;

    //
    // unpack the 2D grid coordinates into 3D grid coordinates based on the known 3D dimensions of the volume.
    // conceptually,
    //
    // threadBlockIndexY = threadBlockIndexXY / ( volumeDimensions.x / threadBlockDimensions.x );
    // threadBlockIndexX = threadBlockIndexXY % ( volumeDimensions.x / threadBlockDimensions.x );
    //
    const unsigned int threadBlockIndexXY = threadBlockIndexWithinGrid.x;
    const unsigned int threadBlockIndexY  = __float2uint_rd( threadBlockIndexXY * invNumThreadBlocksX );    
    const unsigned int threadBlockIndexX  = threadBlockIndexXY - __umul24( threadBlockIndexY, logicalNumThreadBlocksX );
    const unsigned int threadBlockIndexZ  = threadBlockIndexWithinGrid.y;

    //
    // now figure out the 3D voxel coordinates based on the 3D grid coordinates and 3D thread coordinates
    //
    const dim3 elementCoordinates(
        __umul24( threadBlockIndexX, threadBlockDimensions.x ) + threadIndexWithinThreadBlock.x,
        __umul24( threadBlockIndexY, threadBlockDimensions.y ) + threadIndexWithinThreadBlock.y,
        __umul24( threadBlockIndexZ, threadBlockDimensions.z ) + threadIndexWithinThreadBlock.z );

    return elementCoordinates;
}

__device__ unsigned int ComputeIndex3DTo1D( dim3 elementCoordinates, dim3 volumeDimensions )
{
    return __umul24( elementCoordinates.z, __umul24( volumeDimensions.y, volumeDimensions.x ) ) +
           __umul24( elementCoordinates.y, volumeDimensions.x ) +
           elementCoordinates.x;
}

__device__ dim3 ComputeIndex1DTo3D( int elementCoordinate, dim3 volumeDimensions, float invVolumeDimensionsX, float invVolumeDimensionsXY )
{
    int strideXY      = __umul24( volumeDimensions.x, volumeDimensions.y );
    int strideX       = volumeDimensions.x;

    int z             = __float2uint_rd( elementCoordinate * invVolumeDimensionsXY );
    elementCoordinate = elementCoordinate - __umul24( z, strideXY );

    int y             = __float2uint_rd( elementCoordinate * invVolumeDimensionsX );
    int x             = elementCoordinate - __umul24( y, strideX );

    dim3 elementCoordinates( x, y, z );

    return elementCoordinates;
}

__device__ unsigned int ComputeIndex3DToTiled1D( dim3 elementCoordinates, dim3 volumeDimensions )
{
    dim3 tileCoordinates;

    tileCoordinates.x = __float2uint_rd( elementCoordinates.x * INV_TILE_SIZE );
    tileCoordinates.y = __float2uint_rd( elementCoordinates.y * INV_TILE_SIZE );
    tileCoordinates.z = __float2uint_rd( elementCoordinates.z * INV_TILE_SIZE );

    dim3 tileVolumeDimensions;

    tileVolumeDimensions.x = __float2uint_rd( volumeDimensions.x * INV_TILE_SIZE );
    tileVolumeDimensions.y = __float2uint_rd( volumeDimensions.y * INV_TILE_SIZE );
    tileVolumeDimensions.z = __float2uint_rd( volumeDimensions.z * INV_TILE_SIZE );

    unsigned int tileIndex = ComputeIndex3DTo1D( tileCoordinates, tileVolumeDimensions );

    dim3 singleTileDimensions;

    singleTileDimensions.x = TILE_SIZE;
    singleTileDimensions.y = TILE_SIZE;
    singleTileDimensions.z = TILE_SIZE;

    dim3 coordinatesWithinTile;

    coordinatesWithinTile.x = elementCoordinates.x - __umul24( tileCoordinates.x, singleTileDimensions.x );
    coordinatesWithinTile.y = elementCoordinates.y - __umul24( tileCoordinates.y, singleTileDimensions.y );
    coordinatesWithinTile.z = elementCoordinates.z - __umul24( tileCoordinates.z, singleTileDimensions.z );

    unsigned int indexWithinTile = ComputeIndex3DTo1D( coordinatesWithinTile, singleTileDimensions );

    return tileIndex * TILE_NUM_ELEMENTS + indexWithinTile;
}

__device__ dim3 ComputeIndexTiled1DTo3D( int elementCoordinate, dim3 volumeDimensions, float invVolumeDimensionsX, float invVolumeDimensionsXY )
{
    int tileCoordinate1D = __float2uint_rd( elementCoordinate * INV_TILE_NUM_ELEMENTS );

    dim3 tileVolumeDimensions;

    tileVolumeDimensions.x = __float2uint_rd( volumeDimensions.x * INV_TILE_SIZE );
    tileVolumeDimensions.y = __float2uint_rd( volumeDimensions.y * INV_TILE_SIZE );
    tileVolumeDimensions.z = __float2uint_rd( volumeDimensions.z * INV_TILE_SIZE );

    float invTileVolumeDimensionsX  = invVolumeDimensionsX  * TILE_SIZE_FLOAT;
    float invTileVolumeDimensionsXY = invVolumeDimensionsXY * TILE_SIZE_FLOAT * TILE_SIZE_FLOAT;

    dim3 tileCoordinates3D  = ComputeIndex1DTo3D( tileCoordinate1D, tileVolumeDimensions, invTileVolumeDimensionsX, invTileVolumeDimensionsXY );

    int coordinatesWithinTile1D = elementCoordinate - __umul24( tileCoordinate1D, TILE_NUM_ELEMENTS );

    dim3 singleTileDimensions;

    singleTileDimensions.x = TILE_SIZE;
    singleTileDimensions.y = TILE_SIZE;
    singleTileDimensions.z = TILE_SIZE;

    dim3 coordinatesWithinTile3D = ComputeIndex1DTo3D( coordinatesWithinTile1D, singleTileDimensions, INV_TILE_SIZE, INV_TILE_SIZE * INV_TILE_SIZE );

    dim3 coordinates3D;

    coordinates3D.x = tileCoordinates3D.x * TILE_SIZE + coordinatesWithinTile3D.x;
    coordinates3D.y = tileCoordinates3D.y * TILE_SIZE + coordinatesWithinTile3D.y;
    coordinates3D.z = tileCoordinates3D.z * TILE_SIZE + coordinatesWithinTile3D.z;

    return coordinates3D;
}

__device__ dim3 OffsetCoordinates( dim3 unpackedCoordinates, int offsetX, int offsetY, int offsetZ, dim3 volumeDimensions )
{
    int3 newCoordinates;

    newCoordinates.x = unpackedCoordinates.x + offsetX;
    newCoordinates.y = unpackedCoordinates.y + offsetY;
    newCoordinates.z = unpackedCoordinates.z + offsetZ;

    if ( newCoordinates . x < 0 ||                                                   
         newCoordinates . y < 0 ||                                                   
         newCoordinates . z < 0 ||                                                   
         newCoordinates . x >= volumeDimensions . x ||                               
         newCoordinates . y >= volumeDimensions . y ||                               
         newCoordinates . z >= volumeDimensions . z )                                
    {                                                                                    
        newCoordinates . x = unpackedCoordinates . x;                                 
        newCoordinates . y = unpackedCoordinates . y;                                 
        newCoordinates . z = unpackedCoordinates . z;                                 
    }                                                                                    

    dim3 newCoordinatesDim3 = dim3( newCoordinates . x, newCoordinates . y, newCoordinates . z );

    return newCoordinatesDim3;
}

__device__ dim3 OffsetCoordinatesUnsafe( dim3 unpackedCoordinates, int offsetX, int offsetY, int offsetZ, dim3 volumeDimensions )
{
    int3 newCoordinates;

    newCoordinates.x = unpackedCoordinates.x + offsetX;
    newCoordinates.y = unpackedCoordinates.y + offsetY;
    newCoordinates.z = unpackedCoordinates.z + offsetZ;

#ifndef SKIP_BOUNDARY_CHECK_COORDINATES

    if ( newCoordinates . x < 0 ||                                                   
         newCoordinates . y < 0 ||                                                   
         newCoordinates . z < 0 ||                                                   
         newCoordinates . x >= volumeDimensions . x ||                               
         newCoordinates . y >= volumeDimensions . y ||                               
         newCoordinates . z >= volumeDimensions . z )                                
    {                                                                                    
        newCoordinates . x = unpackedCoordinates . x;                                 
        newCoordinates . y = unpackedCoordinates . y;                                 
        newCoordinates . z = unpackedCoordinates . z;                                 
    }                                                                                    

#endif

    dim3 newCoordinatesDim3 = dim3( newCoordinates . x, newCoordinates . y, newCoordinates . z );

    return newCoordinatesDim3;
}

__device__ int PackCoordinates( dim3 unpackedCoordinates, int offsetX, int offsetY, int offsetZ, dim3 volumeDimensions )
{
    int packedCoordinates = 0x0;

    unpackedCoordinates = OffsetCoordinates( unpackedCoordinates, offsetX, offsetY, offsetZ, volumeDimensions );

    packedCoordinates = packedCoordinates | ( ( unpackedCoordinates.x ) << 00 );
    packedCoordinates = packedCoordinates | ( ( unpackedCoordinates.y ) << 10 );
    packedCoordinates = packedCoordinates | ( ( unpackedCoordinates.z ) << 20 );

    return packedCoordinates;
}

__device__ dim3 UnpackCoordinates( int packedCoordinates )
{
    int x = ( packedCoordinates & ( 0x3FF << 00 ) ) >> 00;
    int y = ( packedCoordinates & ( 0x3FF << 10 ) ) >> 10;
    int z = ( packedCoordinates & ( 0x3FF << 20 ) ) >> 20;

    dim3 elementCoordinates( x, y, z );

    return elementCoordinates;
}

#define NUM_DIMENSIONS             3
#define NUM_DIMENSIONS_MASK        0x7
#define MAX_NUM_BITS_PER_DIMENSION 10
#define MAX_NUM_BITS               32

__device__ unsigned int GetBit( unsigned int value, unsigned int bit )
{
    unsigned int mask          = 0x1 << bit;
    unsigned int bitOfInterest = value & mask;

    return bitOfInterest >> bit;
}

__device__ unsigned int GetParity( unsigned int value )
{
    unsigned int numSetBits = 0;

    #pragma unroll
    for ( int i = 0; i < MAX_NUM_BITS; i++ )
    {
        numSetBits += GetBit( value, i );
    }

    return numSetBits;
}

__device__ unsigned int RotateRight( unsigned int value, unsigned int rotateAmount )
{
    return ( ( value >> rotateAmount ) | ( value << ( NUM_DIMENSIONS - rotateAmount ) ) ) & NUM_DIMENSIONS_MASK;
}

__device__ unsigned int RotateLeft( unsigned int value, unsigned int rotateAmount )
{
    return ( ( value << rotateAmount ) | ( value >> ( NUM_DIMENSIONS - rotateAmount ) ) ) & NUM_DIMENSIONS_MASK;
}

__device__ unsigned int ComputeOnes( unsigned int numOnes )
{
    unsigned int ones = ( 0x1 << ( numOnes ) ) - 1;
    return ones;
}

__device__ unsigned int ComputeNumBitsRequired( unsigned int value )
{
    unsigned int i = value;
    int c = 0;

    if ( i == 0 ) return 0;

    if ( i & (ComputeOnes(16) << 16) ) { i >>= 16; c ^= 16; }
    if ( i & (ComputeOnes( 8) <<  8) ) { i >>=  8; c ^=  8; }
    if ( i & (ComputeOnes( 4) <<  4) ) { i >>=  4; c ^=  4; }
    if ( i & (ComputeOnes( 2) <<  2) ) { i >>=  2; c ^=  2; }
    if ( i & (ComputeOnes( 1) <<  1) ) { i >>=  1; c ^=  1; }

    c++;

    return c;
}

__device__ unsigned int ComputeMask( dim3 numBitsRequired, unsigned int iteration, unsigned int direction )
{
    unsigned int mask = 0;

    if ( numBitsRequired.z > iteration ) mask = mask | 0x4;
    if ( numBitsRequired.y > iteration ) mask = mask | 0x2;
    if ( numBitsRequired.x > iteration ) mask = mask | 0x1;

    return mask;
}

__device__ unsigned int ComputeTransform( unsigned int label, unsigned int entryPoint, unsigned int direction )
{
    return RotateRight( label ^ entryPoint, direction + 1 );
}

__device__ unsigned int ComputeGrayCode( unsigned int value )
{
    return value ^ ( value >> 1 );
}

__device__ unsigned int ComputeInverseGrayCode( unsigned int value )
{
    unsigned int m = ComputeNumBitsRequired( value );

    unsigned int i = value;
    unsigned int j = 1;

    if ( j < m ) 
    {
        i = i ^ ( value >> j );
        j++;
    }

    if ( j < m ) 
    {
        i = i ^ ( value >> j );
        j++;
    }

    if ( j < m ) 
    {
        i = i ^ ( value >> j );
        j++;
    }

    return i;
}

__device__ unsigned int ComputeGrayCodeRank( unsigned int mask, unsigned int pattern, unsigned int inverseGrayCode )
{
    unsigned int r = 0;

    #pragma unroll
    for ( int k = NUM_DIMENSIONS - 1; k >= 0; k-- )
    {
        if ( GetBit( mask, k ) == 1 )
        {
            r = ( r << 1 ) | GetBit( inverseGrayCode, k );
        }
    }

    return r;
}

__device__ unsigned int ComputeEntryPoint( int inverseGrayCode )
{
    if ( inverseGrayCode == 0 )
    {
        return 0;
    }
    else
    {
        return ComputeGrayCode( 2 * ( ( inverseGrayCode - 1 ) / 2 ) );
    }
}

__device__ unsigned int ComputeTrailingSetBits( int value )
{
    unsigned int i = value;
    int c = 0;

    if ( i == ComputeOnes( 32 ) ) return 32;

    if ( (i&ComputeOnes(16)) == ComputeOnes(16) ) { i>>=16; c^=16; }
    if ( (i&ComputeOnes( 8)) == ComputeOnes( 8) ) { i>>= 8; c^= 8; }
    if ( (i&ComputeOnes( 4)) == ComputeOnes( 4) ) { i>>= 4; c^= 4; }
    if ( (i&ComputeOnes( 2)) == ComputeOnes( 2) ) { i>>= 2; c^= 2; }
    if ( (i&ComputeOnes( 1)) == ComputeOnes( 1) ) { i>>= 1; c^= 1; }

    return c;
}

__device__ unsigned int ComputeDirection( int inverseGrayCode )
{
    if ( inverseGrayCode == 0 )     return 0;
    if ( inverseGrayCode % 2 == 0 ) return ComputeTrailingSetBits( inverseGrayCode - 1 ) % NUM_DIMENSIONS;
    if ( inverseGrayCode % 2 == 1 ) return ComputeTrailingSetBits( inverseGrayCode )     % NUM_DIMENSIONS;

    return 0xdeadbeef;
}

__device__ unsigned int ComputeIndex3DToHilbert1D( dim3 elementCoordinates, dim3 volumeDimensions )
{
    // assume 8 bits required per dimension for now
    dim3 numBitsRequired( 8, 8, 8 );

    // step 1
    unsigned int compactHilbertIndex = 0;
    unsigned int entryPoint          = 0;
    unsigned int direction           = 0;

    // step 2
    unsigned int numBitsMaxDimension = 10;

    // step 3
    for ( int i = numBitsMaxDimension - 1; i >= 0; i-- )
    {
        // step 4
        unsigned int mask = ComputeMask( numBitsRequired, i, direction );

        // step 5
        mask = RotateRight( mask, direction + 1 );

        // step 6
        unsigned int pattern = RotateRight( entryPoint, direction + 1 ) & ( !mask );

        // step 7
        unsigned int label =
            ( GetBit( elementCoordinates.z, i ) << 2 ) |
            ( GetBit( elementCoordinates.y, i ) << 1 ) |
            ( GetBit( elementCoordinates.x, i ) << 0 );

        // step 8
        label = ComputeTransform( label, entryPoint, direction );

        // step 9
        unsigned int inverseGrayCode = ComputeInverseGrayCode( label );

        // step 10
        unsigned int grayCodeRank = ComputeGrayCodeRank( mask, pattern, inverseGrayCode );

        // step 11
        entryPoint = entryPoint ^ ( RotateLeft( ComputeEntryPoint( inverseGrayCode ), direction + 1 ) );

        // step 12
        direction = ( direction + ComputeDirection( inverseGrayCode ) + 1 ) % NUM_DIMENSIONS;

        // step 13
        compactHilbertIndex = ( compactHilbertIndex << GetParity( mask ) ) | grayCodeRank;
    }

    return compactHilbertIndex;
}

#endif