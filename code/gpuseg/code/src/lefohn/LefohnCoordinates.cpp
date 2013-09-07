#include "lefohn/LefohnCoordinates.hpp"

#include "core/Assert.hpp"


namespace lefohn
{
    const PhysicalCoordinate PhysicalCoordinate::INVALID_COORDINATE = PhysicalCoordinate(-1, -1);
    const VirtualCoordinate VirtualCoordinate::INVALID_COORDINATE = VirtualCoordinate(-1, -1, -1);

    /***
    PhysicalCoordinate
    ***/

    PhysicalCoordinate::PhysicalCoordinate()
    {
        mElements[math::X] = 0;
        mElements[math::Y] = 0;
    }

    PhysicalCoordinate::PhysicalCoordinate(int x, int y)
    {
        mElements[math::X] = x;
        mElements[math::Y] = y;
    }

    PhysicalCoordinate::PhysicalCoordinate(const PhysicalCoordinate& other)
    {
        this->mElements[math::X] = other.mElements[math::X];
        this->mElements[math::Y] = other.mElements[math::Y];
    }

    int& PhysicalCoordinate::operator [] ( int i )
    {
        CheckRange( i, 0, 2, "(PhysicalCoordinate::[i]) index out of range" );
        return( mElements[ i ] );
    }


    const int& PhysicalCoordinate::operator [] ( int i ) const
    {
        CheckRange( i, 0, 2, "(PhysicalCoordinate::[i]) index out of range" );
        return( mElements[ i ] );
    }

    const int* PhysicalCoordinate::Ref() const
    {
        return reinterpret_cast< const int* >( mElements );
    }

    PhysicalCoordinate& PhysicalCoordinate::operator = ( const PhysicalCoordinate &rhs )
    {
        mElements[ math::X ] = rhs[ math::X ];
        mElements[ math::Y ] = rhs[ math::Y ];

        return *this;
    }

    bool PhysicalCoordinate::operator == ( const PhysicalCoordinate &rhs ) const
    {
        return ( mElements[ math::X ] == rhs[ math::X ] &&
                 mElements[ math::Y ] == rhs[ math::Y ] );
    }


    bool PhysicalCoordinate::operator != ( const PhysicalCoordinate &rhs ) const
    {
        return ( mElements[ math::X ] != rhs[ math::X ] ||
                 mElements[ math::Y ] != rhs[ math::Y ] );
    }

    void PhysicalCoordinate::Increment(int rowSize)
    {
        ++mElements[math::X];
        if( mElements[math::X] == rowSize )
        {
            mElements[math::X] = 0;
            ++mElements[math::Y];
        }
    }

    void PhysicalCoordinate::CheckRange( int index, int min, int max, const char* ) const
    {
        Assert( index < max && index >= min );
    }

    /***
    VirtualCoordinate
    ***/

    VirtualCoordinate::VirtualCoordinate()
    {
        mElements[math::X] = 0;
        mElements[math::Y] = 0;
        mElements[math::Z] = 0;
    }

    VirtualCoordinate::VirtualCoordinate(int x, int y, int z)
    {
        mElements[math::X] = x;
        mElements[math::Y] = y;
        mElements[math::Z] = z;
    }

    VirtualCoordinate::VirtualCoordinate(const VirtualCoordinate& other)
    {
        this->mElements[math::X] = other.mElements[math::X];
        this->mElements[math::Y] = other.mElements[math::Y];
        this->mElements[math::Z] = other.mElements[math::Z];
    }

    int& VirtualCoordinate::operator [] ( int i )
    {
        CheckRange( i, 0, 3, "(VirtualCoordinate::[i]) index out of range" );
        return( mElements[ i ] );
    }


    const int& VirtualCoordinate::operator [] ( int i ) const
    {
        CheckRange( i, 0, 3, "(VirtualCoordinate::[i]) index out of range" );
        return( mElements[ i ] );
    }

    const int* VirtualCoordinate::Ref() const
    {
        return reinterpret_cast< const int* >( mElements );
    }

    VirtualCoordinate& VirtualCoordinate::operator = ( const VirtualCoordinate &rhs )
    {
        mElements[ math::X ] = rhs[ math::X ];
        mElements[ math::Y ] = rhs[ math::Y ];
        mElements[ math::Z ] = rhs[ math::Z ];

        return *this;
    }

    bool VirtualCoordinate::operator == ( const VirtualCoordinate &rhs ) const
    {
        return ( mElements[ math::X ] == rhs[ math::X ] &&
                 mElements[ math::Y ] == rhs[ math::Y ] && 
                 mElements[ math::Z ] == rhs[ math::Z ] );
    }


    bool VirtualCoordinate::operator != ( const VirtualCoordinate &rhs ) const
    {
        return ( mElements[ math::X ] != rhs[ math::X ] ||
                 mElements[ math::Y ] != rhs[ math::Y ] ||
                 mElements[ math::Z ] != rhs[ math::Z ] );
    }

    void VirtualCoordinate::Increment(int rowSize, int columnSize)
    {
        ++mElements[math::X];
        if( mElements[math::X] == rowSize )
        {
            mElements[math::X] = 0;
            ++mElements[math::Y];
            if( mElements[math::Y] == columnSize  )
            {
                mElements[math::Y] = 0;
                ++mElements[math::Z];
            }
        }
    }

    void VirtualCoordinate::CheckRange( int index, int min, int max, const char* ) const
    {
        Assert( index < max && index >= min );
    }
}