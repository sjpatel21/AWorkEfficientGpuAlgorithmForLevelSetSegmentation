#ifndef LEFOHN_COORDINATES_HPP
#define LEFOHN_COORDINATES_HPP

#include "math/Constants.hpp"

/***
    A VirtualCoordinate is (x,y,z) where z is the depth in the virtual memory volume.

    A PhysicalCoordinate is (x,y)

    Both represent page addresses.
***/

namespace lefohn
{
    class PhysicalCoordinate
    {
    public:
        PhysicalCoordinate();
        PhysicalCoordinate(int x, int y);
        PhysicalCoordinate( const PhysicalCoordinate& other);

        ~PhysicalCoordinate(){};

        // Accessor operators
        int&       operator [] ( int i );
        const int& operator [] ( int i ) const;

        const int* Ref() const; // get pointer to the array

        // Comparison operators
        bool operator == (const PhysicalCoordinate& rhs) const;
        bool operator != (const PhysicalCoordinate& rhs) const;

        // Assignment operators
        PhysicalCoordinate& operator = (const PhysicalCoordinate& rhs);
        
        /* Don't think we or want these.
        PhysicalCoordinate& operator+=(const PhysicalCoordinate& rhs);
        PhysicalCoordinate& operator-=(const PhysicalCoordinate& rhs);
        */
        
        // Arithmetic operators
        /* Don't think we or want these.
        const PhysicalCoordinate& operator+(const PhysicalCoordinate& rhs);
        const PhysicalCoordinate& operator-(const PhysicalCoordinate& rhs);

        
        */    
        
        // Increments the coordinate bounding the first dimension to rowSize.
        // When the first dimension hits rowSize the second dimension is incremented.
        void Increment(int rowSize);

        static const PhysicalCoordinate INVALID_COORDINATE;

    private:
        int mElements[2];

        void CheckRange( int index, int min, int max, const char* ) const;
    };

    class VirtualCoordinate
    {
    public:
        VirtualCoordinate();
        VirtualCoordinate(int x, int y, int z);
        VirtualCoordinate( const VirtualCoordinate& other);

        ~VirtualCoordinate(){};

        // Accessor operators
        int&       operator [] ( int i );
        const int& operator [] ( int i ) const;

        const int* Ref() const; // get pointer to the array

        // Comparison operators
        bool operator == (const VirtualCoordinate& rhs) const;
        bool operator != (const VirtualCoordinate& rhs) const;

        // Assignment operators
        VirtualCoordinate& operator = (const VirtualCoordinate& rhs);

        /* Don't think we or want these.
        VirtualCoordinate& operator+=(const VirtualCoordinate& rhs);
        VirtualCoordinate& operator-=(const VirtualCoordinate& rhs);
        */

        // Arithmetic operators
        /* Don't think we or want these.
        const VirtualCoordinate& operator+(const VirtualCoordinate& rhs);
        const VirtualCoordinate& operator-(const VirtualCoordinate& rhs);


        */    

        // Increments the coordinate bounding each of the first two dimensions
        // between rowSize and columnSize, respectively.
        // When the first dimension hits rowSize the second dimension is incremented.
        // When the second dimension hits columnSize the third dimension is incremented.
        void Increment(int rowSize, int columnSize);

        static const VirtualCoordinate INVALID_COORDINATE;

    private:
        int mElements[3];

        void CheckRange( int index, int min, int max, const char* ) const;
    };
}

#endif