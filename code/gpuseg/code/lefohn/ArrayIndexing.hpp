#ifndef LEFOHN_ARRAY_INDEXING_HPP
#define LEFOHN_ARRAY_INDEXING_HPP

namespace lefohn
{

inline int Get1DIndexFrom3DIndex(int depthIndex, int columnIndex, int rowIndex, int columnSize, int rowSize)
{
    // ((depthindex*col_size+colindex) * row_size + rowindex)
    return (depthIndex * columnSize + columnIndex)  * rowSize + rowIndex;
}

inline int Get1DIndexFrom2DIndex(int columnIndex, int rowIndex, int rowSize)
{
    return (columnIndex*rowSize) + rowIndex;
}

}

#endif