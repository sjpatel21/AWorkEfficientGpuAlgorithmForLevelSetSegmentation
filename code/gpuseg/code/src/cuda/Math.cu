#ifndef SRC_CUDA_MATH_CU
#define SRC_CUDA_MATH_CU

__device__ float ComputeSmoothStep( float distance, float maxDistance )
{
    if ( distance < maxDistance )
    {
        return maxDistance;
    }
    else
    {
        return 0;
    }
}

__device__ float Sqr( float f )
{
    return f * f;
}

__device__ int Clamp( int i, int min, int max )
{
    if ( i < min )
    {
        return min;
    }

    if ( i > max )
    {
        return max;
    }

    return i;
}

__device__ float Clamp( float i, float min, float max )
{
    if ( i < min )
    {
        return min;
    }

    if ( i > max )
    {
        return max;
    }

    return i;
}

__device__ unsigned char ConvertFloatToUChar( float f )
{
    return Clamp ( (int)( f * 255 ), 0, 255 );
}

__device__ float Abs( float f )
{
    if ( f > 0 )
    {
        return f;
    }
    else
    {
        return -1.0f * f;
    }
}

__device__ bool Equals( float left, float right, float epsilon )
{
    return ( Abs( left - right ) <= epsilon );
}

#endif