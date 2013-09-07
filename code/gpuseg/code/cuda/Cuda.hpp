#ifndef CUDA_CUDA_HPP
#define CUDA_CUDA_HPP

#if defined PLATFORM_WIN32
#define NOMINMAX
#include <windows.h>
#endif

#include <builtin_types.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <cutil_inline.h>
#include <cutil.h>

#include <cudpp.h>

#include "cuda/CudaTypes.hpp"

namespace rendering
{
namespace rtgi
{
class VertexBuffer;
class PixelBuffer;
class BufferTexture;
class Texture;
}
}

//
// initialize and terminate
//
void CudaInitialize();
void CudaTerminate();

//
// debug
//
void CudaCheckErrors();
void CudaCheckErrorsRelease();
void CudaGetMemoryInfo( unsigned int* freeMemory, unsigned int* totalMemory );

//
// synchronize
//
void CudaSynchronize();

//
// warp alignment
//
void         CudaGetWarpAlignment( unsigned int* warpAlignment );
unsigned int CudaGetWarpAlignedValue( unsigned int valueUnaligned );

//
// graphics interop
//
void CudaRtgiRegisterBuffer( const rendering::rtgi::VertexBuffer*  vertexBuffer );
void CudaRtgiRegisterBuffer( const rendering::rtgi::PixelBuffer*   pixelBuffer );
void CudaRtgiRegisterBuffer( const rendering::rtgi::BufferTexture* bufferTexture );

#ifdef CUDA_30
void CudaRtgiRegisterTexture( const rendering::rtgi::Texture* texture );
#endif

void CudaRtgiUnregisterBuffer( const rendering::rtgi::VertexBuffer*  vertexBuffer );
void CudaRtgiUnregisterBuffer( const rendering::rtgi::PixelBuffer*   pixelBuffer );
void CudaRtgiUnregisterBuffer( const rendering::rtgi::BufferTexture* bufferTexture );

#ifdef CUDA_30
void CudaRtgiUnregisterTexture( const rendering::rtgi::Texture* texture );
#endif

template< typename T > void CudaRtgiMapBuffer( T** cudaBuffer, const rendering::rtgi::VertexBuffer*  rtgiVertexBuffer );
template< typename T > void CudaRtgiMapBuffer( T** cudaBuffer, const rendering::rtgi::PixelBuffer*   rtgiPixelBuffer );
template< typename T > void CudaRtgiMapBuffer( T** cudaBuffer, const rendering::rtgi::BufferTexture* rtgiBufferTexture );

#ifdef CUDA_30
void CudaRtgiMapTexture( cudaArray** cudaArrayPtr, const rendering::rtgi::Texture* rtgiTexture );
#endif

void CudaRtgiUnmapBuffer( const rendering::rtgi::VertexBuffer*  rtgiVertexBuffer );
void CudaRtgiUnmapBuffer( const rendering::rtgi::PixelBuffer*   rtgiPixelBuffer );
void CudaRtgiUnmapBuffer( const rendering::rtgi::BufferTexture* rtgiBufferTexture );

#ifdef CUDA_30
void CudaRtgiUnmapTexture( const rendering::rtgi::Texture* rtgiTexture );
#endif

GLuint CudaGetOpenGLBufferID( const rendering::rtgi::VertexBuffer*  rtgiVertexBuffer );
GLuint CudaGetOpenGLBufferID( const rendering::rtgi::PixelBuffer*   rtgiPixelBuffer );
GLuint CudaGetOpenGLBufferID( const rendering::rtgi::BufferTexture* rtgiBufferTexture );


//
// memory management
//
template< typename T > void CudaAllocateDeviceMemory( unsigned int hostDataNumBytes, T** deviceData );
template< typename T > void CudaDeallocateDeviceMemory( T** deviceData );

template< typename T > void CudaAllocatePageLockedMemoryHost( unsigned int hostDataNumBytes, T** hostData );
template< typename T > void CudaDeallocatePageLockedMemoryHost( T** hostData );

void CudaAllocateDeviceArray3D  ( dim3 volumeDimensions, cudaArray** deviceArray3D, unsigned int elementSizeInBytes, bool isSigned );
void CudaDeallocateDeviceArray3D( cudaArray** deviceData );

void CudaMemCopyHostToGlobal        ( const void* sourceDataHost, void* destinationDataGlobal, unsigned int numBytes );
void CudaMemCopyHostToArray3D       ( void*       sourceDataHost, cudaArray* destinationArray, dim3 volumeDimensions, unsigned int elementSizeInBytes );
void CudaMemCopyGlobalToHost        ( void* destinationDataHost,   const void* sourceDataGlobal, unsigned int numBytes );
void CudaMemCopyGlobalToGlobal      ( void* destinationDataGlobal, const void* sourceDataGlobal, unsigned int numBytes );
void CudaMemCopyGlobalToArray3D     ( void* sourceDataGlobal, cudaArray* destinationArray, dim3 volumeDimensions, unsigned int elementSizeInBytes );
void CudaMemCopyGlobalToArray3DAsync( void* sourceDataGlobal, cudaArray* destinationArray, dim3 volumeDimensions, unsigned int elementSizeInBytes );
void CudaMemCopyArray3DToHost       ( void* destinationDataHost, cudaArray* sourceArray, dim3 volumeDimensions, unsigned int elementSizeInBytes );

void CudaMemSet                     ( void* deviceData, int valueToSet, int numBytes );
void CudaMemSetAsync                ( void* deviceData, int valueToSet, int numBytes );

//
// texture management
//
template< typename T >
void              CudaBindTextureToBuffer( const char* textureName, const T* buffer );
void              CudaBindTextureToArray ( const char* textureName, const cudaArray* deviceArray );
void              CudaUnbindTexture      ( const char* textureName );
textureReference* CudaGetTextureReference( const char* textureName );

//
// cudpp operations
//
void CudppPlan       ( CUDPPHandle*              planHandle, 
                       CUDPPConfiguration        config, 
                       size_t                    n, 
                       size_t                    rows, 
                       size_t                    rowPitch );
                                          
void CudppDestroyPlan( CUDPPHandle               planHandle );
                                          
void CudppSort       ( CUDPPHandle               planHandle,
                       CudaCompactElement*       sourceDevice,
                       int                       numElements );

void CudppScan       ( CUDPPHandle               planHandle,
                       const CudaCompactElement* sourceDevice,
                       CudaCompactElement*       destinationDevice,
                       int                       numElements );

void CudppCompact    ( CUDPPHandle               planHandle,
                       const CudaCompactElement* sourceDevice,
                       const CudaCompactElement* validDevice,
                       CudaCompactElement*       destinationDevice,
                       size_t*                   numValidElementsDevice,
                       int                       numElements,
                       size_t*                   numValidElementsHost );

//
// private
//
void CudaRtgiMapBufferPrivate( void** buffer, const rendering::rtgi::VertexBuffer*  rtgiVertexBuffer );
void CudaRtgiMapBufferPrivate( void** buffer, const rendering::rtgi::PixelBuffer*   rtgiPixelBuffer );
void CudaRtgiMapBufferPrivate( void** buffer, const rendering::rtgi::BufferTexture* rtgiBufferTexture );

//
// template implementations
//
template< typename T >
void CudaRtgiMapBuffer( T** cudaBuffer, const rendering::rtgi::VertexBuffer* rtgiVertexBuffer )
{
    CudaRtgiMapBufferPrivate( (void**)cudaBuffer, rtgiVertexBuffer );
}

template< typename T >
void CudaRtgiMapBuffer( T** cudaBuffer, const rendering::rtgi::PixelBuffer* rtgiPixelBuffer )
{
    CudaRtgiMapBufferPrivate( (void**)cudaBuffer, rtgiPixelBuffer );
}

template< typename T >
void CudaRtgiMapBuffer( T** cudaBuffer, const rendering::rtgi::BufferTexture* rtgiBufferTexture )
{
    CudaRtgiMapBufferPrivate( (void**)cudaBuffer, rtgiBufferTexture );
}

#pragma warning( push )
#pragma warning( disable:4700 )

template< typename TActual, typename TInterpretedAs >
void CudaBindTextureToBuffer( const char* textureName, const TActual* buffer )
{
    textureReference* textureReferencePointer = CudaGetTextureReference( textureName );

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc< TInterpretedAs >();

    textureReferencePointer->normalized     = false;
    textureReferencePointer->filterMode     = cudaFilterModePoint;   
    textureReferencePointer->addressMode[0] = cudaAddressModeClamp;
    textureReferencePointer->addressMode[1] = cudaAddressModeClamp;
    textureReferencePointer->addressMode[2] = cudaAddressModeClamp;

    // bind buffer to 1D texture
    CUDA_SAFE_CALL( cudaBindTexture( 0, textureReferencePointer, buffer, &channelDesc ) );
}

template< typename T >
void CudaBindTextureToBuffer( const char* textureName, const T* buffer )
{
    CudaBindTextureToBuffer< T, T >( textureName, buffer );
}

#pragma warning( pop )

template< typename T >
void CudaAllocatePageLockedMemoryHost( unsigned int hostDataNumBytes, T** hostData )
{
    CUDA_SAFE_CALL( cudaMallocHost( hostData, hostDataNumBytes ) );

    CudaSynchronize();
}

template< typename T >
void CudaAllocateDeviceMemory( unsigned int hostDataNumBytes, T** deviceData )
{
    // allocate device memory
    CUDA_SAFE_CALL( cudaMalloc( deviceData, hostDataNumBytes ) );

    CudaSynchronize();
}

template< typename T >
void CudaDeallocateDeviceMemory( T** deviceData )
{
    // free device memory
    CUDA_SAFE_CALL( cudaFree( *deviceData ) );

    CudaSynchronize();

    *deviceData = NULL;
}

template< typename T >
void CudaDeallocatePageLockedMemoryHost( T** hostData )
{
    // free device memory
    CUDA_SAFE_CALL( cudaFreeHost( *hostData ) );

    CudaSynchronize();

    *hostData = NULL;
}

#endif