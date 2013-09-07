#include "cuda/Cuda.hpp"

#if defined PLATFORM_WIN32
#define NOMINMAX
#include <windows.h>
#endif

#include <cuda.h>
#include <cuda_runtime.h>

#include <cutil_inline.h>
#include <cutil.h>

#include <cudpp.h>

#include "core/Assert.hpp"
#include "core/Printf.hpp"

#include "container/HashMap.hpp"

#include "rendering/rtgi/RTGI.hpp"
#include "rendering/rtgi/PixelBuffer.hpp"
#include "rendering/rtgi/Texture.hpp"
#include "rendering/rtgi/opengl/OpenGLPixelBuffer.hpp"
#include "rendering/rtgi/opengl/OpenGLTexture.hpp"
#include "rendering/rtgi/opengl/OpenGLVertexBuffer.hpp"

#ifdef CUDA_30
static container::HashMap< const core::RefCounted*, cudaGraphicsResource* > sCudaGraphicsResources;
#endif

void CudaInitialize()
{
    static bool once = false;

    if ( !once )
    {
        CUDA_SAFE_CALL( cudaSetDevice( cutGetMaxGflopsDeviceId() ) );
        CUDA_SAFE_CALL( cudaGLSetGLDevice( cutGetMaxGflopsDeviceId() ) );

        once = true;
    }

    CUDA_SAFE_CALL( cudaFree( 0 ) );
}

void CudaTerminate()
{
}

void CudaCheckErrors()
{
    cudaError_t error       = cudaGetLastError();
    const char* errorString = cudaGetErrorString( error );

    Assert( error != cudaErrorMissingConfiguration     );   
    Assert( error != cudaErrorMemoryAllocation         );   
    Assert( error != cudaErrorInitializationError      );   
    Assert( error != cudaErrorLaunchFailure            );
    Assert( error != cudaErrorPriorLaunchFailure       );   
    Assert( error != cudaErrorLaunchTimeout            );   
    Assert( error != cudaErrorLaunchOutOfResources     );
    Assert( error != cudaErrorMissingConfiguration     );
    Assert( error != cudaErrorInitializationError      );
    Assert( error != cudaErrorLaunchFailure            );
    Assert( error != cudaErrorPriorLaunchFailure       );
    Assert( error != cudaErrorLaunchTimeout            );
    Assert( error != cudaErrorLaunchOutOfResources     );
    Assert( error != cudaErrorInvalidDeviceFunction    );
    Assert( error != cudaErrorInvalidConfiguration     );
    Assert( error != cudaErrorInvalidDevice            );
    Assert( error != cudaErrorInvalidValue             );
    Assert( error != cudaErrorInvalidPitchValue        );
    Assert( error != cudaErrorInvalidSymbol            );
    Assert( error != cudaErrorMapBufferObjectFailed    );
    Assert( error != cudaErrorUnmapBufferObjectFailed  );
    Assert( error != cudaErrorInvalidHostPointer       );
    Assert( error != cudaErrorInvalidDevicePointer     );
    Assert( error != cudaErrorInvalidTexture           );
    Assert( error != cudaErrorInvalidTextureBinding    );
    Assert( error != cudaErrorInvalidChannelDescriptor );
    Assert( error != cudaErrorInvalidMemcpyDirection   );
    Assert( error != cudaErrorAddressOfConstant        );
    Assert( error != cudaErrorTextureFetchFailed       );
    Assert( error != cudaErrorTextureNotBound          );
    Assert( error != cudaErrorSynchronizationError     );
    Assert( error != cudaErrorInvalidFilterSetting     );
    Assert( error != cudaErrorInvalidNormSetting       );
    Assert( error != cudaErrorMixedDeviceExecution     );
    Assert( error != cudaErrorCudartUnloading          );
    Assert( error != cudaErrorUnknown                  );
    Assert( error != cudaErrorNotYetImplemented        );
    Assert( error != cudaErrorMemoryValueTooLarge      );
    Assert( error != cudaErrorInvalidResourceHandle    );
    Assert( error != cudaErrorNotReady                 );
    Assert( error != cudaErrorInsufficientDriver       );
    Assert( error != cudaErrorSetOnActiveProcess       );
    Assert( error != cudaErrorNoDevice                 );
    Assert( error != cudaErrorStartupFailure           );
    Assert( error != cudaErrorApiFailureBase           );

    if ( error != cudaSuccess )
    {
#if defined(PLATFORM_WIN32)
        OutputDebugString( errorString );
#endif
    }

    Assert( error == cudaSuccess );
}

void CudaCheckErrorsRelease()
{
    cudaError_t error       = cudaGetLastError();
    const char* errorString = cudaGetErrorString( error );

    ReleaseAssert( error != cudaErrorMissingConfiguration     );   
    ReleaseAssert( error != cudaErrorMemoryAllocation         );   
    ReleaseAssert( error != cudaErrorInitializationError      );   
    ReleaseAssert( error != cudaErrorLaunchFailure            );
    ReleaseAssert( error != cudaErrorPriorLaunchFailure       );   
    ReleaseAssert( error != cudaErrorLaunchTimeout            );   
    ReleaseAssert( error != cudaErrorLaunchOutOfResources     );
    ReleaseAssert( error != cudaErrorMissingConfiguration     );
    ReleaseAssert( error != cudaErrorInitializationError      );
    ReleaseAssert( error != cudaErrorLaunchFailure            );
    ReleaseAssert( error != cudaErrorPriorLaunchFailure       );
    ReleaseAssert( error != cudaErrorLaunchTimeout            );
    ReleaseAssert( error != cudaErrorLaunchOutOfResources     );
    ReleaseAssert( error != cudaErrorInvalidDeviceFunction    );
    ReleaseAssert( error != cudaErrorInvalidConfiguration     );
    ReleaseAssert( error != cudaErrorInvalidDevice            );
    ReleaseAssert( error != cudaErrorInvalidValue             );
    ReleaseAssert( error != cudaErrorInvalidPitchValue        );
    ReleaseAssert( error != cudaErrorInvalidSymbol            );
    ReleaseAssert( error != cudaErrorMapBufferObjectFailed    );
    ReleaseAssert( error != cudaErrorUnmapBufferObjectFailed  );
    ReleaseAssert( error != cudaErrorInvalidHostPointer       );
    ReleaseAssert( error != cudaErrorInvalidDevicePointer     );
    ReleaseAssert( error != cudaErrorInvalidTexture           );
    ReleaseAssert( error != cudaErrorInvalidTextureBinding    );
    ReleaseAssert( error != cudaErrorInvalidChannelDescriptor );
    ReleaseAssert( error != cudaErrorInvalidMemcpyDirection   );
    ReleaseAssert( error != cudaErrorAddressOfConstant        );
    ReleaseAssert( error != cudaErrorTextureFetchFailed       );
    ReleaseAssert( error != cudaErrorTextureNotBound          );
    ReleaseAssert( error != cudaErrorSynchronizationError     );
    ReleaseAssert( error != cudaErrorInvalidFilterSetting     );
    ReleaseAssert( error != cudaErrorInvalidNormSetting       );
    ReleaseAssert( error != cudaErrorMixedDeviceExecution     );
    ReleaseAssert( error != cudaErrorCudartUnloading          );
    ReleaseAssert( error != cudaErrorUnknown                  );
    ReleaseAssert( error != cudaErrorNotYetImplemented        );
    ReleaseAssert( error != cudaErrorMemoryValueTooLarge      );
    ReleaseAssert( error != cudaErrorInvalidResourceHandle    );
    ReleaseAssert( error != cudaErrorNotReady                 );
    ReleaseAssert( error != cudaErrorInsufficientDriver       );
    ReleaseAssert( error != cudaErrorSetOnActiveProcess       );
    ReleaseAssert( error != cudaErrorNoDevice                 );
    ReleaseAssert( error != cudaErrorStartupFailure           );
    ReleaseAssert( error != cudaErrorApiFailureBase           );

    if ( error != cudaSuccess )
    {
#if defined(PLATFORM_WIN32)
        OutputDebugString( errorString );
#endif
    }

    ReleaseAssert( error == cudaSuccess );
}

void CudaSynchronize()
{
    cudaThreadSynchronize();
}

void CudaGetMemoryInfo( unsigned int* freeMemory, unsigned int* totalMemory )
{
    CUresult memInfoResult = cuMemGetInfo( freeMemory, totalMemory );
    Assert( memInfoResult == CUDA_SUCCESS );
}

void CudaGetWarpAlignment( unsigned int* warpAlignment )
{
    Assert( warpAlignment != NULL );

    *warpAlignment = 256;
}

unsigned int CudaGetWarpAlignedValue( unsigned int valueUnaligned )
{
    unsigned int warpAlignment;
    CudaGetWarpAlignment( &warpAlignment );

    unsigned int numPadElementsPerSection      = ( warpAlignment - ( valueUnaligned % warpAlignment ) ) % warpAlignment;
    unsigned int numElementsPerSectionAfterPad = valueUnaligned + numPadElementsPerSection;

    return numElementsPerSectionAfterPad;
}

#ifdef CUDA_30
void CudaRtgiRegisterBuffer( const rendering::rtgi::VertexBuffer* vertexBuffer )
{
    Assert( !sCudaGraphicsResources.Contains( vertexBuffer ) );

    const rendering::rtgi::OpenGLVertexBuffer* openGLVertexBuffer = dynamic_cast< const rendering::rtgi::OpenGLVertexBuffer* >( vertexBuffer );
    cudaGraphicsResource* resource;

    CUDA_SAFE_CALL( cudaGraphicsGLRegisterBuffer( &resource, openGLVertexBuffer->GetOpenGLVertexBufferID(), cudaGraphicsMapFlagsReadOnly ) );

    sCudaGraphicsResources.Insert( vertexBuffer, resource );
}

void CudaRtgiRegisterBuffer( const rendering::rtgi::PixelBuffer* pixelBuffer )
{
    Assert( !sCudaGraphicsResources.Contains( pixelBuffer ) );

    const rendering::rtgi::OpenGLPixelBuffer* openGLPixelBuffer = dynamic_cast< const rendering::rtgi::OpenGLPixelBuffer* >( pixelBuffer );
    cudaGraphicsResource* resource;

    CUDA_SAFE_CALL( cudaGraphicsGLRegisterBuffer( &resource, openGLPixelBuffer->GetOpenGLPixelBufferID(), cudaGraphicsMapFlagsNone ) );

    sCudaGraphicsResources.Insert( pixelBuffer, resource );
}

void CudaRtgiRegisterBuffer( const rendering::rtgi::BufferTexture* bufferTexture )
{
    Assert( !sCudaGraphicsResources.Contains( bufferTexture ) );

    const rendering::rtgi::OpenGLBufferTexture* openGLBufferTexture = dynamic_cast< const rendering::rtgi::OpenGLBufferTexture* >( bufferTexture );
    cudaGraphicsResource* resource;

    CUDA_SAFE_CALL( cudaGraphicsGLRegisterBuffer( &resource, openGLBufferTexture->GetOpenGLBufferID(), cudaGraphicsMapFlagsNone ) );

    sCudaGraphicsResources.Insert( bufferTexture, resource );
}

void CudaRtgiRegisterTexture( const rendering::rtgi::Texture* texture )
{
    Assert( !sCudaGraphicsResources.Contains( texture ) );

    const rendering::rtgi::OpenGLTexture* openGLTexture = dynamic_cast< const rendering::rtgi::OpenGLTexture* >( texture );
    cudaGraphicsResource* resource;

    // bind the texture but hard code the minification filter to nearest neighbor
    rendering::rtgi::TextureSamplerStateDesc textureSamplerStateDesc;

    textureSamplerStateDesc.textureSamplerInterpolationMode = rendering::rtgi::TextureSamplerInterpolationMode_Smooth;
    textureSamplerStateDesc.textureSamplerWrapMode          = rendering::rtgi::TextureSamplerWrapMode_ClampToEdge;

    texture->Bind( textureSamplerStateDesc );

    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );

    rendering::rtgi::CheckErrors();

    // register the texture
    CUDA_SAFE_CALL( cudaGraphicsGLRegisterImage( &resource, openGLTexture->GetOpenGLTextureID(), openGLTexture->GetOpenGLTextureTarget(), cudaGraphicsMapFlagsNone ) );

    // unbind the texture
    texture->Unbind();

    sCudaGraphicsResources.Insert( texture, resource );
}

void CudaRtgiUnregisterBuffer( const rendering::rtgi::VertexBuffer* vertexBuffer )
{
    Assert( sCudaGraphicsResources.Contains( vertexBuffer ) );
    CUDA_SAFE_CALL( cudaGraphicsUnregisterResource( sCudaGraphicsResources.Value( vertexBuffer ) ) );

    sCudaGraphicsResources.Remove( vertexBuffer );
}

void CudaRtgiUnregisterBuffer( const rendering::rtgi::PixelBuffer* pixelBuffer )
{
    Assert( sCudaGraphicsResources.Contains( pixelBuffer ) );
    CUDA_SAFE_CALL( cudaGraphicsUnregisterResource( sCudaGraphicsResources.Value( pixelBuffer ) ) );

    sCudaGraphicsResources.Remove( pixelBuffer );
}

void CudaRtgiUnregisterBuffer( const rendering::rtgi::BufferTexture* bufferTexture )
{
    Assert( sCudaGraphicsResources.Contains( bufferTexture ) );
    CUDA_SAFE_CALL( cudaGraphicsUnregisterResource( sCudaGraphicsResources.Value( bufferTexture ) ) );

    sCudaGraphicsResources.Remove( bufferTexture );
}

void CudaRtgiUnregisterTexture( const rendering::rtgi::Texture* texture )
{
    Assert( sCudaGraphicsResources.Contains( texture ) );
    CUDA_SAFE_CALL( cudaGraphicsUnregisterResource( sCudaGraphicsResources.Value( texture ) ) );

    sCudaGraphicsResources.Remove( texture );
}

void CudaRtgiMapTexture( cudaArray** cudaArrayPtr, const rendering::rtgi::Texture* rtgiTexture )
{
    Assert( sCudaGraphicsResources.Contains( rtgiTexture ) );

    cudaGraphicsResource* resources[ 1 ];
    resources[ 0 ] = sCudaGraphicsResources.Value( rtgiTexture );

    CUDA_SAFE_CALL( cudaGraphicsMapResources( 1, resources, 0 ) );
    CUDA_SAFE_CALL( cudaGraphicsSubResourceGetMappedArray( cudaArrayPtr, sCudaGraphicsResources.Value( rtgiTexture ), cudaGraphicsCubeFacePositiveX, 0 ) );
}

void CudaRtgiUnmapBuffer( const rendering::rtgi::VertexBuffer* rtgiVertexBuffer )
{
    Assert( sCudaGraphicsResources.Contains( rtgiVertexBuffer ) );
    cudaGraphicsResource* resources[ 1 ];

    resources[ 0 ] = sCudaGraphicsResources.Value( rtgiVertexBuffer );

    CUDA_SAFE_CALL( cudaGraphicsUnmapResources( 1, resources, 0 ) );
}

void CudaRtgiUnmapBuffer( const rendering::rtgi::PixelBuffer* rtgiPixelBuffer )
{
    Assert( sCudaGraphicsResources.Contains( rtgiPixelBuffer ) );
    cudaGraphicsResource* resources[ 1 ];

    resources[ 0 ] = sCudaGraphicsResources.Value( rtgiPixelBuffer );

    CUDA_SAFE_CALL( cudaGraphicsUnmapResources( 1, resources, 0 ) );
}

void CudaRtgiUnmapBuffer( const rendering::rtgi::BufferTexture* rtgiBufferTexture )
{
    Assert( sCudaGraphicsResources.Contains( rtgiBufferTexture ) );
    cudaGraphicsResource* resources[ 1 ];

    resources[ 0 ] = sCudaGraphicsResources.Value( rtgiBufferTexture );

    CUDA_SAFE_CALL( cudaGraphicsUnmapResources( 1, resources, 0 ) );
}

void CudaRtgiUnmapTexture( const rendering::rtgi::Texture* rtgiTexture )
{
    Assert( sCudaGraphicsResources.Contains( rtgiTexture ) );
    cudaGraphicsResource* resources[ 1 ];

    resources[ 0 ] = sCudaGraphicsResources.Value( rtgiTexture );

    CUDA_SAFE_CALL( cudaGraphicsUnmapResources( 1, resources, 0 ) );
}

void CudaRtgiMapBufferPrivate( void** buffer, const rendering::rtgi::VertexBuffer* rtgiVertexBuffer )
{
    Assert( sCudaGraphicsResources.Contains( rtgiVertexBuffer ) );

    cudaGraphicsResource* resources[ 1 ];
    size_t size = 0;

    resources[ 0 ] = sCudaGraphicsResources.Value( rtgiVertexBuffer );

    CUDA_SAFE_CALL( cudaGraphicsMapResources( 1, resources, 0 ) );
    CUDA_SAFE_CALL( cudaGraphicsResourceGetMappedPointer( buffer, &size, sCudaGraphicsResources.Value( rtgiVertexBuffer ) ) );

    Assert( size != 0 );
}

void CudaRtgiMapBufferPrivate( void** buffer, const rendering::rtgi::PixelBuffer* rtgiPixelBuffer )
{
    Assert( sCudaGraphicsResources.Contains( rtgiPixelBuffer ) );

    cudaGraphicsResource* resources[ 1 ];
    size_t size = 0;

    resources[ 0 ] = sCudaGraphicsResources.Value( rtgiPixelBuffer );

    CUDA_SAFE_CALL( cudaGraphicsMapResources( 1, resources, 0 ) );
    CUDA_SAFE_CALL( cudaGraphicsResourceGetMappedPointer( buffer, &size, sCudaGraphicsResources.Value( rtgiPixelBuffer ) ) );

    Assert( size != 0 );
}

void CudaRtgiMapBufferPrivate( void** buffer, const rendering::rtgi::BufferTexture* rtgiBufferTexture )
{
    Assert( sCudaGraphicsResources.Contains( rtgiBufferTexture ) );

    cudaGraphicsResource* resources[ 1 ];
    size_t size = 0;

    resources[ 0 ] = sCudaGraphicsResources.Value( rtgiBufferTexture );

    CUDA_SAFE_CALL( cudaGraphicsMapResources( 1, resources, 0 ) );
    CUDA_SAFE_CALL( cudaGraphicsResourceGetMappedPointer( buffer, &size, sCudaGraphicsResources.Value( rtgiBufferTexture ) ) );

    Assert( size != 0 );
}

#else

void CudaRtgiRegisterBuffer( const rendering::rtgi::VertexBuffer* vertexBuffer )
{
    const rendering::rtgi::OpenGLVertexBuffer* openGLVertexBuffer = dynamic_cast< const rendering::rtgi::OpenGLVertexBuffer* >( vertexBuffer );

    CUDA_SAFE_CALL( cudaGLRegisterBufferObject( openGLVertexBuffer->GetOpenGLVertexBufferID() ) );
}

void CudaRtgiRegisterBuffer( const rendering::rtgi::PixelBuffer* pixelBuffer )
{
    const rendering::rtgi::OpenGLPixelBuffer* openGLPixelBuffer = dynamic_cast< const rendering::rtgi::OpenGLPixelBuffer* >( pixelBuffer );

    CUDA_SAFE_CALL( cudaGLRegisterBufferObject( openGLPixelBuffer->GetOpenGLPixelBufferID() ) );
}

void CudaRtgiRegisterBuffer( const rendering::rtgi::BufferTexture* bufferTexture )
{
    const rendering::rtgi::OpenGLBufferTexture* openGLBufferTexture = dynamic_cast< const rendering::rtgi::OpenGLBufferTexture* >( bufferTexture );

    CUDA_SAFE_CALL( cudaGLRegisterBufferObject( openGLBufferTexture->GetOpenGLBufferID() ) );
}

void CudaRtgiUnregisterBuffer( const rendering::rtgi::VertexBuffer* vertexBuffer )
{
    const rendering::rtgi::OpenGLVertexBuffer* openGLVertexBuffer = dynamic_cast< const rendering::rtgi::OpenGLVertexBuffer* >( vertexBuffer );

    CUDA_SAFE_CALL( cudaGLUnregisterBufferObject( openGLVertexBuffer->GetOpenGLVertexBufferID() ) );
}

void CudaRtgiUnregisterBuffer( const rendering::rtgi::PixelBuffer* pixelBuffer )
{
    const rendering::rtgi::OpenGLPixelBuffer* openGLPixelBuffer = dynamic_cast< const rendering::rtgi::OpenGLPixelBuffer* >( pixelBuffer );

    CUDA_SAFE_CALL( cudaGLUnregisterBufferObject( openGLPixelBuffer->GetOpenGLPixelBufferID() ) );
}

void CudaRtgiUnregisterBuffer( const rendering::rtgi::BufferTexture* bufferTexture )
{
    const rendering::rtgi::OpenGLBufferTexture* openGLBufferTexture = dynamic_cast< const rendering::rtgi::OpenGLBufferTexture* >( bufferTexture );

    CUDA_SAFE_CALL( cudaGLUnregisterBufferObject( openGLBufferTexture->GetOpenGLBufferID() ) );
}

void CudaRtgiMapBufferPrivate( void** buffer, const rendering::rtgi::VertexBuffer* rtgiVertexBuffer )
{
    CUDA_SAFE_CALL( cudaGLMapBufferObject( (void**)buffer, CudaGetOpenGLBufferID( rtgiVertexBuffer ) ) );
}

void CudaRtgiMapBufferPrivate( void** buffer, const rendering::rtgi::PixelBuffer* rtgiPixelBuffer )
{
    CUDA_SAFE_CALL( cudaGLMapBufferObject( (void**)buffer, CudaGetOpenGLBufferID( rtgiPixelBuffer ) ) );
}

void CudaRtgiMapBufferPrivate( void** buffer, const rendering::rtgi::BufferTexture* rtgiBufferTexture )
{
    CUDA_SAFE_CALL( cudaGLMapBufferObject( (void**)buffer, CudaGetOpenGLBufferID( rtgiBufferTexture ) ) );
}

void CudaRtgiUnmapBuffer( const rendering::rtgi::VertexBuffer* vertexBuffer )
{
    CUDA_SAFE_CALL( cudaGLUnmapBufferObject( dynamic_cast< const rendering::rtgi::OpenGLVertexBuffer* >( vertexBuffer )->GetOpenGLVertexBufferID() ) );
}

void CudaRtgiUnmapBuffer( const rendering::rtgi::PixelBuffer* pixelBuffer )
{
    CUDA_SAFE_CALL( cudaGLUnmapBufferObject( dynamic_cast< const rendering::rtgi::OpenGLPixelBuffer* >( pixelBuffer )->GetOpenGLPixelBufferID() ) );
}

void CudaRtgiUnmapBuffer( const rendering::rtgi::BufferTexture* bufferTexture )
{
    CUDA_SAFE_CALL( cudaGLUnmapBufferObject( dynamic_cast< const rendering::rtgi::OpenGLBufferTexture* >( bufferTexture )->GetOpenGLBufferID() ) );
}

#endif

void CudaAllocateDeviceArray3D( dim3 volumeDimensions, cudaArray** deviceArray3D, unsigned int numBytesPerVoxel, bool isSigned )
{
    cudaChannelFormatDesc channelDesc;

    if ( numBytesPerVoxel == 1 && !isSigned )
    {
        channelDesc = cudaCreateChannelDesc< unsigned char >();
    }
    else
    if ( numBytesPerVoxel == 1 && isSigned )
    {
        channelDesc = cudaCreateChannelDesc< char >();
    }
    else
    if ( numBytesPerVoxel == 2 && !isSigned )
    {
        channelDesc = cudaCreateChannelDesc< unsigned short >();
    }
    else
    if ( numBytesPerVoxel == 2 && isSigned )
    {
        channelDesc = cudaCreateChannelDesc< short >();
    }
    else
    if ( numBytesPerVoxel == 4 && !isSigned )
    {
        channelDesc = cudaCreateChannelDesc< unsigned int >();
    }
    else
    if ( numBytesPerVoxel == 4 && isSigned )
    {
        channelDesc = cudaCreateChannelDesc< int >();
    }

    cudaExtent volumeSize = make_cudaExtent( volumeDimensions.x, volumeDimensions.y, volumeDimensions.z );

    CUDA_SAFE_CALL( cudaMalloc3DArray( deviceArray3D, &channelDesc, volumeSize ) );

    CudaSynchronize();
}

void CudaDeallocateDeviceArray3D( cudaArray** deviceArray )
{
    // free device memory
    CUDA_SAFE_CALL( cudaFreeArray( *deviceArray ) );

    CudaSynchronize();

    *deviceArray = NULL;
}

void CudaMemCopyHostToGlobal( const void* hostData, void* deviceData, unsigned int numBytes )
{
    // upload host memory to device
    CUDA_SAFE_CALL( cudaMemcpy( deviceData, hostData, numBytes, cudaMemcpyHostToDevice ) );

    CudaSynchronize();
}

void CudaMemCopyGlobalToHost( void* hostData, const void* deviceData, unsigned int numBytes )
{
    // download device memory to host
    CUDA_SAFE_CALL( cudaMemcpy( hostData, deviceData, numBytes, cudaMemcpyDeviceToHost ) );

    CudaSynchronize();
}

void CudaMemCopyHostToArray3D( void* hostData, cudaArray* deviceArray, dim3 volumeDimensions, unsigned int elementSizeInBytes )
{
    //
    // copy host data to 3D array
    //
    cudaMemcpy3DParms copyParams = {0};

    cudaPitchedPtr pitchedPointer;

    pitchedPointer.ptr   = hostData;
    pitchedPointer.pitch = volumeDimensions.x * elementSizeInBytes;
    pitchedPointer.xsize = volumeDimensions.x;
    pitchedPointer.ysize = volumeDimensions.y;

    cudaExtent volumeExtent = make_cudaExtent( volumeDimensions.x, volumeDimensions.y, volumeDimensions.z );

    copyParams.srcPtr   = pitchedPointer;
    copyParams.dstArray = deviceArray;
    copyParams.extent   = volumeExtent;
    copyParams.kind     = cudaMemcpyHostToDevice;

    CUDA_SAFE_CALL( cudaMemcpy3D( &copyParams ) );

    CudaSynchronize();
}

void CudaMemCopyArray3DToHost( void* destinationDataHost, cudaArray* sourceArray, dim3 volumeDimensions, unsigned int elementSizeInBytes )
{
    //
    // copy host data to 3D array
    //
    cudaMemcpy3DParms copyParams = {0};

    cudaPitchedPtr pitchedPointer;

    pitchedPointer.ptr   = destinationDataHost;
    pitchedPointer.pitch = volumeDimensions.x * elementSizeInBytes;
    pitchedPointer.xsize = volumeDimensions.x;
    pitchedPointer.ysize = volumeDimensions.y;

    cudaExtent volumeExtent = make_cudaExtent( volumeDimensions.x, volumeDimensions.y, volumeDimensions.z );

    copyParams.srcArray = sourceArray;
    copyParams.dstPtr   = pitchedPointer;
    copyParams.extent   = volumeExtent;
    copyParams.kind     = cudaMemcpyDeviceToHost;

    CUDA_SAFE_CALL( cudaMemcpy3D( &copyParams ) );

    CudaSynchronize();
}

void CudaMemCopyGlobalToArray3D( void* sourceDataGlobal, cudaArray* destinationArray, dim3 volumeDimensions, unsigned int elementSizeInBytes )
{
    CudaMemCopyGlobalToArray3DAsync( sourceDataGlobal, destinationArray, volumeDimensions, elementSizeInBytes );

    CudaSynchronize();
}

void CudaMemCopyGlobalToArray3DAsync( void* sourceDataGlobal, cudaArray* destinationArray, dim3 volumeDimensions, unsigned int elementSizeInBytes )
{
    //
    // copy host data to 3D array
    //
    cudaMemcpy3DParms copyParams = {0};

    cudaPitchedPtr pitchedPointer;

    pitchedPointer.ptr   = sourceDataGlobal;
    pitchedPointer.pitch = volumeDimensions.x * elementSizeInBytes;
    pitchedPointer.xsize = volumeDimensions.x;
    pitchedPointer.ysize = volumeDimensions.y;

    cudaExtent volumeExtent = make_cudaExtent( volumeDimensions.x, volumeDimensions.y, volumeDimensions.z );

    copyParams.srcPtr   = pitchedPointer;
    copyParams.dstArray = destinationArray;
    copyParams.extent   = volumeExtent;
    copyParams.kind     = cudaMemcpyDeviceToDevice;

    CUDA_SAFE_CALL( cudaMemcpy3D( &copyParams ) );
}

void CudaMemCopyGlobalToGlobal( void* destinationData, const void* sourceData, unsigned int numBytes )
{
    // download device memory to host
    CUDA_SAFE_CALL( cudaMemcpy( destinationData, sourceData, numBytes, cudaMemcpyDeviceToDevice ) );

    CudaSynchronize();
}

void CudaMemSet( void* deviceData, int valueToSet, int numBytes )
{
    CudaMemSetAsync( deviceData, valueToSet, numBytes );

    CudaSynchronize();
}

void CudaMemSetAsync( void* deviceData, int valueToSet, int numBytes )
{
    CUDA_SAFE_CALL( cudaMemset( deviceData, valueToSet, numBytes ) );
}

void CudppPlan( CUDPPHandle*       planHandle,
                CUDPPConfiguration config, 
                size_t             n, 
                size_t             rows, 
                size_t             rowPitch )
{
    CUDPPResult result;
    result = cudppPlan( planHandle, config, n, rows, rowPitch );
    Assert( result == CUDPP_SUCCESS );
}

void CudppDestroyPlan( CUDPPHandle planHandle )
{
    CUDPPResult result;
    result = cudppDestroyPlan( planHandle );
    Assert( result == CUDPP_SUCCESS );
}

void CudppSort( CUDPPHandle         planHandle,
                CudaCompactElement* sourceDevice,
                int                 numElements )
{
    int maxNumElements = 512 * 256 * 256;

    Assert( numElements < maxNumElements );

    CUDPPResult result = cudppSort(
        planHandle,
        sourceDevice,
        sourceDevice,
        30,
        numElements );

    CudaSynchronize();

    Assert( result == CUDPP_SUCCESS );
}

void CudppScan( CUDPPHandle               planHandle,
                const CudaCompactElement* sourceDevice,
                CudaCompactElement*       destinationDevice,
                int                       numElements )
{
    int maxNumElements = 512 * 256 * 256;

    Assert( numElements < maxNumElements );

    CUDPPResult result = cudppScan(
        planHandle,
        destinationDevice,
        sourceDevice,
        numElements );

    CudaSynchronize();

    Assert( result == CUDPP_SUCCESS );
}

void CudppCompact( CUDPPHandle               planHandle,
                   const CudaCompactElement* sourceDevice,
                   const CudaCompactElement* validDevice,
                   CudaCompactElement*       destinationDevice,
                   size_t*                   numValidElementsDevice,
                   int                       numElements,
                   size_t*                   numValidElementsHost )
{
    unsigned int maxNumElements        = 512 * 256 * 256;
    unsigned int numFullCompactPasses  = numElements / maxNumElements;
    unsigned int numLeftoverElements   = numElements % maxNumElements;
    bool         doLeftoverCompactPass = numLeftoverElements != 0;

    unsigned int currentSourceIndex      = 0;
    unsigned int currentDestinationIndex = 0;
    size_t       currentNumValidElements = 0;
    *numValidElementsHost                = 0;

    for ( unsigned int i = 0; i < numFullCompactPasses; i++ )
    {
        CUDPPResult result = cudppCompact(
            planHandle,
            destinationDevice + currentDestinationIndex,
            numValidElementsDevice,
            sourceDevice      + currentSourceIndex,
            validDevice       + currentSourceIndex,
            maxNumElements );

        CudaSynchronize();

        Assert( result == CUDPP_SUCCESS );

        CudaMemCopyGlobalToHost( &currentNumValidElements, numValidElementsDevice, sizeof( size_t ) );

        currentSourceIndex      += maxNumElements;
        currentDestinationIndex += currentNumValidElements;
        *numValidElementsHost   += currentNumValidElements;
    }

    if ( doLeftoverCompactPass )
    {
        CUDPPResult result = cudppCompact(
            planHandle,
            destinationDevice + currentDestinationIndex,
            numValidElementsDevice,
            sourceDevice      + currentSourceIndex,
            validDevice       + currentSourceIndex,
            numLeftoverElements );

        CudaSynchronize();

        Assert( result == CUDPP_SUCCESS );

        CudaMemCopyGlobalToHost( &currentNumValidElements, numValidElementsDevice, sizeof( size_t ) );

        currentSourceIndex      += maxNumElements;
        currentDestinationIndex += currentNumValidElements;
        *numValidElementsHost   += currentNumValidElements;
    }
}

void CudaBindTextureToArray( const char* textureName, const cudaArray* deviceArray )
{
    textureReference* textureReferencePointer = CudaGetTextureReference( textureName );

    cudaChannelFormatDesc channelDesc;
    cudaGetChannelDesc( &channelDesc, deviceArray );

    textureReferencePointer->normalized     = false;
    textureReferencePointer->filterMode     = cudaFilterModePoint;   
    textureReferencePointer->addressMode[0] = cudaAddressModeClamp;
    textureReferencePointer->addressMode[1] = cudaAddressModeClamp;
    textureReferencePointer->addressMode[2] = cudaAddressModeClamp;

    CUDA_SAFE_CALL( cudaBindTextureToArray( textureReferencePointer, deviceArray, &channelDesc ) );
}

void CudaUnbindTexture( const char* textureName )
{
    textureReference* textureReferencePointer = CudaGetTextureReference( textureName );

    CUDA_SAFE_CALL( cudaUnbindTexture( textureReferencePointer ) );
}

textureReference* CudaGetTextureReference( const char* textureName )
{
    const textureReference* constTextureReferencePointer = NULL;

    cudaGetTextureReference( &constTextureReferencePointer, textureName );

    Assert( constTextureReferencePointer != NULL );

    return const_cast< textureReference* >( constTextureReferencePointer );
}

GLuint CudaGetOpenGLBufferID( const rendering::rtgi::VertexBuffer* rtgiVertexBuffer )
{
    return dynamic_cast< const rendering::rtgi::OpenGLVertexBuffer* >( rtgiVertexBuffer )->GetOpenGLVertexBufferID(); 
}

GLuint CudaGetOpenGLBufferID( const rendering::rtgi::PixelBuffer* rtgiPixelBuffer )
{
    return dynamic_cast< const rendering::rtgi::OpenGLPixelBuffer* >( rtgiPixelBuffer )->GetOpenGLPixelBufferID(); 
}

GLuint CudaGetOpenGLBufferID( const rendering::rtgi::BufferTexture* rtgiTexture )
{
    return dynamic_cast< const rendering::rtgi::OpenGLBufferTexture* >( rtgiTexture )->GetOpenGLBufferID(); 
}