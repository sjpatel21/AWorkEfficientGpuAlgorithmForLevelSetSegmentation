#if defined(PLATFORM_OSX)
#define FCOLLADA_NOMINMAX
#include <stdio.h>

#include <dcmtk/dcmimgle/dcmimage.h>
#include <dcmtk/dcmimgle/dipixel.h>
#include <dcmtk/dcmdata/dcdeftag.h>
#include <dcmtk/dcmdata/dcfilefo.h>
#endif

#include "VolumeLoader.hpp"

#include <boost/foreach.hpp>
#include <boost/filesystem.hpp>

#if defined(PLATFORM_WIN32)
#include <dcmtk/dcmimgle/dcmimage.h>
#include <dcmtk/dcmimgle/dipixel.h>
#include <dcmtk/dcmdata/dcdeftag.h>
#include <dcmtk/dcmdata/dcfilefo.h>
#endif

#include "math/Utility.hpp"

#include "container/List.hpp"
#include "container/Map.hpp"

#include "VolumeFileDesc.hpp"

EP_Representation GetPixelRepresentationFromDicomMetaData( const boost::filesystem::path& dicomPath );

struct DicomImageFileDesc
{
    boost::filesystem::path filePath;
    DicomImage*             dicomImage;
};

VolumeDesc VolumeLoader::LoadVolume( const VolumeFileDesc& volumeFileDesc )
{
    boost::filesystem::path filePath( volumeFileDesc.fileName.ToStdString() );

    ReleaseAssert( boost::filesystem::exists( filePath ) );

    if ( boost::filesystem::is_directory( filePath ) )
    {
        return LoadVolumeDicom( filePath, volumeFileDesc );
    }
    else
    {
        return LoadVolumeRaw( filePath, volumeFileDesc );
    }
}

void VolumeLoader::UnloadVolume( VolumeDesc& volumeDesc )
{
    Assert( volumeDesc.volumeData != NULL );

    delete[] volumeDesc.volumeData;

	VolumeDesc newVolumeDesc;
    volumeDesc = newVolumeDesc;
}

VolumeDesc VolumeLoader::LoadVolumeDicom( const boost::filesystem::path& dicomPath, const VolumeFileDesc& volumeFileDesc )
{
    boost::filesystem::directory_iterator dicomFilesBegin( dicomPath );
    boost::filesystem::directory_iterator dicomFilesEnd;

    container::Map< float, DicomImageFileDesc > dicomImageFileDescs;

    //
    // create DCMTK objects for all dicom files in directory
    //
    boost_foreach ( boost::filesystem::path dicomFile, std::make_pair( dicomFilesBegin, dicomFilesEnd ) )
    {
        if ( !boost::filesystem::is_directory( dicomFile ) )
        {
            DicomImage* dicomImage = new DicomImage( dicomFile.native_file_string().c_str() );

            ReleaseAssert( dicomImage != NULL && "Selected directory contains invalid DICOM files" );

            EI_Status status = dicomImage->getStatus();

            ReleaseAssert( status == EIS_Normal && "Selected directory contains invalid DICOM files" );

            DcmFileFormat dcmFileFormat;
            OFCondition metadataStatus;
            
            metadataStatus = dcmFileFormat.loadFile( dicomFile.native_file_string().c_str() );
            ReleaseAssert( metadataStatus.good() && "Invalid DICOM header information" );

            DcmDataset* dcmDataset = dcmFileFormat.getDataset();

            Float64 imagePositionPatientZ;

            metadataStatus = dcmDataset->findAndGetFloat64( DCM_ImagePositionPatient, imagePositionPatientZ, 2, true );

            ReleaseAssert( metadataStatus.good() && "'Image Position (Patient)' is not present in the DICOM metadata. The tag I'm looking for is (group,element)=(0x20,0x32)" );
            ReleaseAssert( !dicomImageFileDescs.Contains( imagePositionPatientZ ) );

            DicomImageFileDesc dicomImageFileDesc;
            dicomImageFileDesc.filePath   = dicomFile.native_file_string();
            dicomImageFileDesc.dicomImage = dicomImage;

            dicomImageFileDescs.Insert( static_cast< float >( imagePositionPatientZ ), dicomImageFileDesc );
        }
    }

    ReleaseAssert( dicomImageFileDescs.Size() > 0 );

    //
    // determine global min max values
    //
    float globalMinValue = 65535.0f;
    float globalMaxValue = -1.0f;

    foreach ( DicomImageFileDesc dicomImageFileDesc, dicomImageFileDescs )
    {
        double currentImageMin, currentImageMax;

        dicomImageFileDesc.dicomImage->getMinMaxValues( currentImageMin, currentImageMax, 0 );

        globalMinValue = math::Min( (float) globalMinValue, (float) currentImageMin );
        globalMaxValue = math::Max( (float) globalMaxValue, (float) currentImageMax );
    }

    //
    // determine format from the 0th dicom image
    //
    DcmFileFormat dcmFileFormat;
    OFCondition status;
    

    DicomImageFileDesc representativeDicomImageFileDesc = dicomImageFileDescs.Value( dicomImageFileDescs.Keys().At( 0 ) );

    status = dcmFileFormat.loadFile( representativeDicomImageFileDesc.filePath.native_file_string().c_str() );
    ReleaseAssert( status.good() && "Invalid DICOM header information" );

    DcmDataset* dcmDataset = dcmFileFormat.getDataset();
    Float64     pixelSpacingMillimetersWidth;
    Float64     pixelSpacingMillimetersHeight;
    Float64     spacingBetweenSlicesMillimeters;

    status = dcmDataset->findAndGetFloat64( DCM_PixelSpacing, pixelSpacingMillimetersWidth, 0, true );
    ReleaseAssert( status.good() && "'Pixel Spacing' is not present in the DICOM metadata. The tag I'm looking for is (group,element)=(0x28,0x30)" );

    status = dcmDataset->findAndGetFloat64( DCM_PixelSpacing, pixelSpacingMillimetersHeight, 1, true );
    ReleaseAssert( status.good() && "'Pixel Spacing' is not present in the DICOM metadata. The tag I'm looking for is (group,element)=(0x28,0x30)" );

    //
    // use 'spacing between slices' if present, since this takes into account the possible gaps between slices.
    // use 'slice thickness' otherwise, which assumes there are no gaps between slices.
    //
    status = dcmDataset->findAndGetFloat64( DCM_SpacingBetweenSlices, spacingBetweenSlicesMillimeters, 0, true );
    
    if ( !status.good() )
    {
        status = dcmDataset->findAndGetFloat64( DCM_SliceThickness, spacingBetweenSlicesMillimeters, 0, true );
        ReleaseAssert( status.good() && "Neither 'Spacing Between Slices' nor 'Slice Thickness' are present in the DICOM metadata. The tags I'm looking for are (group,element)=(0x18,0x88) or (group,element)=(0x18,0x50)" );
    }

    const DiPixel*    pixelData                 = representativeDicomImageFileDesc.dicomImage->getInterData();
    unsigned int      globalWidthNumVoxels      = representativeDicomImageFileDesc.dicomImage->getWidth();
    unsigned int      globalHeightNumVoxels     = representativeDicomImageFileDesc.dicomImage->getHeight();
    unsigned int      globalDepthNumVoxels      = dicomImageFileDescs.Size();
    unsigned int      globalNumPlanes           = pixelData->getPlanes();
    unsigned int      globalNumElementsPerImage = pixelData->getCount();
    unsigned int      globalNumElements         = globalNumElementsPerImage * globalDepthNumVoxels;
    unsigned int      globalNumBytesPerElement  = -1;
    bool              globalIsSigned            = false;
    EP_Representation globalRepresentation      = GetPixelRepresentationFromDicomMetaData( representativeDicomImageFileDesc.filePath );

    Assert( globalNumPlanes == 1 );

    switch ( globalRepresentation )
    {
        case EPR_Uint8:
            globalNumBytesPerElement = 1;
            globalIsSigned           = false;
            break;
        case EPR_Sint8:
            globalNumBytesPerElement = 1;
            globalIsSigned           = true;
            break;
        case EPR_Uint16:
            globalNumBytesPerElement = 2;
            globalIsSigned           = false;
            break;
        case EPR_Sint16:
            globalNumBytesPerElement = 2;
            globalIsSigned           = true;
            break;
        case EPR_Uint32:
            globalNumBytesPerElement = 4;
            globalIsSigned           = false;
            break;
        case EPR_Sint32:
            globalNumBytesPerElement = 4;
            globalIsSigned           = true;
            break;
        default:
            Assert( 0 );
            break;
    }

    //
    // allocate space for the entire volume
    //
    unsigned int                               globalNumBytes         = globalNumElements * globalNumBytesPerElement;
    unsigned int                               globalNumBytesPerImage = globalNumElementsPerImage * globalNumBytesPerElement;
    unsigned char*                             volumeData             = new unsigned char[ globalNumBytes ];
    container::List< boost::filesystem::path > filePaths;

    //
    // copy volume from internal DCMTK buffers into our own buffer
    //
    int   currentVolumeDataIndex = 0;
    int   globalDicomBytesRead   = 0;

    foreach ( DicomImageFileDesc dicomImageFileDesc, dicomImageFileDescs )
    {
        filePaths.Append( dicomImageFileDesc.filePath );

        const DiPixel* pixelData              = dicomImageFileDesc.dicomImage->getInterData();
        EP_Representation pixelRepresentation = GetPixelRepresentationFromDicomMetaData( dicomImageFileDesc.filePath.native_file_string().c_str() );

        ReleaseAssert( pixelRepresentation    == globalRepresentation );
        ReleaseAssert( pixelData->getPlanes() == globalNumPlanes );
        ReleaseAssert( pixelData->getCount()  == globalNumElementsPerImage );

        const void* rawPixelData = pixelData->getData();

        ReleaseAssert( rawPixelData != NULL );

        memcpy( volumeData + globalDicomBytesRead, rawPixelData, globalNumBytesPerImage );

        globalDicomBytesRead += globalNumBytesPerImage;
    }

    ReleaseAssert( filePaths.Size() == globalDepthNumVoxels );

    //
    // clean up the DCMTK objects we created at the top of this function
    //
    foreach ( DicomImageFileDesc dicomImageFileDesc, dicomImageFileDescs )
    {
        delete dicomImageFileDesc.dicomImage;
    }

    dicomImageFileDescs.Clear();

    //
    // fill in output struct
    //
    VolumeDesc volumeDesc;

    volumeDesc.upDirection            = volumeFileDesc.upDirection;
    volumeDesc.voxelHeightMillimeters = pixelSpacingMillimetersHeight;
    volumeDesc.voxelWidthMillimeters  = pixelSpacingMillimetersWidth;
    volumeDesc.voxelDepthMillimeters  = spacingBetweenSlicesMillimeters;
    volumeDesc.volumeData             = volumeData;
    volumeDesc.numVoxelsX             = globalWidthNumVoxels;
    volumeDesc.numVoxelsY             = globalHeightNumVoxels;
    volumeDesc.numVoxelsZ             = globalDepthNumVoxels;
    volumeDesc.zAnisotropy            = volumeFileDesc.zAnisotropy;
    volumeDesc.minValue               = globalMinValue;
    volumeDesc.maxValue               = globalMaxValue;
    volumeDesc.numBytesPerVoxel       = globalNumBytesPerElement;
    volumeDesc.isSigned               = globalIsSigned;
    volumeDesc.filePaths              = filePaths;

    return volumeDesc;
}

VolumeDesc VolumeLoader::LoadVolumeRaw( const boost::filesystem::path& rawPath, const VolumeFileDesc& volumeFileDesc )
{
    FILE*          filePointer       = NULL;
    unsigned int   globalNumElements = volumeFileDesc.numVoxelsX * volumeFileDesc.numVoxelsY * volumeFileDesc.numVoxelsZ;
    unsigned int   globalNumBytes    = globalNumElements * volumeFileDesc.numBytesPerVoxel;
    unsigned char* volumeData        = new unsigned char[ globalNumBytes ];

    //
    // read the file
    //
#if defined(PLATFORM_WIN32)
    
    errno_t errorCode;
    errorCode = fopen_s( &filePointer, volumeFileDesc.fileName.ToAscii(), "rb" );
    
    Assert( errorCode   == 0 );

#elif defined(PLATFORM_OSX)
    
    filePointer = fopen( volumeFileDesc.fileName.ToAscii(), "rb" );
    
#endif
    
    Assert( filePointer != NULL );

    size_t numBytesRead;
    numBytesRead = fread( volumeData, 1, globalNumBytes, filePointer );
    fclose( filePointer );

    Assert( numBytesRead == globalNumBytes );

    //
    // determine the min and max values
    //
    float globalMinValue = 65535.0f;
    float globalMaxValue = -1.0f;

    if ( volumeFileDesc.numBytesPerVoxel == 1 && !volumeFileDesc.isSigned )
    {
        unsigned char* volumeDataTyped = (unsigned char*)volumeData;

        for ( unsigned int i = 0; i < globalNumElements; i++ )
        {
            globalMinValue = math::Min( (float) globalMinValue, (float)( volumeDataTyped[ i ] ) );
            globalMaxValue = math::Max( (float) globalMaxValue, (float)( volumeDataTyped[ i ] ) );
        }
    }
    else
    if ( volumeFileDesc.numBytesPerVoxel == 1 && volumeFileDesc.isSigned )
    {
        char* volumeDataTyped = (char*)volumeData;

        for ( unsigned int i = 0; i < globalNumElements; i++ )
        {
            globalMinValue = math::Min( (float) globalMinValue, (float)( volumeDataTyped[ i ] ) );
            globalMaxValue = math::Max( (float) globalMaxValue, (float)( volumeDataTyped[ i ] ) );
        }
    }
    else
    if ( volumeFileDesc.numBytesPerVoxel == 2 && !volumeFileDesc.isSigned )
    {
        unsigned short* volumeDataTyped = (unsigned short*)volumeData;

        for ( unsigned int i = 0; i < globalNumElements; i++ )
        {
            globalMinValue = math::Min( (float) globalMinValue, (float)( volumeDataTyped[ i ] ) );
            globalMaxValue = math::Max( (float) globalMaxValue, (float)( volumeDataTyped[ i ] ) );
        }
    }
    else
    if ( volumeFileDesc.numBytesPerVoxel == 2 && volumeFileDesc.isSigned )
    {
        short* volumeDataTyped = (short*)volumeData;

        for ( unsigned int i = 0; i < globalNumElements; i++ )
        {
            globalMinValue = math::Min( (float) globalMinValue, (float)( volumeDataTyped[ i ] ) );
            globalMaxValue = math::Max( (float) globalMaxValue, (float)( volumeDataTyped[ i ] ) );
        }
    }
    else
    if ( volumeFileDesc.numBytesPerVoxel == 4 && !volumeFileDesc.isSigned )
    {
        unsigned int* volumeDataTyped = (unsigned int*)volumeData;

        for ( unsigned int i = 0; i < globalNumElements; i++ )
        {
            globalMinValue = math::Min( (float) globalMinValue, (float)( volumeDataTyped[ i ] ) );
            globalMaxValue = math::Max( (float) globalMaxValue, (float)( volumeDataTyped[ i ] ) );
        }
    }
    else
    if ( volumeFileDesc.numBytesPerVoxel == 4 && volumeFileDesc.isSigned )
    {
        int* volumeDataTyped = (int*)volumeData;

        for ( unsigned int i = 0; i < globalNumElements; i++ )
        {
            globalMinValue = math::Min( (float) globalMinValue, (float)( volumeDataTyped[ i ] ) );
            globalMaxValue = math::Max( (float) globalMaxValue, (float)( volumeDataTyped[ i ] ) );
        }
    }
    else
    {
        Assert( 0 );
    }

    VolumeDesc volumeDesc;

    volumeDesc.upDirection      = volumeFileDesc.upDirection;
    volumeDesc.numVoxelsX       = volumeFileDesc.numVoxelsX;
    volumeDesc.numVoxelsY       = volumeFileDesc.numVoxelsY;
    volumeDesc.numVoxelsZ       = volumeFileDesc.numVoxelsZ;
    volumeDesc.zAnisotropy      = volumeFileDesc.zAnisotropy;
    volumeDesc.minValue         = globalMinValue;
    volumeDesc.maxValue         = globalMaxValue;
    volumeDesc.volumeData       = volumeData;
    volumeDesc.numBytesPerVoxel = volumeFileDesc.numBytesPerVoxel;
    volumeDesc.isSigned         = volumeFileDesc.isSigned;

    return volumeDesc;
}

EP_Representation GetPixelRepresentationFromDicomMetaData( const boost::filesystem::path& dicomPath )
{
    EP_Representation pixelRepresentation;

    // DicomImage constructs its pixel data representation from the min and max values it finds
    // in the data, instead of reading the dicom meta data.
    // This causes problems if we have a signed volume but one of the slices doesn't actually have
    // data that is signed. ie: The first slice of a volume may contain values on [1,127] but the 
    // next slice contains [-127, 127]. DicomImage will determine the first image is Uint8 and the 
    // second image is Sint8, however, since both belong to the same series they were most likely
    // written with a pixel representation of signed.
    DcmFileFormat dicomFile;
    dicomFile.loadFile( dicomPath.native_file_string().c_str() );

    DcmDataset* dataset             = dicomFile.getDataset();
    unsigned short signedOrUnsigned = 0;
    unsigned short bitsPerPixel     = 0;

    ReleaseAssert( dataset->findAndGetUint16( DCM_PixelRepresentation, signedOrUnsigned ).good() && "Cannot read pixel representation from DICOM file." );
    ReleaseAssert( dataset->findAndGetUint16( DCM_BitsAllocated,       bitsPerPixel ).good()     && "Cannot read pixel representation from DICOM file." );
    
    // For DCM_PixelRepresentation, 1 == signed and 0 == unsigned
    if( signedOrUnsigned == 1 )
    {
        switch( bitsPerPixel )
        {
            case 8:
                pixelRepresentation = EPR_Sint8;
                break;
            case 16:
                pixelRepresentation = EPR_Sint16;
                break;
            case 32:
                pixelRepresentation = EPR_Sint32;
                break;
            default:
                ReleaseAssert( 0 );
                break;
        }
    }
    else
    {
        switch( bitsPerPixel )
        {
        case 8:
            pixelRepresentation = EPR_Uint8;
            break;
        case 16:
            pixelRepresentation = EPR_Uint16;
            break;
        case 32:
            pixelRepresentation = EPR_Uint32;
            break;
        default:
            ReleaseAssert( 0 );
            break;
        }
    }

    return pixelRepresentation;
}