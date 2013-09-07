#if defined(PLATFORM_OSX)
#define FCOLLADA_NOMINMAX
#include <stdio.h>

#include <dcmtk/dcmdata/dcfilefo.h>
#include <dcmtk/dcmdata/dcuid.h>
#include <dcmtk/dcmdata/dcdeftag.h>

#endif

#include "VolumeSaver.hpp"

#include <boost/foreach.hpp>
#include <boost/filesystem.hpp>

#if defined(PLATFORM_WIN32)
#include <dcmtk/dcmdata/dcfilefo.h>
#include <dcmtk/dcmdata/dcuid.h>
#include <dcmtk/dcmdata/dcdeftag.h>
#endif

#include "math/Utility.hpp"

#include "container/List.hpp"
#include "core/Printf.hpp"
#include "core/String.hpp"

#include "VolumeFileDesc.hpp"

#include <sstream>


void VolumeSaver::SaveVolume( const core::String& fileName, const VolumeDesc& volumeDesc )
{
    boost::filesystem::path filePath( fileName.ToStdString() );

    ReleaseAssert( boost::filesystem::exists( filePath ) );

    if ( boost::filesystem::is_directory( filePath ) )
    {
        SaveVolumeAsDicom( filePath, volumeDesc );
    }

}

void VolumeSaver::SaveVolumeAsDicom( const boost::filesystem::path &dicomPath, const VolumeDesc &volumeDesc )
{
    // This identifies all the images as part of our series. This is the same for every image.
    char seriesUID[ 128 ];
    dcmGenerateUniqueIdentifier( seriesUID, SITE_INSTANCE_UID_ROOT );

    unsigned int   sliceNumVoxels       = volumeDesc.numVoxelsX * volumeDesc.numVoxelsY;
    unsigned int   sliceNumBytes        = sliceNumVoxels * volumeDesc.numBytesPerVoxel;
    unsigned int   bytesAllocatedStored = volumeDesc.numBytesPerVoxel * 8;
    unsigned short pixelRepresentation  = volumeDesc.isSigned ? 1 : 0;

    std::stringstream maxLengthDigitStream;
    maxLengthDigitStream << volumeDesc.numVoxelsZ;
    unsigned int numSliceDigits = maxLengthDigitStream.str().length();

    // For each slice, generate a dicom file and add it as IMG_XXX.dcm where
    // XXX is the image number in the series.
    for( unsigned int z = 0; z < volumeDesc.numVoxelsZ; ++z )
    {
        DcmFileFormat dcmFileFormat;
        OFCondition status;
        
        status = dcmFileFormat.loadFile( volumeDesc.filePaths.At( z ).native_file_string().c_str() );
        ReleaseAssert( status.good() && "Invalid DICOM header information" );

        DcmDataset* dcmDataset = dcmFileFormat.getDataset();

        dcmDataset->putAndInsertString( DCM_SourceApplicationEntityTitle, "GPUSEG" );

        if ( volumeDesc.filePaths.Size() == 0 )
        {
            // Unique image identifier within the series.
            char sopUID[ 128 ];
            dcmGenerateUniqueIdentifier( sopUID, SITE_INSTANCE_UID_ROOT );

			dcmDataset->putAndInsertString( DCM_PhotometricInterpretation,    "MONOCHROME2" );
            dcmDataset->putAndInsertString( DCM_SOPClassUID,                  UID_SecondaryCaptureImageStorage );
            dcmDataset->putAndInsertString( DCM_SeriesInstanceUID,            seriesUID );
            dcmDataset->putAndInsertString( DCM_SOPInstanceUID,               sopUID );

		    dcmDataset->putAndInsertUint16( DCM_Rows,    volumeDesc.numVoxelsY );
            dcmDataset->putAndInsertUint16( DCM_Columns, volumeDesc.numVoxelsX );

            std::stringstream tmpStream;
            tmpStream << "0\\0\\" << z;

			dcmDataset->putAndInsertString( DCM_ImagePositionPatient, tmpStream.str().c_str() );
            dcmDataset->putAndInsertString( DCM_SliceThickness,       "1" );
            dcmDataset->putAndInsertString( DCM_PixelSpacing,         "1\\1" );
            dcmDataset->putAndInsertString( DCM_ImagerPixelSpacing,   "1\\1" );
		}

        dcmDataset->putAndInsertUint16( DCM_SamplesPerPixel,         1 );
        dcmDataset->putAndInsertUint16( DCM_BitsAllocated,           bytesAllocatedStored );
        dcmDataset->putAndInsertUint16( DCM_BitsStored,              bytesAllocatedStored );
        dcmDataset->putAndInsertUint16( DCM_HighBit,                 bytesAllocatedStored - 1 );
        dcmDataset->putAndInsertUint16( DCM_PixelRepresentation,     pixelRepresentation );
        dcmDataset->putAndInsertSint16( DCM_SmallestImagePixelValue, volumeDesc.minValue );
        dcmDataset->putAndInsertSint16( DCM_LargestImagePixelValue,  volumeDesc.maxValue );
        dcmDataset->putAndInsertSint16( DCM_RescaleIntercept,        0 );
        dcmDataset->putAndInsertSint16( DCM_RescaleSlope,            1 );

        dcmDataset->putAndInsertUint8Array( DCM_PixelData, &( reinterpret_cast< unsigned char* >( volumeDesc.volumeData )[ z * sliceNumBytes ] ), sliceNumBytes );

        // We will write out to IMG_XXXX.dcm. We need to replace XXXX with the slice number with 0's
        // prepended, ie: for slice 123 the path should be IMG_0123.dcm, likewise for 4 -> IMG_0004.dcm
        // However, this depends on numSliceDigits. So if we only have 128 slices the file names
        // become IMG_XXX.dcm
        std::stringstream sliceNumDigitStream;
        sliceNumDigitStream << z;

        unsigned char numZeroesPrepend = numSliceDigits - sliceNumDigitStream.str().length();
        std::stringstream fileNameStream;
        fileNameStream << "IMG_";

        for( unsigned int i = 0; i < numZeroesPrepend; ++i )
        {
            fileNameStream << 0;
        }

        fileNameStream << z << ".dcm";

        core::String outputPath = ( dicomPath / fileNameStream.str() ).native_file_string();

        status = dcmFileFormat.saveFile( outputPath.ToAscii(), EXS_LittleEndianExplicit );
        ReleaseAssert( status.good() && "Cannot write DICOM file" );
    }
}