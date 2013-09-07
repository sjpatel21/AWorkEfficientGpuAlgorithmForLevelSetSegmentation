#include "widgets/MainWindow.hpp"

#include <boost/filesystem.hpp>

#if defined(PLATFORM_OSX)

#ifdef check
#undef check
#endif

#endif


#include <QtCore/QTimer>
#include <QtCore/QFile>
#include <QtCore/QMutex>
#include <QtGui/QMenuBar>
#include <QtGui/QToolBar>
#include <QtGui/QDockWidget>
#include <QtGui/QPaintEvent>
#include <QtGui/QStatusBar>
#include <QtGui/QLabel>
#include <QtGui/QStandardItemModel>
#include <QtGui/QHeaderView>
#include <QtGui/QTreeView>
#include <QtGui/QAction>
#include <QtGui/QPushButton>
#include <QtGui/QFileDialog>
#include <QtGui/QMessageBox>
#include <QtGui/QFileDialog>
#include <QtGui/QProgressBar>
#include <QtGui/QApplication>
#include <QtGui/QLineEdit>
#include <QtGui/QRadioButton>
#include <QtXml/QDomDocument>

#include "widgets/ParameterManagerDelegate.hpp"

#include "core/Assert.hpp"
#include "core/RefCounted.hpp"
#include "core/String.hpp"

#include "content/ParameterManager.hpp"

#include "Engine.hpp"
#include "VolumeFileDesc.hpp"
#include "Config.hpp"

#include "userinputhandlers/SketchUserInputHandler.hpp"
#include "userinputhandlers/MayaCameraUserInputHandler.hpp"

#include "widgets/RenderingWindow.hpp"


#undef foreach
#define foreach Q_FOREACH

static const int DATA_NAME_ROLE      = Qt::EditRole;
static const int DATA_VALUE_ROLE     = Qt::EditRole;
static const int PARAMETER_NAME_ROLE = Qt::UserRole;
static const int PARAMETER_TYPE_ROLE = Qt::UserRole + 1;
static const int SYSTEM_NAME_ROLE    = Qt::UserRole + 2;

static QMutex sUpdatingEngineMutex( QMutex::Recursive );

MainWindow::MainWindow( QWidget* parent ) :
QMainWindow                 ( parent ),
mCurrentUserInputHandler    ( NULL ),
mRenderingWindow            ( NULL ),
mSettingsWindow             ( NULL ),
mVolumeDirectoryDescriptionDialog( NULL ),
mEngine                     ( NULL ),
mMainWindowForm             ( NULL ),
mSettingsWindowForm         ( NULL ),
mProgressBar                ( NULL ),
mStandardItemModel          ( NULL ),
mParameterManagerDelegate             ( NULL ),
mMenuBar                    ( NULL ),
mToolBar                    ( NULL ),
mStatusBar                  ( NULL ),
mStatusBarLabel             ( NULL )
{
    // initialize the windows first
    initializeRenderingWindow();
    initializeParameterManagerWindow();
    initializeMainWindow();
    initializeVolumeDescriptionDialogs();

    initializeActions();

    // force any resize events to happen so we can initialize the engine to the correct size
    show();

    // now we can initialize the engine
    initializeEngine();

    // now we can safely initialize everything else that depends on the engine
    initializeTimer();
    initializeUserInputHandlers();

    // now load the actual settings from the engine
    loadParameters();

    // initialize to the maya camera tool
    selectTool( Tool_MayaCamera );

    // load the file specified on the command line
    if ( qApp->arguments().size() > 1 )
    {
        Assert( qApp->arguments().size() == 2 );

        openProject( qApp->arguments().at( 1 ) );
    }
}


MainWindow::~MainWindow()
{
    terminateUserInputHandlers();
    terminateTimer();
    terminateActions();

    terminateEngine();

    terminateVolumeDescriptionDialogs();
    terminateMainWindow();
    terminateParameterManagerWindow();
    terminateRenderingWindow();
}

void MainWindow::assetChangedAfterReloadCallback( void* )
{
    loadParameters();
}

void MainWindow::segmentationFinishedCallback( void* )
{
    mProgressBar->setMaximum( 1 );
    mProgressBar->reset();
}

void MainWindow::segmentationStartedCallback( void* )
{
    mProgressBar->setMaximum( 0 );
    mProgressBar->reset();
}

void MainWindow::parametersChangedCallback( void* )
{
    // I know this exists in assetChangedAfterReloadCallback but I didn't want to
    // reused it because of the specific name
    loadParameters();
}

void MainWindow::updateUserEditableParameter( QModelIndex index )
{
    //
    // get info about the changed parameter
    //
    content::ParameterType parameterType = static_cast< content::ParameterType >( qvariant_cast< int >( index.data( PARAMETER_TYPE_ROLE ) ) );
    QString                parameterName = qvariant_cast< QString >( index.data( PARAMETER_NAME_ROLE ) );
    QString                systemName    = qvariant_cast< QString >( index.data( SYSTEM_NAME_ROLE ) );

    Assert( content::ParameterManager::Contains( systemName, parameterName ) );

    switch ( parameterType )
    {
    case content::ParameterType_Float:
        {
            //
            // retrieve new value from the model
            //
            float newValue = static_cast< float >( qvariant_cast< double >( index.data( DATA_VALUE_ROLE ) ) );


            //
            // set the value in the engine
            //
            content::ParameterManager::SetParameter( systemName, parameterName, newValue );
        }
        break;

    case content::ParameterType_Vector3:
        {
            //
            // retrieve new value from the model
            //
            QModelIndex parentIndex   = index.parent();

            QModelIndex xIndex        = index.sibling( 0, 1 );
            QModelIndex yIndex        = index.sibling( 1, 1 );
            QModelIndex zIndex        = index.sibling( 2, 1 );

            math::Vector3 newValue;

            newValue[ math::X ] = static_cast< float >( qvariant_cast< double >( xIndex.data( DATA_VALUE_ROLE ) ) );
            newValue[ math::Y ] = static_cast< float >( qvariant_cast< double >( yIndex.data( DATA_VALUE_ROLE ) ) );
            newValue[ math::Z ] = static_cast< float >( qvariant_cast< double >( zIndex.data( DATA_VALUE_ROLE ) ) );


            //
            // set the value in the engine
            //
            content::ParameterManager::SetParameter( systemName, parameterName, newValue );
        }
        break;

    case content::ParameterType_Vector4:
        {
            //
            // retrieve new value from the model
            //
            QModelIndex parentIndex   = index.parent();

            QModelIndex xIndex        = index.sibling( 0, 1 );
            QModelIndex yIndex        = index.sibling( 1, 1 );
            QModelIndex zIndex        = index.sibling( 2, 1 );
            QModelIndex hIndex        = index.sibling( 3, 1 );

            math::Vector4 newValue;

            newValue[ math::X ] = static_cast< float >( qvariant_cast< double >( xIndex.data( DATA_VALUE_ROLE ) ) );
            newValue[ math::Y ] = static_cast< float >( qvariant_cast< double >( yIndex.data( DATA_VALUE_ROLE ) ) );
            newValue[ math::Z ] = static_cast< float >( qvariant_cast< double >( zIndex.data( DATA_VALUE_ROLE ) ) );
            newValue[ math::H ] = static_cast< float >( qvariant_cast< double >( hIndex.data( DATA_VALUE_ROLE ) ) );

            //
            // set the value in the engine
            //
            content::ParameterManager::SetParameter( systemName, parameterName, newValue );
        }
        break;

    case content::ParameterType_Matrix44:
        Assert( 0 );
        break;

    default:
        Assert( 0 );
        break;
    }
}


//
// rendering window event handling
//
void MainWindow::renderingWindowMousePressEvent( QMouseEvent* mouseEvent )
{
    mCurrentUserInputHandler->renderingWindowMousePressEvent( mouseEvent );
}

void MainWindow::renderingWindowMouseReleaseEvent( QMouseEvent* mouseEvent )
{
    mCurrentUserInputHandler->renderingWindowMouseReleaseEvent( mouseEvent );
}

void MainWindow::renderingWindowMouseMoveEvent( QMouseEvent* mouseEvent )
{
    mCurrentUserInputHandler->renderingWindowMouseMoveEvent( mouseEvent );
}

void MainWindow::renderingWindowMouseWheelEvent( QWheelEvent* wheelEvent )
{
    mCurrentUserInputHandler->renderingWindowMouseWheelEvent( wheelEvent );
}

void MainWindow::renderingWindowResizeEvent( QResizeEvent* resizeEvent )
{
    if ( mEngine != NULL )
    {
        mEngine->SetViewport( resizeEvent->size().width(), resizeEvent->size().height() );
    }
}

void MainWindow::renderingWindowkeyPressEvent( QKeyEvent* keyEvent )
{
    mCurrentUserInputHandler->renderingWindowKeyPressEvent( keyEvent );
}

void MainWindow::renderingWindowkeyReleaseEvent( QKeyEvent* keyEvent )
{
    mCurrentUserInputHandler->renderingWindowKeyReleaseEvent( keyEvent );
}

bool MainWindow::update()
{
    if ( mEngine != NULL )
    {
        sUpdatingEngineMutex.lock();

        mEngine->Update();

        sUpdatingEngineMutex.unlock();
    }

    return true;
}


void MainWindow::exitApplication( bool )
{
    close();
}

void MainWindow::volumeDirectoryDescriptionParameterFileSelect()
{
    QString fileName = QFileDialog::getOpenFileName( this, "Open File", "parameters", "Parameter XML Files (*.par)" );

    mVolumeDirectoryDescriptionDialog->findChild< QLineEdit* >( "parameterFileName" )->setText( fileName );
}

void MainWindow::volumeFileDescriptionParameterFileSelect()
{
    QString fileName = QFileDialog::getOpenFileName( this, "Open File", "parameters", "Parameter XML Files (*.par)" );

    mVolumeFileDescriptionDialog->findChild< QLineEdit* >( "parameterFileName" )->setText( fileName );
}

void MainWindow::volumeDirectoryDescriptionOK()
{
    bool convertOK;

    // file name
    mVolumeFileDesc.fileName = mNextFileName;

    // z anisotropy
    float zAnisotropy = mVolumeDirectoryDescriptionDialog->findChild< QLineEdit* >( "zAnisotropy" )->text().toFloat( &convertOK );

    if ( convertOK )
    {
        mVolumeFileDesc.zAnisotropy = zAnisotropy;
    }

    // up direction
    if ( mVolumeDirectoryDescriptionDialog->findChild< QRadioButton* >( "positiveX" )->isChecked() )
    {
        mVolumeFileDesc.upDirection = math::Vector3( 1, 0, 0 );
    }

    if ( mVolumeDirectoryDescriptionDialog->findChild< QRadioButton* >( "positiveY" )->isChecked() )
    {
        mVolumeFileDesc.upDirection = math::Vector3( 0, 1, 0 );
    }

    if ( mVolumeDirectoryDescriptionDialog->findChild< QRadioButton* >( "positiveZ" )->isChecked() )
    {
        mVolumeFileDesc.upDirection = math::Vector3( 0, 0, 1 );
    }

    if ( mVolumeDirectoryDescriptionDialog->findChild< QRadioButton* >( "negativeX" )->isChecked() )
    {
        mVolumeFileDesc.upDirection = math::Vector3( -1, 0, 0 );
    }

    if ( mVolumeDirectoryDescriptionDialog->findChild< QRadioButton* >( "negativeY" )->isChecked() )
    {
        mVolumeFileDesc.upDirection = math::Vector3( 0, -1, 0 );
    }

    if ( mVolumeDirectoryDescriptionDialog->findChild< QRadioButton* >( "negativeZ" )->isChecked() )
    {
        mVolumeFileDesc.upDirection = math::Vector3( 0, 0, -1 );
    }

    if ( mVolumeDirectoryDescriptionDialog->findChild< QRadioButton* >( "negativeZ" )->isChecked() )
    {
        mVolumeFileDesc.upDirection = math::Vector3( 0, 0, -1 );
    }

    loadVolume( mVolumeFileDesc, mVolumeDirectoryDescriptionDialog->findChild< QLineEdit* >( "parameterFileName" )->text() );

    mVolumeDirectoryDescriptionDialog->findChild< QLineEdit* >( "parameterFileName" )->setText( "" );
}

void MainWindow::volumeFileDescriptionOK()
{
    bool convertOK;

    // file name
    mVolumeFileDesc.fileName = mNextFileName;

    // z anisotropy
    float zAnisotropy = mVolumeFileDescriptionDialog->findChild< QLineEdit* >( "zAnisotropy" )->text().toFloat( &convertOK );

    if ( convertOK )
    {
        mVolumeFileDesc.zAnisotropy = zAnisotropy;
    }

    // up direction
    if ( mVolumeFileDescriptionDialog->findChild< QRadioButton* >( "positiveX" )->isChecked() )
    {
        mVolumeFileDesc.upDirection = math::Vector3( 1, 0, 0 );
    }

    if ( mVolumeFileDescriptionDialog->findChild< QRadioButton* >( "positiveY" )->isChecked() )
    {
        mVolumeFileDesc.upDirection = math::Vector3( 0, 1, 0 );
    }

    if ( mVolumeFileDescriptionDialog->findChild< QRadioButton* >( "positiveZ" )->isChecked() )
    {
        mVolumeFileDesc.upDirection = math::Vector3( 0, 0, 1 );
    }

    if ( mVolumeFileDescriptionDialog->findChild< QRadioButton* >( "negativeX" )->isChecked() )
    {
        mVolumeFileDesc.upDirection = math::Vector3( -1, 0, 0 );
    }

    if ( mVolumeFileDescriptionDialog->findChild< QRadioButton* >( "negativeY" )->isChecked() )
    {
        mVolumeFileDesc.upDirection = math::Vector3( 0, -1, 0 );
    }

    if ( mVolumeFileDescriptionDialog->findChild< QRadioButton* >( "negativeZ" )->isChecked() )
    {
        mVolumeFileDesc.upDirection = math::Vector3( 0, 0, -1 );
    }

    if ( mVolumeFileDescriptionDialog->findChild< QRadioButton* >( "negativeZ" )->isChecked() )
    {
        mVolumeFileDesc.upDirection = math::Vector3( 0, 0, -1 );
    }

    // num voxels
    int numVoxelsX = mVolumeFileDescriptionDialog->findChild< QLineEdit* >( "numVoxelsX" )->text().toInt( &convertOK );

    if ( convertOK )
    {
        mVolumeFileDesc.numVoxelsX = numVoxelsX;
    }

    int numVoxelsY = mVolumeFileDescriptionDialog->findChild< QLineEdit* >( "numVoxelsY" )->text().toInt( &convertOK );

    if ( convertOK )
    {
        mVolumeFileDesc.numVoxelsY = numVoxelsY;
    }

    int numVoxelsZ = mVolumeFileDescriptionDialog->findChild< QLineEdit* >( "numVoxelsZ" )->text().toInt( &convertOK );

    if ( convertOK )
    {
        mVolumeFileDesc.numVoxelsZ = numVoxelsZ;
    }

    // voxel type
    if ( mVolumeFileDescriptionDialog->findChild< QRadioButton* >( "unsignedChar" )->isChecked() )
    {
        mVolumeFileDesc.isSigned         = false;
        mVolumeFileDesc.numBytesPerVoxel = 1;
    }

    if ( mVolumeFileDescriptionDialog->findChild< QRadioButton* >( "signedChar" )->isChecked() )
    {
        mVolumeFileDesc.isSigned         = true;
        mVolumeFileDesc.numBytesPerVoxel = 1;
    }

    if ( mVolumeFileDescriptionDialog->findChild< QRadioButton* >( "unsignedShort" )->isChecked() )
    {
        mVolumeFileDesc.isSigned         = false;
        mVolumeFileDesc.numBytesPerVoxel = 2;
    }

    if ( mVolumeFileDescriptionDialog->findChild< QRadioButton* >( "signedShort" )->isChecked() )
    {
        mVolumeFileDesc.isSigned         = true;
        mVolumeFileDesc.numBytesPerVoxel = 2;
    }

    if ( mVolumeFileDescriptionDialog->findChild< QRadioButton* >( "unsignedInt" )->isChecked() )
    {
        mVolumeFileDesc.isSigned         = false;
        mVolumeFileDesc.numBytesPerVoxel = 4;
    }

    if ( mVolumeFileDescriptionDialog->findChild< QRadioButton* >( "signedInt" )->isChecked() )
    {
        mVolumeFileDesc.isSigned         = true;
        mVolumeFileDesc.numBytesPerVoxel = 4;
    }

    loadVolume( mVolumeFileDesc, mVolumeFileDescriptionDialog->findChild< QLineEdit* >( "parameterFileName" )->text() );

    mVolumeFileDescriptionDialog->findChild< QLineEdit* >( "parameterFileName" )->setText( "" );
}

void MainWindow::openProject( bool )
{
    QString fileName = QFileDialog::getOpenFileName( this, "Open GPUSeg Project", "projects", "GPUSeg Project Files (*.gproj)" );

    if ( fileName != "" )
    {
        openProject( fileName );
    }
}

void MainWindow::openDirectory( bool )
{
    QString fileName = QFileDialog::getExistingDirectory( this, "Select the directory containing the DICOM image stack", mVolumeFileDesc.fileName.ToQString() );

    if ( fileName != "" )
    {
        mNextFileName = fileName;
        mVolumeDirectoryDescriptionDialog->show();
    }
}

void MainWindow::openFile( bool )
{
    QString fileName = QFileDialog::getOpenFileName( this, "Open File", mVolumeFileDesc.fileName.ToQString(), "Raw Data File (*.*)" );

    if ( fileName != "" )
    {
        mNextFileName = fileName;
        mVolumeFileDescriptionDialog->show();
    }
}

bool MainWindow::saveParametersAs( bool )
{
    boost::filesystem::path filePath( mVolumeFileDesc.fileName.ToStdString() );

    QString leaf              = filePath.leaf().c_str();
    QString suggestedFileName = "parameters/" + leaf + ".par";
    QString fileName          = QFileDialog::getSaveFileName(this, "Save Parameters As", suggestedFileName, "Parameter XML Files (*.par)" );

    if ( fileName != NULL )
    {
        if ( !fileName.endsWith( ".par", Qt::CaseInsensitive ) )
        {
            fileName.append( ".par" );
        }

        mEngine->SaveParametersAs( fileName );
        return true;
    }
    else
    {
        return false;
    }
}

bool MainWindow::saveProjectAs( bool )
{
    core::String currentParametersFile = mEngine->GetCurrentParametersFileName();

    if ( currentParametersFile == "" )
    {
        if ( !saveParametersAs( true ) )
        {
            return false;
        }
    }

    currentParametersFile = mEngine->GetCurrentParametersFileName();
    Assert( currentParametersFile != "" );

    QString projectFileName = QFileDialog::getSaveFileName( this, "Save Project As", "projects", "GPUSeg Project Files (*.gproj)" );

    if ( projectFileName != "" )
    {
        if ( !projectFileName.endsWith( ".gproj", Qt::CaseInsensitive ) )
        {
            projectFileName.append( ".gproj" );
        }

        QDomDocument projectDomDocument;

        QDomElement rootElement    = projectDomDocument.createElement( "Root" );
        QDomElement projectElement = projectDomDocument.createElement( "GPUSegProject" );

        QDir current = QDir::current();

        QString relativeDataPath       = current.relativeFilePath( mVolumeFileDesc.fileName.ToQString() );
        QString relativeParametersPath = current.relativeFilePath( currentParametersFile.ToQString() );

        projectElement.setAttribute( "dataFile",         relativeDataPath );
        projectElement.setAttribute( "isSigned",         mVolumeFileDesc.isSigned );
        projectElement.setAttribute( "numBytesPerVoxel", mVolumeFileDesc.numBytesPerVoxel );
        projectElement.setAttribute( "numVoxelsX",       mVolumeFileDesc.numVoxelsX );
        projectElement.setAttribute( "numVoxelsY",       mVolumeFileDesc.numVoxelsY );
        projectElement.setAttribute( "numVoxelsZ",       mVolumeFileDesc.numVoxelsZ );
        projectElement.setAttribute( "upDirectionX",     mVolumeFileDesc.upDirection[ math::X ] );
        projectElement.setAttribute( "upDirectionY",     mVolumeFileDesc.upDirection[ math::Y ] );
        projectElement.setAttribute( "upDirectionZ",     mVolumeFileDesc.upDirection[ math::Z ] );
        projectElement.setAttribute( "zAnisotropy",      mVolumeFileDesc.zAnisotropy );
        projectElement.setAttribute( "parametersFile",   relativeParametersPath );

        rootElement.appendChild( projectElement );
        projectDomDocument.appendChild( rootElement );

        QFile projectFile( projectFileName );
        projectFile.open( QFile::WriteOnly );
        projectFile.write( projectDomDocument.toString( 4 ).toAscii() );

        projectFile.close();

        return true;
    }
    else
    {
        return false;
    }
}

bool MainWindow::saveSegmentationAs( bool )
{
    QString fileName = QFileDialog::getExistingDirectory( this, "Select the directory to save the segmentation as a DICOM stack", mVolumeFileDesc.fileName.ToQString() );

    if ( fileName != "" )
    {
        mEngine->SaveSegmentationAs( fileName );
        return true;
    }

    return false;
}

void MainWindow::mayaCameraTool( bool )
{
    selectTool( Tool_MayaCamera );
}

void MainWindow::sketchSeedTool( bool )
{
    selectTool( Tool_SketchSeed );
}

void MainWindow::clearCurrentSegmentation( bool )
{
    mEngine->ClearCurrentSegmentation();
}

void MainWindow::freezeCurrentSegmentation( bool )
{
    mEngine->FreezeCurrentSegmentation();
}

void MainWindow::clearAllSegmentations( bool )
{
    mEngine->ClearAllSegmentations();
}

void MainWindow::finishedSegmentationSession( bool )
{
    mEngine->FinishedSegmentationSession();
}

void MainWindow::play( bool )
{
    mEngine->PlaySegmentation();
}

void MainWindow::stop( bool )
{
    mEngine->StopSegmentation();
}

void MainWindow::loadVolume( const VolumeFileDesc& volumeFileDesc, const QString& parameterFileName )
{
    mEngine->LoadVolume( volumeFileDesc, parameterFileName );

    foreach( QAction* action, mVolumeLoadedActions )
    {
        action->setEnabled( true );
    }

    QFileInfo fileInfo( volumeFileDesc.fileName.ToQString() );

    if( parameterFileName == "" )
    {
        setLockParameters( false );
    }
    else
    {
        setLockParameters( true );
    }

#ifdef ANONYMOUS_STATUS_BAR
    mStatusBarLabel->setText( fileInfo.filePath() );
#else
    mStatusBarLabel->setText( fileInfo.absoluteFilePath() );
#endif

    mStatusBarLabel->show();
}

void MainWindow::openProject( const QString& projectFileName )
{
    QDomDocument projectDomDocument;

    QFile projectFile( projectFileName );
    projectFile.open( QFile::ReadOnly );

    projectDomDocument.setContent( &projectFile );

    projectFile.close();

    QDomElement rootElement = projectDomDocument.firstChildElement();
    Assert( rootElement.tagName() == "Root" );

    QDomElement projectElement = rootElement.firstChildElement();
    Assert( projectElement.tagName() == "GPUSegProject" );

    VolumeFileDesc volumeFileDesc;

    volumeFileDesc.fileName         = projectElement.attribute( "dataFile" );
    volumeFileDesc.isSigned         = projectElement.attribute( "isSigned" ).toInt() != 0;
    volumeFileDesc.numBytesPerVoxel = projectElement.attribute( "numBytesPerVoxel" ).toInt();
    volumeFileDesc.numVoxelsX       = projectElement.attribute( "numVoxelsX" ).toInt();
    volumeFileDesc.numVoxelsY       = projectElement.attribute( "numVoxelsY" ).toInt();
    volumeFileDesc.numVoxelsZ       = projectElement.attribute( "numVoxelsZ" ).toInt();
    volumeFileDesc.upDirection      = math::Vector3(
        projectElement.attribute( "upDirectionX" ).toInt(),
        projectElement.attribute( "upDirectionY" ).toInt(),
        projectElement.attribute( "upDirectionZ" ).toInt() );

    volumeFileDesc.zAnisotropy = projectElement.attribute( "zAnisotropy" ).toFloat();

    loadVolume( volumeFileDesc, projectElement.attribute( "parametersFile" ) );
}

void MainWindow::closeEvent( QCloseEvent* closeEvent )
{
    int response = QMessageBox::question( this, "GPUSeg", "Are you sure you want to exit?", QMessageBox::Yes, QMessageBox::No );

    if ( response == QMessageBox::Yes )
    {
        closeEvent->accept();
    }
    else
    {
        closeEvent->ignore();
    }
}


void MainWindow::resizeEvent( QResizeEvent* resizeEvent )
{
    if ( mEngine != NULL )
    {
        sUpdatingEngineMutex.lock();

        mEngine->Update();

        sUpdatingEngineMutex.unlock();
    }
}

void MainWindow::keyPressEvent( QKeyEvent* keyEvent )
{
    keyEvent->accept();

    renderingWindowkeyPressEvent( keyEvent );
}

void MainWindow::keyReleaseEvent( QKeyEvent* keyEvent )
{
    keyEvent->accept();

    renderingWindowkeyReleaseEvent( keyEvent );
}

void MainWindow::loadParameters()
{
    mParameterManagerDelegate->setInitializing( true );
    mParameterManagerDelegate->clearSliderList();
    mParameterManagerDelegate->clearSpinBoxList();

    //
    // initialize data model
    //
    mStandardItemModel->clear();
    QStandardItem* root      = mStandardItemModel->invisibleRootItem();
    QModelIndex    rootIndex = root->index();

    //
    // fill in header data
    //
    root->setColumnCount( 3 );

    mStandardItemModel->setHeaderData( 0, Qt::Horizontal, "Parameter Name" );
    mStandardItemModel->setHeaderData( 1, Qt::Horizontal, "Current Value" );
    mStandardItemModel->setHeaderData( 2, Qt::Horizontal, "" );

    //
    // hold onto the relevent system indices
    //
    QList< QModelIndex > expandSystemIndices;

    //
    // now iterate over the engine parameters
    //
    container::List< core::String >::ConstIterator systemIteratorBegin = content::ParameterManager::GetSystemsBegin();
    container::List< core::String >::ConstIterator systemIteratorEnd   = content::ParameterManager::GetSystemsEnd();

    for ( container::List< core::String >::ConstIterator i = systemIteratorBegin; i != systemIteratorEnd; i++ )
    {
        core::String systemName        = *i;
        QString      systemNameQString = systemName.ToQString();

        if ( systemName == "materials" )
        {
            continue;
        }

        //
        // create a top-level node for the system
        //
        ModelViewDataDesc modelViewDataDesc;

        modelViewDataDesc.dataName      = systemNameQString;
        modelViewDataDesc.parameterName = systemNameQString;
        modelViewDataDesc.systemName    = systemNameQString;
        modelViewDataDesc.parameterType = content::ParameterType_Null;
        modelViewDataDesc.value         = "";
        modelViewDataDesc.minimumValue  = -98765.4321f;
        modelViewDataDesc.maximumValue  = -98765.4321f;
        modelViewDataDesc.parentIndex   = rootIndex;
        modelViewDataDesc.usage         = ModelViewDataUsage_Static;

        QModelIndex systemIndex = addDataToModelAndView( modelViewDataDesc );

        if ( systemName == "GPUSegRenderStrategy" || systemName == "Segmenter" )
        {
            expandSystemIndices.append( systemIndex );
        }

        //
        // iterate over the system's parameters
        //
        container::List< core::String >::ConstIterator parameterIteratorBegin = content::ParameterManager::GetParametersBegin( systemName );
        container::List< core::String >::ConstIterator parameterIteratorEnd   = content::ParameterManager::GetParametersEnd( systemName );

        for ( container::List< core::String >::ConstIterator j = parameterIteratorBegin; j != parameterIteratorEnd; j++ )
        {
            core::String parameterName        = *j;
            QString      parameterNameQString = parameterName.ToQString();

            content::ParameterType type = content::ParameterManager::GetParameterType( systemName, parameterName );
            float minValue              = content::ParameterManager::GetMinimumValue(  systemName, parameterName );
            float maxValue              = content::ParameterManager::GetMaximumValue(  systemName, parameterName );

            QModelIndex vector3ModelIndex;
            QModelIndex vector4ModelIndex;
            
            switch ( type )
            {
            case content::ParameterType_Float:

                modelViewDataDesc.dataName      = parameterNameQString;
                modelViewDataDesc.parameterName = parameterNameQString;
                modelViewDataDesc.systemName    = systemNameQString;
                modelViewDataDesc.parameterType = type;
                modelViewDataDesc.value         = content::ParameterManager::GetParameter< float >( systemName, parameterName );
                modelViewDataDesc.minimumValue  = minValue;
                modelViewDataDesc.maximumValue  = maxValue;
                modelViewDataDesc.parentIndex   = systemIndex;
                modelViewDataDesc.usage         = ModelViewDataUsage_Editable;
                                                  
                addDataToModelAndView( modelViewDataDesc );

                break;

            case content::ParameterType_Vector3:

                modelViewDataDesc.dataName      = parameterNameQString;
                modelViewDataDesc.parameterName = parameterNameQString;
                modelViewDataDesc.systemName    = systemNameQString;
                modelViewDataDesc.parameterType = content::ParameterType_Null;
                modelViewDataDesc.value         = "[ ... ]";
                modelViewDataDesc.minimumValue  = -98765.4321f;
                modelViewDataDesc.maximumValue  = -98765.4321f;
                modelViewDataDesc.parentIndex   = systemIndex;
                modelViewDataDesc.usage         = ModelViewDataUsage_Static;

                vector3ModelIndex = addDataToModelAndView( modelViewDataDesc );

                modelViewDataDesc.dataName      = "x";
                modelViewDataDesc.parameterName = parameterNameQString;
                modelViewDataDesc.systemName    = systemNameQString;
                modelViewDataDesc.parameterType = type;
                modelViewDataDesc.value         = content::ParameterManager::GetParameter< math::Vector3 >( systemName, parameterName )[ math::X ];
                modelViewDataDesc.minimumValue  = minValue;
                modelViewDataDesc.maximumValue  = maxValue;
                modelViewDataDesc.parentIndex   = vector3ModelIndex;
                modelViewDataDesc.usage         = ModelViewDataUsage_Editable;

                addDataToModelAndView( modelViewDataDesc );

                modelViewDataDesc.dataName      = "y";
                modelViewDataDesc.value         = content::ParameterManager::GetParameter< math::Vector3 >( systemName, parameterName )[ math::Y ];

                addDataToModelAndView( modelViewDataDesc );

                modelViewDataDesc.dataName      = "z";
                modelViewDataDesc.value         = content::ParameterManager::GetParameter< math::Vector3 >( systemName, parameterName )[ math::Z ];

                addDataToModelAndView( modelViewDataDesc );

                break;

            case content::ParameterType_Vector4:
                modelViewDataDesc.dataName      = parameterNameQString;
                modelViewDataDesc.parameterName = parameterNameQString;
                modelViewDataDesc.systemName    = systemNameQString;
                modelViewDataDesc.parameterType = content::ParameterType_Null;
                modelViewDataDesc.value         = "[ ... ]";
                modelViewDataDesc.minimumValue  = -98765.4321f;
                modelViewDataDesc.maximumValue  = -98765.4321f;
                modelViewDataDesc.parentIndex   = systemIndex;
                modelViewDataDesc.usage         = ModelViewDataUsage_Static;

                vector4ModelIndex = addDataToModelAndView( modelViewDataDesc );

                modelViewDataDesc.dataName      = "x";
                modelViewDataDesc.parameterName = parameterNameQString;
                modelViewDataDesc.systemName    = systemNameQString;
                modelViewDataDesc.parameterType = type;
                modelViewDataDesc.value         = content::ParameterManager::GetParameter< const math::Vector4 >( systemName, parameterName )[ math::X ];
                modelViewDataDesc.minimumValue  = minValue;
                modelViewDataDesc.maximumValue  = maxValue;
                modelViewDataDesc.parentIndex   = vector4ModelIndex;
                modelViewDataDesc.usage         = ModelViewDataUsage_Editable;

                addDataToModelAndView( modelViewDataDesc );

                modelViewDataDesc.dataName      = "y";
                modelViewDataDesc.value         = content::ParameterManager::GetParameter< const math::Vector4 >( systemName, parameterName )[ math::Y ];

                addDataToModelAndView( modelViewDataDesc );

                modelViewDataDesc.dataName      = "z";
                modelViewDataDesc.value         = content::ParameterManager::GetParameter< const math::Vector4 >( systemName, parameterName )[ math::Z ];

                addDataToModelAndView( modelViewDataDesc );

                modelViewDataDesc.dataName      = "h";
                modelViewDataDesc.value         = content::ParameterManager::GetParameter< const math::Vector4 >( systemName, parameterName )[ math::H ];

                addDataToModelAndView( modelViewDataDesc );

                break;

            case content::ParameterType_Matrix44:
                Assert( 0 );
                break;

            default:
                Assert( 0 );
                break;
            }
        }
    }

    //
    // get the tree view object
    //
    QTreeView* treeView = mSettingsWindow->findChild< QTreeView* >( "treeView" );
    Assert( treeView != NULL );

    treeView->header()->resizeSections( QHeaderView::ResizeToContents );
    treeView->header()->resizeSection( 0, 255 );
    treeView->header()->setResizeMode( QHeaderView::Fixed );

    treeView->expandAll();
    treeView->collapseAll();

    foreach ( QModelIndex expandSystemIndex, expandSystemIndices )
    {
        treeView->expand( expandSystemIndex );
    }

    //
    // inform the delegate that we're done adding data to the view,
    // so the delegate can activate the editable parameters
    //
    mParameterManagerDelegate->setInitializing( false );

    //
    // now that we're done initializing the data, we can send the settings file data to the engine.
    //
    mParameterManagerDelegate->updateModel();
}

QModelIndex MainWindow::addDataToModelAndView( ModelViewDataDesc modelViewDataDesc )
{
    //
    // get the tree view object
    //
    QTreeView* treeView = mSettingsWindow->findChild< QTreeView* >( "treeView" );
    Assert( treeView != NULL );


    //
    // create item
    //
    QStandardItem* parentItem = mStandardItemModel->itemFromIndex( modelViewDataDesc.parentIndex );
    QStandardItem* item       = new QStandardItem();


    //
    // allocate space for our new property in the model
    //
    item->setColumnCount( 3 );

    int currentParentRow;

    if ( parentItem == NULL )
    {
        mStandardItemModel->appendRow( item );
        currentParentRow = mStandardItemModel->rowCount() - 1;
    }
    else
    {
        parentItem->appendRow( item );
        currentParentRow = parentItem->rowCount() - 1;
    }


    //
    // get a handle to the new model data and set values for the data.
    //
    QModelIndex itemKeyIndex    = mStandardItemModel->index( currentParentRow, 0, modelViewDataDesc.parentIndex );
    QModelIndex itemValueIndex  = mStandardItemModel->index( currentParentRow, 1, modelViewDataDesc.parentIndex );
    QModelIndex itemSliderIndex = mStandardItemModel->index( currentParentRow, 2, modelViewDataDesc.parentIndex );

    mStandardItemModel->setData( itemKeyIndex, modelViewDataDesc.dataName,        DATA_NAME_ROLE      );
    mStandardItemModel->setData( itemKeyIndex, modelViewDataDesc.parameterName,   PARAMETER_NAME_ROLE );
    mStandardItemModel->setData( itemKeyIndex, modelViewDataDesc.parameterType,   PARAMETER_TYPE_ROLE );
    mStandardItemModel->setData( itemKeyIndex, modelViewDataDesc.systemName,      SYSTEM_NAME_ROLE    );

    mStandardItemModel->setData( itemValueIndex, modelViewDataDesc.value,         DATA_VALUE_ROLE     );
    mStandardItemModel->setData( itemValueIndex, modelViewDataDesc.parameterName, PARAMETER_NAME_ROLE );
    mStandardItemModel->setData( itemValueIndex, modelViewDataDesc.parameterType, PARAMETER_TYPE_ROLE );
    mStandardItemModel->setData( itemValueIndex, modelViewDataDesc.systemName,    SYSTEM_NAME_ROLE    );

    mStandardItemModel->setData( itemSliderIndex, modelViewDataDesc.value,         DATA_VALUE_ROLE     );
    mStandardItemModel->setData( itemSliderIndex, modelViewDataDesc.parameterName, PARAMETER_NAME_ROLE );
    mStandardItemModel->setData( itemSliderIndex, modelViewDataDesc.parameterType, PARAMETER_TYPE_ROLE );
    mStandardItemModel->setData( itemSliderIndex, modelViewDataDesc.systemName,    SYSTEM_NAME_ROLE    );


    if ( modelViewDataDesc.usage == ModelViewDataUsage_Editable )
    {
        mParameterManagerDelegate->setParameterRange( modelViewDataDesc.minimumValue, modelViewDataDesc.maximumValue );

        //
        // create custom spin box for editable data
        //
        treeView->openPersistentEditor( itemValueIndex );

        //
        // create custom slider for editable data
        //
        treeView->openPersistentEditor( itemSliderIndex );
    }

    return itemKeyIndex;
}

void MainWindow::selectTool( Tool tool )
{
    foreach ( QAction* action, mToolActionMap )
    {
        action->setChecked( false );
    }
    
    mToolActionMap.value( tool )->setChecked( true );

    AssignRef( mCurrentUserInputHandler, mToolUserInputHandlerMap.value( tool ) );
}

void MainWindow::initializeRenderingWindow()
{
    mRenderingWindow = new RenderingWindow( this );
}

void MainWindow::initializeEngine()
{
    WINDOW_HANDLE windowHandle = (WINDOW_HANDLE)mRenderingWindow->winId();

    mEngine = new Engine( windowHandle );
    mEngine->AddRef();

    mEngine->LoadScript( "defaults/GPUSeg.py" );

    mEngine->SetAssetChangedAfterReloadCallback( new core::Functor< MainWindow >( this, &MainWindow::assetChangedAfterReloadCallback ) );
    mEngine->SetSegmentationStartedCallback( new core::Functor< MainWindow >( this, &MainWindow::segmentationStartedCallback ) );
    mEngine->SetSegmentationFinishedCallback( new core::Functor< MainWindow >( this, &MainWindow::segmentationFinishedCallback ) );
    mEngine->SetParametersChangedCallback( new core::Functor< MainWindow >( this, &MainWindow::parametersChangedCallback ) );
}

void MainWindow::initializeParameterManagerWindow()
{
    //
    // create a data model
    //
    mStandardItemModel = new QStandardItemModel( 0, 0, this );


    //
    // load the ui file
    //
    ReleaseAssert( QFile::exists( "ui/ParameterManagerWindow.ui" ) );

    QFile settingsWindowFormFile( "ui/ParameterManagerWindow.ui" );
    QUiLoader uiLoader;

    settingsWindowFormFile.open( QFile::ReadOnly );

    mSettingsWindowForm = qobject_cast< QWidget* >( uiLoader.load( &settingsWindowFormFile ) );
    Assert( mSettingsWindowForm != NULL );

    settingsWindowFormFile.close();


    //
    // find the settings window and initialize the view
    //
    mSettingsWindow = mSettingsWindowForm->findChild< QDockWidget* >( "dockWidget" );
    Assert( mSettingsWindow != NULL );

    QTreeView* treeView = mSettingsWindow->findChild< QTreeView* >( "treeView" );
    Assert( treeView != NULL );

    mSettingsWindow->setMinimumWidth( 650 );

    //
    // wire the model to the view
    //
    treeView->setModel( mStandardItemModel );

    mParameterManagerDelegate = new ParameterManagerDelegate( this, mStandardItemModel, treeView, this );
    treeView->setItemDelegate( mParameterManagerDelegate );

    treeView->header()->setStretchLastSection( false );
    treeView->header()->setMovable( false );
    treeView->setFocusPolicy( Qt::NoFocus );
}


void MainWindow::initializeMainWindow()
{
    move( 20, 20 );
    resize( 800, 600 );

    setCentralWidget( mRenderingWindow );
    addDockWidget( Qt::RightDockWidgetArea, mSettingsWindow );

    //
    // load the icon
    //
    QIcon icon( "ui/GPUSeg.png" );
    setWindowIcon( icon );

    //
    // window title
    //
    QString windowTitle = "GPUSeg";

#if defined(BUILD_DEBUG)
    #if defined(BUILD_EMULATE)
        windowTitle += " - Emulate";
    #else
        windowTitle += " - Debug";
    #endif
#elif defined(BUILD_RELEASE)
    windowTitle += " - Release";
#else
    #error( "Build configuration not defined...BUILD_DEBUG, BUILD_EMULATE or BUILD_RELEASE must be defined" )
#endif

#if defined(CUDA_ARCH_SM_10)
    windowTitle += " - SM_10";
#elif defined(CUDA_ARCH_SM_13)
    windowTitle += " - SM_13";
#else
    #error( "CUDA Architecture not defined...CUDA_ARCH_SM_10 or CUDA_ARCH_SM_13 must be defined" )
#endif

    setWindowTitle( windowTitle );
    setWindowIconText( windowTitle );

    //
    // create a menu bar and tool bar by loading them from a file
    //
    ReleaseAssert( QFile::exists( "ui/MainWindow.ui" ) );

    QFile mainWindowFormFile( "ui/MainWindow.ui" );
    QUiLoader uiLoader;

    mainWindowFormFile.open( QFile::ReadOnly );

    mMainWindowForm = qobject_cast< QMainWindow* >( uiLoader.load( &mainWindowFormFile ) );
    Assert( mMainWindowForm != NULL );

    mainWindowFormFile.close();

    mMenuBar = mMainWindowForm->findChild< QMenuBar* >( "menuBar" );
    Assert( mMenuBar != NULL );

    setMenuBar( mMenuBar );

    mToolBar = mMainWindowForm->findChild< QToolBar* >( "toolBar" );
    Assert( mToolBar != NULL );

    mToolBar->clear();
    addToolBar( Qt::TopToolBarArea, mToolBar );

    mStatusBar = mMainWindowForm->findChild< QStatusBar* >( "statusBar" );
    Assert( mStatusBar != NULL );

    mStatusBarLabel = new QLabel( "No Image Loaded", this );
    mStatusBar->addWidget( mStatusBarLabel, 1 );

    mProgressBar = new QProgressBar( this );
    mProgressBar->setMinimum( 0 );
    mProgressBar->setMaximum( 1 );
    mProgressBar->setMaximumHeight( 10 );
    mProgressBar->setMaximumWidth( 650 );
    mProgressBar->setTextVisible( false );
    mStatusBar->addWidget( mProgressBar, 1 );

    setStatusBar( mStatusBar );
}

void MainWindow::initializeVolumeDescriptionDialogs()
{
    //
    // load directory dialog
    //
    ReleaseAssert( QFile::exists( "ui/VolumeDirectoryDesc.ui" ) );

    QFile        volumeDirectoryDescDialogFile( "ui/VolumeDirectoryDesc.ui" );
    QIcon        icon( "ui/GPUSeg.png" );
    QUiLoader    uiLoader;
    bool         connected                 = false;
    QPushButton* parameterFileSelectButton = NULL;

    volumeDirectoryDescDialogFile.open( QFile::ReadOnly );

    mVolumeDirectoryDescriptionDialog = qobject_cast< QDialog* >( uiLoader.load( &volumeDirectoryDescDialogFile ) );
    Assert( mVolumeDirectoryDescriptionDialog != NULL );

    mVolumeDirectoryDescriptionDialog->setWindowIcon( icon );
    mVolumeDirectoryDescriptionDialog->setParent( this, Qt::Dialog );

    connected = connect( mVolumeDirectoryDescriptionDialog, SIGNAL( accepted( void ) ), this, SLOT( volumeDirectoryDescriptionOK( void ) ) );
    Assert( connected );

    parameterFileSelectButton = mVolumeDirectoryDescriptionDialog->findChild< QPushButton* >( "parameterFileSelect" );
    Assert( parameterFileSelectButton != NULL );

    connected = connect( parameterFileSelectButton, SIGNAL( clicked( void ) ), this, SLOT( volumeDirectoryDescriptionParameterFileSelect( void ) ) );
    Assert( connected );

    //
    // load file dialog
    //
    ReleaseAssert( QFile::exists( "ui/VolumeFileDesc.ui" ) );

    QFile volumeFileDescDialogFile( "ui/VolumeFileDesc.ui" );

    volumeFileDescDialogFile.open( QFile::ReadOnly );

    mVolumeFileDescriptionDialog = qobject_cast< QDialog* >( uiLoader.load( &volumeFileDescDialogFile ) );
    Assert( mVolumeFileDescriptionDialog != NULL );


    mVolumeFileDescriptionDialog->setWindowIcon( icon );
    mVolumeFileDescriptionDialog->setParent( this, Qt::Dialog );

    connected = connect( mVolumeFileDescriptionDialog, SIGNAL( accepted( void ) ), this, SLOT( volumeFileDescriptionOK( void ) ) );
    Assert( connected );

    parameterFileSelectButton = mVolumeFileDescriptionDialog->findChild< QPushButton* >( "parameterFileSelect" );
    Assert( parameterFileSelectButton != NULL );

    connected = connect( parameterFileSelectButton, SIGNAL( clicked( void ) ), this, SLOT( volumeFileDescriptionParameterFileSelect( void ) ) );
    Assert( connected );
}

void MainWindow::initializeActions()
{
    //
    // create a menu bar by loading it from a file
    //
    ReleaseAssert( QFile::exists( "ui/MainWindow.ui" ) );

    QFile     mainWindowFormFile( "ui/MainWindow.ui" );
    QUiLoader uiLoader;
    QAction*  action;
    bool      connected;
    QIcon     icon;

    mainWindowFormFile.open( QFile::ReadOnly );

    QMainWindow* mainWindowForm = qobject_cast< QMainWindow* >( uiLoader.load( &mainWindowFormFile ) );
    Assert( mainWindowForm != NULL );

    //
    // find menus
    //
    QMenu* fileMenu = mMenuBar->findChild< QMenu* >( "menuFile" );
    Assert( fileMenu != NULL );
    fileMenu->clear();

    QMenu* interactMenu = mMenuBar->findChild< QMenu* >( "menuInteract" );
    Assert( interactMenu != NULL );
    interactMenu->clear();

    QMenu* simulationMenu = mMenuBar->findChild< QMenu* >( "menuSimulation" );
    Assert( simulationMenu != NULL );
    simulationMenu->clear();

    //
    // Open Directory
    //
    action = mainWindowForm->findChild< QAction* >( "actionOpenProject" );
    Assert( action != NULL );

    icon = QIcon( "ui/OpenProject.png" );
    action->setIcon( icon );

    mToolBar->addAction( action );
    fileMenu->addAction( action );
    connected = connect( action, SIGNAL( triggered( bool ) ), this, SLOT( openProject( bool ) ) );
    Assert( connected );

    //
    // Open Directory
    //
    action = mainWindowForm->findChild< QAction* >( "actionOpenDirectory" );
    Assert( action != NULL );

    icon = QIcon( "ui/OpenDirectory.png" );
    action->setIcon( icon );

    mToolBar->addAction( action );
    fileMenu->addAction( action );
    connected = connect( action, SIGNAL( triggered( bool ) ), this, SLOT( openDirectory( bool ) ) );
    Assert( connected );

    //
    // Open File
    //
    action = mainWindowForm->findChild< QAction* >( "actionOpenFile" );
    Assert( action != NULL );

    icon = QIcon( "ui/OpenFile.png" );
    action->setIcon( icon );

    mToolBar->addAction( action );
    fileMenu->addAction( action );
    connected = connect( action, SIGNAL( triggered( bool ) ), this, SLOT( openFile( bool ) ) );
    Assert( connected );

    //
    // -----------
    //
    fileMenu->addSeparator();
    mToolBar->addSeparator();

    //
    // Save Parameters As
    //
    action = mainWindowForm->findChild< QAction* >( "actionSaveParametersAs" );
    Assert( action != NULL );

    mVolumeLoadedActions.append( action );

    icon = QIcon( "ui/SaveParametersAs.png" );
    action->setIcon( icon );

    mToolBar->addAction( action );
    fileMenu->addAction( action );
    connected = connect( action, SIGNAL( triggered( bool ) ), this, SLOT( saveParametersAs( bool ) ) );
    Assert( connected );

    //
    // Save Project As
    //
    action = mainWindowForm->findChild< QAction* >( "actionSaveProjectAs" );
    Assert( action != NULL );

    mVolumeLoadedActions.append( action );

    icon = QIcon( "ui/SaveProjectAs.png" );
    action->setIcon( icon );

    mToolBar->addAction( action );
    fileMenu->addAction( action );
    connected = connect( action, SIGNAL( triggered( bool ) ), this, SLOT( saveProjectAs( bool ) ) );
    Assert( connected );

    //
    // Save Segmentation As
    //
    action = mainWindowForm->findChild< QAction* >( "actionSaveSegmentationAs" );
    Assert( action != NULL );

    mVolumeLoadedActions.append( action );

    icon = QIcon( "ui/SaveSegmentationAs.png" );
    action->setIcon( icon );

    mToolBar->addAction( action );
    fileMenu->addAction( action );
    connected = connect( action, SIGNAL( triggered( bool ) ), this, SLOT( saveSegmentationAs( bool ) ) );
    Assert( connected );

    //
    // -----------
    //
    fileMenu->addSeparator();

    //
    // Exit
    //
    action = mainWindowForm->findChild< QAction* >( "actionExit" );

    icon = QIcon( "ui/Exit.png" );
    action->setIcon( icon );

    Assert( action != NULL );
    fileMenu->addAction( action );
    connected = connect( action, SIGNAL( triggered( bool ) ), this, SLOT( exitApplication( bool ) ) );
    Assert( connected );

    //
    // -----------
    //
    mToolBar->addSeparator();

    //
    // Maya Camera Tool
    //
    action = mainWindowForm->findChild< QAction* >( "actionMayaCameraTool" );
    Assert( action != NULL );

    mVolumeLoadedActions.append( action );

    icon = QIcon( "ui/MayaCameraTool.png" );
    action->setIcon( icon );

    mToolBar->addAction( action );
    interactMenu->addAction( action );
    mToolActionMap.insert( Tool_MayaCamera, action );

    connected = connect( action, SIGNAL( triggered( bool ) ), this, SLOT( mayaCameraTool( bool ) ) );
    Assert( connected );

    //
    // Sketch Seed Tool
    //
    action = mainWindowForm->findChild< QAction* >( "actionSketchSeedTool" );
    Assert( action != NULL );

    mVolumeLoadedActions.append( action );

    icon = QIcon( "ui/SketchSeedTool.png" );
    action->setIcon( icon );

    mToolBar->addAction( action );
    interactMenu->addAction( action );
    mToolActionMap.insert( Tool_SketchSeed, action );

    connected = connect( action, SIGNAL( triggered( bool ) ), this, SLOT( sketchSeedTool( bool ) ) );
    Assert( connected );

    //
    // -----------
    //
    mToolBar->addSeparator();
    interactMenu->addSeparator();

    //
    // Clear Current Segmentation
    //
    action = mainWindowForm->findChild< QAction* >( "actionClearCurrentSegmentation" );
    Assert( action != NULL );

    mVolumeLoadedActions.append( action );

    icon = QIcon( "ui/ClearCurrentSegmentation.png" );
    action->setIcon( icon );

    mToolBar->addAction( action );
    interactMenu->addAction( action );

    connected = connect( action, SIGNAL( triggered( bool ) ), this, SLOT( clearCurrentSegmentation( bool ) ) );
    Assert( connected );

    //
    // Freeze Current Segmentation
    //
    action = mainWindowForm->findChild< QAction* >( "actionFreezeCurrentSegmentation" );
    Assert( action != NULL );

    mVolumeLoadedActions.append( action );

    icon = QIcon( "ui/FreezeCurrentSegmentation.png" );
    action->setIcon( icon );

    mToolBar->addAction( action );
    interactMenu->addAction( action );

    connected = connect( action, SIGNAL( triggered( bool ) ), this, SLOT( freezeCurrentSegmentation( bool ) ) );
    Assert( connected );

    //
    // Clear All Segmentations
    //
    action = mainWindowForm->findChild< QAction* >( "actionClearAllSegmentations" );
    Assert( action != NULL );

    mVolumeLoadedActions.append( action );

    icon = QIcon( "ui/ClearAllSegmentations.png" );
    action->setIcon( icon );

    mToolBar->addAction( action );
    interactMenu->addAction( action );

    connected = connect( action, SIGNAL( triggered( bool ) ), this, SLOT( clearAllSegmentations( bool ) ) );
    Assert( connected );

    //
    // -----------
    //
    mToolBar->addSeparator();
    interactMenu->addSeparator();

    //
    // Lock parameters
    //
    action = mainWindowForm->findChild< QAction* >( "actionLockParameters" );
    Assert( action != NULL );

    icon = QIcon( "ui/LockParameters.png" );
    action->setIcon( icon );

    mVolumeLoadedActions.append( action );

    mToolBar->addAction( action );
    interactMenu->addAction( action );

    connected = connect( action, SIGNAL( triggered( bool ) ), this, SLOT( lockParameters( bool ) ) );
    Assert( connected );

    mLockParametersAction = action;

    //
    // -----------
    //
    mToolBar->addSeparator();
    interactMenu->addSeparator();

    //
    // Finished Segmentation
    //
    action = mainWindowForm->findChild< QAction* >( "actionFinishedSegmentationSession" );
    Assert( action != NULL );

    mVolumeLoadedActions.append( action );

    icon = QIcon( "ui/FinishedSegmentationSession.png" );
    action->setIcon( icon );

    mToolBar->addAction( action );
    interactMenu->addAction( action );

    connected = connect( action, SIGNAL( triggered( bool ) ), this, SLOT( finishedSegmentationSession( bool ) ) );
    Assert( connected );

    //
    // -----------
    //
    mToolBar->addSeparator();

    //
    // Play
    //
    action = mainWindowForm->findChild< QAction* >( "actionPlay" );
    Assert( action != NULL );

    mVolumeLoadedActions.append( action );

    icon = QIcon( "ui/Play.png" );
    action->setIcon( icon );

    mToolBar->addAction( action );
    simulationMenu->addAction( action );

    connected = connect( action, SIGNAL( triggered( bool ) ), this, SLOT( play( bool ) ) );
    Assert( connected );

    //
    // Stop
    //
    action = mainWindowForm->findChild< QAction* >( "actionStop" );
    Assert( action != NULL );

    mVolumeLoadedActions.append( action );

    icon = QIcon( "ui/Stop.png" );
    action->setIcon( icon );

    mToolBar->addAction( action );
    simulationMenu->addAction( action );

    connected = connect( action, SIGNAL( triggered( bool ) ), this, SLOT( stop( bool ) ) );
    Assert( connected );

    foreach( QAction* action, mVolumeLoadedActions )
    {
        action->setEnabled( false );
    }
}

void MainWindow::initializeTimer()
{
    //
    // set up a timer so we can get an event loop going
    //
    QTimer* timer = new QTimer( this );
    timer->setInterval( 0 );

    bool connected = connect( timer, SIGNAL( timeout() ), this, SLOT( update() ) );
    Assert( connected );

    timer->start();
}

void MainWindow::initializeUserInputHandlers()
{
    UserInputHandler* userInputHandler;

    userInputHandler = new MayaCameraUserInputHandler( mRenderingWindow, mEngine );
    userInputHandler->AddRef();
    mToolUserInputHandlerMap.insert( Tool_MayaCamera, userInputHandler );

    userInputHandler = new SketchUserInputHandler( mRenderingWindow, mEngine );
    userInputHandler->AddRef();
    mToolUserInputHandlerMap.insert( Tool_SketchSeed, userInputHandler );
}

void MainWindow::terminateEngine()
{
    mEngine->UnloadVolume();
    mEngine->UnloadScript();

    mEngine->Release();
    mEngine = NULL;
}

void MainWindow::terminateUserInputHandlers()
{
    AssignRef( mCurrentUserInputHandler, NULL );

    foreach ( UserInputHandler* userInputHandler, mToolUserInputHandlerMap )
    {
        userInputHandler->Release();
    }
}

void MainWindow::terminateMainWindow()
{
    delete mMainWindowForm;
    mMainWindowForm = NULL;
}

void MainWindow::terminateParameterManagerWindow()
{
    delete mSettingsWindowForm;
    mSettingsWindowForm = NULL;
}

void MainWindow::terminateVolumeDescriptionDialogs()
{
    // no-op.  volume description dialog belongs to main window
}

void MainWindow::terminateRenderingWindow()
{
    // no-op.  rendering window belongs to main window
}

void MainWindow::terminateTimer()
{
    // no-op.  timer belongs to main window
}

void MainWindow::terminateActions()
{
    // no-op.  actions belong to main window
}

void MainWindow::lockParameters( bool checked )
{
    setLockParameters( checked );
}

void MainWindow::setLockParameters( bool locked )
{
    if( locked )
    {
        mLockParametersAction->setChecked( true );
        mEngine->SetAutomaticParameterAdjustEnabled( false );
    }
    else
    {
        mLockParametersAction->setChecked( false );
        mEngine->SetAutomaticParameterAdjustEnabled( true );
    }
}