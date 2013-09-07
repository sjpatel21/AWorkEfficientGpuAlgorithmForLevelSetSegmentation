#ifndef GPUSEG_WIDGETS_MAIN_WINDOW_HPP
#define GPUSEG_WIDGETS_MAIN_WINDOW_HPP

#include <QtCore/QString>
#include <QtCore/QModelIndex>
#include <QtCore/QFile>
#include <QtCore/QSettings>
#include <QtCore/QMap>
#include <QtGui/QMainWindow>
#include <QtGui/QAction>
#include <QtGui/QProgressBar>
#include <QtUiTools/QUiLoader>

#include "content/ParameterType.hpp"

#include "VolumeFileDesc.hpp"

class QCloseEvent;
class QMoveEvent;
class QDockWidget;
class QStandardItemModel;
class RenderingWindow;
class ParameterManagerDelegate;
class QMenuBar;
class QToolBar;
class QStatusBar;
class QLabel;

class UserInputHandler;

class Engine;

class MainWindow : public QMainWindow
{
    Q_OBJECT

enum ModelViewDataUsage
{
    ModelViewDataUsage_Static,
    ModelViewDataUsage_Editable
};

enum Tool
{
    Tool_MayaCamera,
    Tool_SketchSeed
};

struct ModelViewDataDesc
{
    QString                dataName;
    QString                parameterName;
    QString                systemName;
    content::ParameterType parameterType;
    QVariant               value;
    float                  minimumValue;
    float                  maximumValue;
    QModelIndex            parentIndex;
    ModelViewDataUsage     usage;
};

public:
    MainWindow( QWidget* parent = NULL );
    ~MainWindow();

    void assetChangedAfterReloadCallback( void* );
    void segmentationStartedCallback( void* );
    void segmentationFinishedCallback( void* );

    void parametersChangedCallback( void* );

    void updateUserEditableParameter( QModelIndex index );

    void renderingWindowMousePressEvent  ( QMouseEvent*  mouseEvent );
    void renderingWindowMouseReleaseEvent( QMouseEvent*  mouseEvent );
    void renderingWindowMouseMoveEvent   ( QMouseEvent*  mouseEvent );
    void renderingWindowMouseWheelEvent  ( QWheelEvent*  wheelEvent );
    void renderingWindowResizeEvent      ( QResizeEvent* resizeEvent );
    void renderingWindowkeyPressEvent    ( QKeyEvent*    keyEvent );
    void renderingWindowkeyReleaseEvent  ( QKeyEvent*    keyEvent );

    void setLockParameters( bool locked );

public slots:
    virtual bool update();
    virtual void volumeDirectoryDescriptionParameterFileSelect();
    virtual void volumeFileDescriptionParameterFileSelect();
    virtual void volumeDirectoryDescriptionOK();
    virtual void volumeFileDescriptionOK();

    virtual void openProject( bool );
    virtual void openDirectory( bool );
    virtual void openFile( bool );

    virtual bool saveParametersAs( bool );
    virtual bool saveProjectAs( bool );
    virtual bool saveSegmentationAs( bool );

    virtual void exitApplication( bool exit );
    virtual void mayaCameraTool( bool );
    virtual void sketchSeedTool( bool );
    virtual void clearCurrentSegmentation( bool );
    virtual void freezeCurrentSegmentation( bool );
    virtual void clearAllSegmentations( bool );
    virtual void finishedSegmentationSession( bool );
    virtual void play( bool );
    virtual void stop( bool );

    virtual void lockParameters( bool );

protected:
    virtual void closeEvent     ( QCloseEvent*  closeEvent );
    virtual void resizeEvent    ( QResizeEvent* resizeEvent );
    virtual void keyPressEvent  ( QKeyEvent*    keyEvent );
    virtual void keyReleaseEvent( QKeyEvent*    keyEvent );

private:
    void openProject( const QString& projectFileName );
    void loadVolume( const VolumeFileDesc& volumeFileDesc, const QString& parameterFile );

    void initializeRenderingWindow();
    void initializeParameterManagerWindow();
    void initializeMainWindow();
    void initializeVolumeDescriptionDialogs();
    void initializeEngine();
    void initializeActions();
    void initializeTimer();
    void initializeUserInputHandlers();

    void terminateRenderingWindow();
    void terminateParameterManagerWindow();
    void terminateMainWindow();
    void terminateVolumeDescriptionDialogs();
    void terminateEngine();
    void terminateActions();
    void terminateTimer();
    void terminateUserInputHandlers();

    void loadParameters();

    QModelIndex addDataToModelAndView( ModelViewDataDesc modelViewDataDesc );

    void selectTool( Tool tool );

    VolumeFileDesc      mVolumeFileDesc;

    Engine*             mEngine;
    UserInputHandler*   mCurrentUserInputHandler;
    RenderingWindow*    mRenderingWindow;

    ParameterManagerDelegate*     mParameterManagerDelegate;
    QDockWidget*        mSettingsWindow;
    QDialog*            mVolumeFileDescriptionDialog;
    QDialog*            mVolumeDirectoryDescriptionDialog;
    QStandardItemModel* mStandardItemModel;
    QMenuBar*           mMenuBar;
    QToolBar*           mToolBar;
    QStatusBar*         mStatusBar;
    QLabel*             mStatusBarLabel;
    QWidget*            mSettingsWindowForm;
    QMainWindow*        mMainWindowForm;
    QProgressBar*       mProgressBar;
    QString             mNextFileName;
    QSettings           mSettings;
    QAction*            mLockParametersAction;

    QList< QAction* >               mVolumeLoadedActions;

    QMap< Tool, QAction* >          mToolActionMap;
    QMap< Tool, UserInputHandler* > mToolUserInputHandlerMap;
};

#endif
