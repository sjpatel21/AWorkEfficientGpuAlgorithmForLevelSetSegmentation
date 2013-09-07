#include "widgets/ParameterManagerDelegate.hpp"

#include <QtCore/QFile>
#include <QtCore/QVariant>
#include <QtGui/QTreeView>
#include <QtGui/QFrame>
#include <QtGui/QPainter>
#include <QtGui/QStyleOptionViewItem>
#include <QtGui/QStandardItemModel>
#include <QtUiTools/QUiLoader>

#include "core/Assert.hpp"

static const int KEY_COLUMN    = 0;
static const int VALUE_COLUMN  = 1;
static const int SLIDER_COLUMN = 2;

static const int NUM_DISPLAY_DIGITS = 7;

static const int ROW_HEIGHT_PIXELS                = 24;
static const int SLIDER_HEIGHT_PIXELS             = 20;
static const int SLIDER_WIDTH_PIXELS              = 250;
static const int SLIDER_VERTICAL_PADDING_PIXELS   = ( ROW_HEIGHT_PIXELS - SLIDER_HEIGHT_PIXELS ) / 2;
static const int SLIDER_HORIZONTAL_PADDING_PIXELS = 10;
static const int FONT                             = 5;
static const int TEXT_INDENTATION                 = 10;
static const int MINIMUM_ELEMENT_WIDTH            = 25;
static const int MAXIMUM_ELEMENT_WIDTH            = 80;

// this has to be static because the createEditor callback is declared const and therefore
// can't modify any member data in the slider delegate
QHash< QString, QList< Slider* > >    sSliderLists;
QHash< QString, QList< SpinBox* > >   sSpinBoxLists;

static const int DELEGATE_SYSTEM_NAME_ROLE    = Qt::UserRole + 2;

ParameterManagerDelegate::ParameterManagerDelegate(
    MainWindow*         mainWindow,
    QAbstractItemModel* model,
    QAbstractItemView*  view,
    QObject*            parent ) :
QItemDelegate             ( parent ),
mMainWindow               ( mainWindow ),
mModel                    ( model ),
mView                     ( view ),
mCurrentMinimumValue( -9999999 ),
mCurrentMaximumValue( -9999998 ),
mInitializing             ( false )
{
}

QWidget* ParameterManagerDelegate::createEditor( QWidget* parent, const QStyleOptionViewItem& option, const QModelIndex& index ) const
{
    QWidget* createdWidget = NULL;

    if( index.column() == SLIDER_COLUMN )
    {
        Slider* slider = new Slider(
            mMainWindow,
            mModel,
            mView,
            QPersistentModelIndex( index ),
            mCurrentMinimumValue,
            mCurrentMaximumValue,
            parent );

        slider->setMaximumWidth( SLIDER_WIDTH_PIXELS );
        slider->setMinimumWidth( SLIDER_WIDTH_PIXELS );

        slider->setMaximumHeight( SLIDER_HEIGHT_PIXELS );
        slider->setMinimumHeight( SLIDER_HEIGHT_PIXELS );

        slider->setMinimum( 0 );
        slider->setMaximum( SLIDER_WIDTH_PIXELS );

        QString systemName = index.data( DELEGATE_SYSTEM_NAME_ROLE ).toString();

        if( sSliderLists.constFind( systemName ) == sSliderLists.constEnd() )
            sSliderLists.insert( systemName, QList< Slider* >() );

        sSliderLists[ systemName ].append( slider );

        createdWidget = slider;
    }
    else if( index.column() == VALUE_COLUMN )
    {
        SpinBox* spinBox = new SpinBox(
            mMainWindow,
            mModel,
            mView,
            QPersistentModelIndex( index ),
            mCurrentMinimumValue,
            mCurrentMaximumValue,
            parent );

        spinBox->setMaximumWidth( MAXIMUM_ELEMENT_WIDTH );
        spinBox->setMinimumWidth( MINIMUM_ELEMENT_WIDTH );

        spinBox->setMaximumHeight( SLIDER_HEIGHT_PIXELS );
        spinBox->setMinimumHeight( SLIDER_HEIGHT_PIXELS );

        QString systemName = index.data( DELEGATE_SYSTEM_NAME_ROLE ).toString();

        if( sSpinBoxLists.constFind( systemName ) == sSpinBoxLists.constEnd() )
            sSpinBoxLists.insert( systemName, QList< SpinBox* >() );

        sSpinBoxLists[ systemName ].append( spinBox );


        createdWidget = spinBox;
    }

    return createdWidget;
}

void ParameterManagerDelegate::setEditorData( QWidget *editor, const QModelIndex& index ) const
{
    if( index.column() == SLIDER_COLUMN )
    {
        Slider* slider = qobject_cast< Slider* >( editor );

        float sliderMinimumValue, sliderMaximumValue;

        slider->getSliderRange( &sliderMinimumValue, &sliderMaximumValue );

        double modelValue = index.model()->data( index, Qt::DisplayRole ).toDouble();
        float sliderRange = sliderMaximumValue - sliderMinimumValue;
        float sliderValue = ( modelValue - sliderMinimumValue ) * slider->maximum() / sliderRange;

        slider->setValue( sliderValue );

        QString systemName = index.data( DELEGATE_SYSTEM_NAME_ROLE ).toString();

        // Notify the spinbox to change
        if( sSpinBoxLists.constFind( systemName ) != sSpinBoxLists.constEnd() &&
            sSpinBoxLists[ systemName ].size() > index.row() )
        {
            SpinBox* spinBox = sSpinBoxLists[ systemName ].at( index.row() );
            spinBox->setValue( modelValue );
        }
    }
    else if( index.column() == VALUE_COLUMN )
    {
        SpinBox* spinBox = qobject_cast< SpinBox* >( editor );

        double modelValue = index.model()->data( index, Qt::DisplayRole ).toDouble();

        spinBox->setValue( modelValue );
 
        QString systemName = index.data( DELEGATE_SYSTEM_NAME_ROLE ).toString();

        // Notify the slider to change
        if( sSliderLists.constFind( systemName ) != sSliderLists.constEnd() &&
            sSliderLists[ systemName ].size() > index.row() )
        {
            Slider* slider = sSliderLists[ systemName ].at( index.row() );

            float sliderMinimumValue, sliderMaximumValue;

            slider->getSliderRange( &sliderMinimumValue, &sliderMaximumValue );

            float sliderRange = sliderMaximumValue - sliderMinimumValue;
            float sliderValue = ( modelValue - sliderMinimumValue ) * slider->maximum() / sliderRange;

            slider->setValue( sliderValue );
        }
    }
}

void ParameterManagerDelegate::setModelData( QWidget* editor, QAbstractItemModel* model, const QModelIndex& index ) const
{
    //
    // updating the model is handled in the Slider class.  See Slider::updateModel( ... ) below.
    //
}

void ParameterManagerDelegate::updateEditorGeometry( QWidget* editor, const QStyleOptionViewItem& option, const QModelIndex& index ) const
{
    if ( index.column() == SLIDER_COLUMN )
    {
        QRect rect;

        rect.setBottomLeft( option.rect.bottomLeft() + QPoint( + SLIDER_HORIZONTAL_PADDING_PIXELS, - SLIDER_VERTICAL_PADDING_PIXELS ) );
        rect.setTopRight(   option.rect.topRight()   + QPoint( + SLIDER_HORIZONTAL_PADDING_PIXELS, + SLIDER_VERTICAL_PADDING_PIXELS ) );

        editor->setGeometry( rect );
    }
    else if( index.column() == VALUE_COLUMN )
    {
        QRect rect;

        rect.setBottomLeft( option.rect.bottomLeft() + QPoint( + SLIDER_HORIZONTAL_PADDING_PIXELS, - SLIDER_VERTICAL_PADDING_PIXELS ) );
        rect.setTopRight(   option.rect.topRight()   + QPoint( + SLIDER_HORIZONTAL_PADDING_PIXELS, + SLIDER_VERTICAL_PADDING_PIXELS ) );

        editor->setGeometry( rect );
    }
    else
    {
        editor->setGeometry( option.rect );
    }
}

void ParameterManagerDelegate::paint( QPainter* painter, const QStyleOptionViewItem& option, const QModelIndex& index ) const
{
    //
    // text position
    //
    int textX = option.rect.bottomLeft().x() + TEXT_INDENTATION;
    int textY = option.rect.bottomLeft().y() - ( ROW_HEIGHT_PIXELS / 2 ) + FONT;

    QString value = qvariant_cast< QString >( index.data() );

    if ( index.column() == KEY_COLUMN )
    {
        painter->drawText( QPoint( textX, textY ), value );
    }
    /*else if ( index.column() == VALUE_COLUMN )
    {
        //
        // trim off excessive decimal places
        //
        value = value.left( NUM_DISPLAY_DIGITS );
        painter->drawText( QPoint( textX, textY ), value );
    }*/
}

QSize ParameterManagerDelegate::sizeHint( const QStyleOptionViewItem& option, const QModelIndex& index ) const
{
    if ( index.column() == SLIDER_COLUMN )
    {
        // return minimum size that makes sense
        return QSize( SLIDER_WIDTH_PIXELS + ( SLIDER_HORIZONTAL_PADDING_PIXELS * 2 ), ROW_HEIGHT_PIXELS );
    }
    else
    {
        return QSize( MINIMUM_ELEMENT_WIDTH, ROW_HEIGHT_PIXELS );
    }
}

void ParameterManagerDelegate::setParameterRange( float minimumValue, float maximumValue )
{
    Assert( minimumValue < maximumValue );

    mCurrentMinimumValue = minimumValue;
    mCurrentMaximumValue = maximumValue;
}

void ParameterManagerDelegate::setInitializing( bool initializing )
{
    if ( !initializing )
    {
        //
        // user experience is improved when a signal fires on the valueChanged event, but
        // this causes erroneous updates at initialization time.  we only turn this on after
        // initialization.
        //
        foreach ( QList< Slider* > sliderList, sSliderLists )
        {
            foreach ( Slider* slider, sliderList )
            {
                slider->connectValueChangedUpdateModelSlot();
            }
        }
        foreach ( QList< SpinBox* > spinBoxList, sSpinBoxLists )
        {
            foreach ( SpinBox* spinBox, spinBoxList )
            {
                spinBox->connectValueChangedUpdateModelSlot();
            }
        }
    }
}

void ParameterManagerDelegate::updateModel()
{
    foreach ( QList< Slider* > sliderList, sSliderLists )
    {
        foreach( Slider* slider, sliderList )
        {
            slider->updateModel();
        }
    }
    foreach( QList< SpinBox* > spinBoxList, sSpinBoxLists )
    {
        foreach( SpinBox* spinBox, spinBoxList )
        {
            spinBox->updateModel();
        }
    }
}

void ParameterManagerDelegate::clearSliderList()
{
    sSliderLists.clear();
}

void ParameterManagerDelegate::clearSpinBoxList()
{
    sSpinBoxLists.clear();
}

Slider::Slider(
    MainWindow*           mainWindow,
    QAbstractItemModel*   model,
    QAbstractItemView*    view,
    QPersistentModelIndex persistentModelIndex,
    float                 minimumValue,
    float                 maximumValue,
    QWidget*              parent ) :
QSlider              ( parent ),
mMainWindow          ( mainWindow ),
mModel               ( model ),
mView                ( view ),
mPersistentModelIndex( persistentModelIndex ),
mMinimumValue        ( minimumValue ),
mMaximumValue        ( maximumValue )
{
    setFocusPolicy( Qt::NoFocus );
    setOrientation( Qt::Horizontal );

    bool connected = connect( this, SIGNAL( sliderMoved(  int ) ), this, SLOT( updateModel( int ) ) );
    Assert( connected );
}

void Slider::updateModel( int value )
{
    float range = mMaximumValue - mMinimumValue;

    //QModelIndex textFieldModelIndex = mPersistentModelIndex.sibling( mPersistentModelIndex.row(), 1 );

    mModel->setData( mPersistentModelIndex, ( range * value / maximum() ) + mMinimumValue );
    //mModel->setData( textFieldModelIndex,   ( range * value / maximum() ) + mMinimumValue );

    mMainWindow->updateUserEditableParameter( mPersistentModelIndex );
}

void Slider::updateModel()
{
    mMainWindow->updateUserEditableParameter( mPersistentModelIndex );
}

void Slider::getSliderRange( float* minimumValue, float* maximumValue )
{
    Assert( minimumValue != NULL && maximumValue != NULL );

    *minimumValue = mMinimumValue;
    *maximumValue = mMaximumValue;
}

void Slider::connectValueChangedUpdateModelSlot()
{
    bool connected = connect( this, SIGNAL( valueChanged( int ) ), this, SLOT( updateModel( int ) ) );
    Assert( connected );
}

SpinBox::SpinBox(
    MainWindow*           mainWindow,
    QAbstractItemModel*   model,
    QAbstractItemView*    view,
    QPersistentModelIndex persistentModelIndex,
    float                 minimumValue,
    float                 maximumValue,
    QWidget*              parent ) :
QDoubleSpinBox              ( parent ),
mMainWindow          ( mainWindow ),
mModel               ( model ),
mView                ( view ),
mPersistentModelIndex( persistentModelIndex ),
mMinimumValue        ( minimumValue ),
mMaximumValue        ( maximumValue )
{
    //setFocusPolicy( Qt::NoFocus );
    setAccelerated( false );
    setKeyboardTracking( false );

    setRange( mMinimumValue, mMaximumValue );
    setDecimals( 3 );

    double range = mMaximumValue - mMinimumValue;
    setSingleStep( range / static_cast< double >( SLIDER_WIDTH_PIXELS ) );

    setSizePolicy( QSizePolicy::Minimum, QSizePolicy::Minimum );

    // Don't connect valueChanged slot here!
}

void SpinBox::updateModel( double value )
{
    //QModelIndex sliderFieldModelIndex = mPersistentModelIndex.sibling( mPersistentModelIndex.row(), 2 );

    mModel->setData( mPersistentModelIndex, value );
   //mModel->setData( sliderFieldModelIndex, value );

    mMainWindow->updateUserEditableParameter( mPersistentModelIndex );

}

void SpinBox::updateModel()
{
    mMainWindow->updateUserEditableParameter( mPersistentModelIndex );
}

void SpinBox::getRange( float* minimumValue, float* maximumValue )
{
    Assert( minimumValue != NULL && maximumValue != NULL );

    *minimumValue = mMinimumValue;
    *maximumValue = mMaximumValue;
}

void SpinBox::connectValueChangedUpdateModelSlot()
{
    bool connected = connect( this, SIGNAL( valueChanged( double ) ), this, SLOT( updateModel( double ) ) );
    Assert( connected );
}
