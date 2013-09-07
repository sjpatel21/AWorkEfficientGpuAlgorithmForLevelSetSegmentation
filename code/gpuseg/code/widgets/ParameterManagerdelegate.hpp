#ifndef GPUSEG_WIDGETS_PARAMETER_MANAGER_DELEGATE_HPP
#define GPUSEG_WIDGETS_PARAMETER_MANAGER_DELEGATE_HPP

#include <QtCore/QList>
#include <QtCore/QObject>
#include <QtCore/QSize>
#include <QtCore/QPersistentModelIndex>
#include <QtGui/QAbstractItemView>
#include <QtGui/QItemDelegate>
#include <QtGui/QSlider>
#include <QtGui/QDoubleSpinBox>

#include "widgets/MainWindow.hpp"

class Slider;

class ParameterManagerDelegate : public QItemDelegate
{
    Q_OBJECT

public:

    ParameterManagerDelegate(
        MainWindow*         mainWindow,
        QAbstractItemModel* model,
        QAbstractItemView*  view,
        QObject*            parent = NULL );

    virtual QWidget* createEditor      ( QWidget* parent, const QStyleOptionViewItem& option, const QModelIndex& index ) const;
    virtual void setEditorData         ( QWidget* editor, const QModelIndex &index ) const;
    virtual void setModelData          ( QWidget* editor, QAbstractItemModel* model, const QModelIndex& index ) const;
    virtual void updateEditorGeometry  ( QWidget* editor, const QStyleOptionViewItem& option, const QModelIndex& index ) const;

    virtual void paint         ( QPainter* painter, const QStyleOptionViewItem& option, const QModelIndex& index ) const;
    virtual QSize sizeHint     ( const QStyleOptionViewItem& option, const QModelIndex& index ) const;

    void updateModel();

    //
    // set the state of the delegate
    //
    void setParameterRange ( float minimumValue, float maximumValue );
    void setInitializing( bool initializing );

    void clearSliderList();
    void clearSpinBoxList();

protected:
    MainWindow*         mMainWindow;
    QAbstractItemModel* mModel;
    QAbstractItemView*  mView;

    QString             mCurrentParameterName;
    float               mCurrentMinimumValue;
    float               mCurrentMaximumValue;
    bool                mInitializing;
};

class Slider : public QSlider
{
    Q_OBJECT

public:
    Slider(
        MainWindow*           mainWindow,
        QAbstractItemModel*   model,
        QAbstractItemView*    view,
        QPersistentModelIndex persistentModelIndex,
        float                 minimumValue,
        float                 maximumValue,
        QWidget*              parent );

    void getSliderRange( float* minimumValue, float* maximumValue );
    void connectValueChangedUpdateModelSlot();
    void updateModel();

public slots:
    virtual void updateModel( int value );

protected:
    MainWindow*           mMainWindow;
    QPersistentModelIndex mPersistentModelIndex;
    QAbstractItemModel*   mModel;
    QAbstractItemView*    mView;

    float                 mMinimumValue;
    float                 mMaximumValue;
    QString               mParameterName;

};

class SpinBox : public QDoubleSpinBox
{
    Q_OBJECT

public:
    SpinBox(
        MainWindow*           mainWindow,
        QAbstractItemModel*   model,
        QAbstractItemView*    view,
        QPersistentModelIndex persistentModelIndex,
        float                 minimumValue,
        float                 maximumValue,
        QWidget*              parent );

    void getRange( float* minimumValue, float* maximumValue );
    void connectValueChangedUpdateModelSlot();
    void updateModel();

public slots:
    virtual void updateModel( double value );

protected:
    MainWindow*           mMainWindow;
    QPersistentModelIndex mPersistentModelIndex;
    QAbstractItemModel*   mModel;
    QAbstractItemView*    mView;

    float                 mMinimumValue;
    float                 mMaximumValue;
    QString               mParameterName;

};
#endif
