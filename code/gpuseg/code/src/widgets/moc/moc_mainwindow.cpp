/****************************************************************************
** Meta object code from reading C++ file 'MainWindow.hpp'
**
** Created: Fri Sep 6 21:20:18 2013
**      by: The Qt Meta Object Compiler version 61 (Qt 4.5.2)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../../widgets/MainWindow.hpp"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'MainWindow.hpp' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 61
#error "This file was generated using the moc from 4.5.2. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_MainWindow[] = {

 // content:
       2,       // revision
       0,       // classname
       0,    0, // classinfo
      21,   12, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors

 // slots: signature, parameters, type, tag, flags
      17,   11,   12,   11, 0x0a,
      26,   11,   11,   11, 0x0a,
      74,   11,   11,   11, 0x0a,
     117,   11,   11,   11, 0x0a,
     148,   11,   11,   11, 0x0a,
     174,   11,   11,   11, 0x0a,
     192,   11,   11,   11, 0x0a,
     212,   11,   11,   11, 0x0a,
     227,   11,   12,   11, 0x0a,
     250,   11,   12,   11, 0x0a,
     270,   11,   12,   11, 0x0a,
     300,  295,   11,   11, 0x0a,
     322,   11,   11,   11, 0x0a,
     343,   11,   11,   11, 0x0a,
     364,   11,   11,   11, 0x0a,
     395,   11,   11,   11, 0x0a,
     427,   11,   11,   11, 0x0a,
     455,   11,   11,   11, 0x0a,
     489,   11,   11,   11, 0x0a,
     500,   11,   11,   11, 0x0a,
     511,   11,   11,   11, 0x0a,

       0        // eod
};

static const char qt_meta_stringdata_MainWindow[] = {
    "MainWindow\0\0bool\0update()\0"
    "volumeDirectoryDescriptionParameterFileSelect()\0"
    "volumeFileDescriptionParameterFileSelect()\0"
    "volumeDirectoryDescriptionOK()\0"
    "volumeFileDescriptionOK()\0openProject(bool)\0"
    "openDirectory(bool)\0openFile(bool)\0"
    "saveParametersAs(bool)\0saveProjectAs(bool)\0"
    "saveSegmentationAs(bool)\0exit\0"
    "exitApplication(bool)\0mayaCameraTool(bool)\0"
    "sketchSeedTool(bool)\0"
    "clearCurrentSegmentation(bool)\0"
    "freezeCurrentSegmentation(bool)\0"
    "clearAllSegmentations(bool)\0"
    "finishedSegmentationSession(bool)\0"
    "play(bool)\0stop(bool)\0lockParameters(bool)\0"
};

const QMetaObject MainWindow::staticMetaObject = {
    { &QMainWindow::staticMetaObject, qt_meta_stringdata_MainWindow,
      qt_meta_data_MainWindow, 0 }
};

const QMetaObject *MainWindow::metaObject() const
{
    return &staticMetaObject;
}

void *MainWindow::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_MainWindow))
        return static_cast<void*>(const_cast< MainWindow*>(this));
    return QMainWindow::qt_metacast(_clname);
}

int MainWindow::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QMainWindow::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        switch (_id) {
        case 0: { bool _r = update();
            if (_a[0]) *reinterpret_cast< bool*>(_a[0]) = _r; }  break;
        case 1: volumeDirectoryDescriptionParameterFileSelect(); break;
        case 2: volumeFileDescriptionParameterFileSelect(); break;
        case 3: volumeDirectoryDescriptionOK(); break;
        case 4: volumeFileDescriptionOK(); break;
        case 5: openProject((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 6: openDirectory((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 7: openFile((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 8: { bool _r = saveParametersAs((*reinterpret_cast< bool(*)>(_a[1])));
            if (_a[0]) *reinterpret_cast< bool*>(_a[0]) = _r; }  break;
        case 9: { bool _r = saveProjectAs((*reinterpret_cast< bool(*)>(_a[1])));
            if (_a[0]) *reinterpret_cast< bool*>(_a[0]) = _r; }  break;
        case 10: { bool _r = saveSegmentationAs((*reinterpret_cast< bool(*)>(_a[1])));
            if (_a[0]) *reinterpret_cast< bool*>(_a[0]) = _r; }  break;
        case 11: exitApplication((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 12: mayaCameraTool((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 13: sketchSeedTool((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 14: clearCurrentSegmentation((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 15: freezeCurrentSegmentation((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 16: clearAllSegmentations((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 17: finishedSegmentationSession((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 18: play((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 19: stop((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 20: lockParameters((*reinterpret_cast< bool(*)>(_a[1]))); break;
        default: ;
        }
        _id -= 21;
    }
    return _id;
}
QT_END_MOC_NAMESPACE
