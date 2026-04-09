from __future__ import annotations

import sys
import types


class _DummyWidget:
    def __init__(self, *args, **kwargs):
        pass


class _DummyLayout(_DummyWidget):
    def setSpacing(self, *args, **kwargs):
        pass

    def addWidget(self, *args, **kwargs):
        pass

    def addLayout(self, *args, **kwargs):
        pass


class _DummyMessageBox:
    @staticmethod
    def critical(*args, **kwargs):
        pass


def _install_stub_modules() -> None:
    qtcore = types.ModuleType("qtpy.QtCore")
    qtcore.QTimer = _DummyWidget
    qtcore.Qt = types.SimpleNamespace(Horizontal=1, AlignCenter=0, AlignHCenter=0, AlignTop=0)

    qtgui = types.ModuleType("qtpy.QtGui")
    qtgui.QColor = _DummyWidget
    qtgui.QFont = _DummyWidget
    qtgui.QPainter = _DummyWidget
    qtgui.QPixmap = _DummyWidget

    qtwidgets = types.ModuleType("qtpy.QtWidgets")
    qtwidgets.QApplication = _DummyWidget
    qtwidgets.QCheckBox = _DummyWidget
    qtwidgets.QComboBox = _DummyWidget
    qtwidgets.QFileDialog = types.SimpleNamespace(getOpenFileName=lambda *args, **kwargs: ("", ""))
    qtwidgets.QGroupBox = _DummyWidget
    qtwidgets.QHBoxLayout = _DummyLayout
    qtwidgets.QLabel = _DummyWidget
    qtwidgets.QListWidget = _DummyWidget
    qtwidgets.QListWidgetItem = _DummyWidget
    qtwidgets.QMainWindow = _DummyWidget
    qtwidgets.QMessageBox = _DummyMessageBox
    qtwidgets.QPushButton = _DummyWidget
    qtwidgets.QSplashScreen = _DummyWidget
    qtwidgets.QScrollArea = _DummyWidget
    qtwidgets.QSlider = _DummyWidget
    qtwidgets.QSpinBox = _DummyWidget
    qtwidgets.QSplitter = _DummyWidget
    qtwidgets.QVBoxLayout = _DummyLayout
    qtwidgets.QWidget = _DummyWidget

    qtpy = types.ModuleType("qtpy")
    qtpy.__path__ = []
    qtpy.QtCore = qtcore
    qtpy.QtGui = qtgui
    qtpy.QtWidgets = qtwidgets

    pyvista = types.ModuleType("pyvista")
    pyvista.XdmfReader = type("XdmfReader", (), {})
    pyvista.UnstructuredGrid = type("UnstructuredGrid", (), {})

    pyvistaqt = types.ModuleType("pyvistaqt")
    pyvistaqt.QtInteractor = type("QtInteractor", (), {})

    vtk_rendering_core = types.ModuleType("vtkmodules.vtkRenderingCore")
    vtk_rendering_core.vtkCellPicker = type("vtkCellPicker", (), {})

    vtkmodules = types.ModuleType("vtkmodules")
    vtkmodules.__path__ = []
    vtkmodules.vtkRenderingCore = vtk_rendering_core

    sys.modules.setdefault("qtpy", qtpy)
    sys.modules.setdefault("qtpy.QtCore", qtcore)
    sys.modules.setdefault("qtpy.QtGui", qtgui)
    sys.modules.setdefault("qtpy.QtWidgets", qtwidgets)
    sys.modules.setdefault("pyvista", pyvista)
    sys.modules.setdefault("pyvistaqt", pyvistaqt)
    sys.modules.setdefault("vtkmodules", vtkmodules)
    sys.modules.setdefault("vtkmodules.vtkRenderingCore", vtk_rendering_core)


_install_stub_modules()