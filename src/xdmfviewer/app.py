# -*- coding: utf-8 -*-
"""Interactive Qt GUI for plotting XDMF time-series results.

This module requires ``pyvistaqt`` and ``pyvista`` to be installed.

Run with::

    python -m xdmfviewer
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyvista as pv
from vtkmodules.vtkRenderingCore import vtkCellPicker

from .version import __version__

try:
    from pyvistaqt import QtInteractor
    from qtpy.QtCore import QTimer, Qt
    from qtpy.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QFileDialog,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QListWidget,
        QListWidgetItem,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QSplashScreen,
        QScrollArea,
        QSlider,
        QSpinBox,
        QSplitter,
        QVBoxLayout,
        QWidget,
    )
except ModuleNotFoundError as exc:
    raise SystemExit(
        "This example requires pyvistaqt and a Qt binding. "
        "Install with: pip install pyvistaqt"
    ) from exc


# ============================================================================
# Data Classes & Helper Classes
# ============================================================================

@dataclass
class FieldMeta:
    """Metadata for a plottable field."""

    location: str
    name: str
    kind: str
    flat_size: int


@dataclass
class DisplayState:
    """Tracks 3D display rendering state."""

    axes_added: bool = False
    camera_is_initialized: bool = False


@dataclass
class RenderConfig:
    """Encapsulates rendering parameters and visual settings."""

    warp_vector_name: str | None = None
    discrete_cmap: bool = False
    cmap_levels: int = 10
    lock_limits: bool = False
    log_scale: bool = False
    invert_cmap: bool = False
    show_edges: bool = True
    colormap: str = "turbo"

    def to_plotter_kwargs(self) -> dict:
        """Convert to PyVista add_mesh kwargs for colormap config."""
        kwargs = {}
        if self.discrete_cmap:
            kwargs["n_colors"] = self.cmap_levels
            kwargs["interpolate_before_map"] = False
        return kwargs


@dataclass
class HoverState:
    """Hover tooltip state machine."""

    observer_id: int | None = None
    last_target: tuple[str, int] | None = None
    enabled: bool = False


@dataclass
class XDMFFileState:
    """XDMF file and mesh data state."""

    filename: str | None = None
    reader: pv.XdmfReader | None = None
    mesh: pv.UnstructuredGrid | None = None
    times: list[float] = field(default_factory=list)
    current_step: int = 0
    field_signature: tuple[tuple[str, ...], tuple[str, ...]] | None = None

    def clear(self) -> None:
        """Clear all state."""
        self.filename = None
        self.reader = None
        self.mesh = None
        self.times = []
        self.current_step = 0
        self.field_signature = None

    def is_loaded(self) -> bool:
        """Check if file is loaded."""
        return self.reader is not None and self.mesh is not None


class ScalarExtractor:
    """Extracts and processes scalar data from meshes."""

    @staticmethod
    def extract_component(values: np.ndarray, component: int) -> np.ndarray:
        """Extract scalar component from vector/tensor data."""
        arr = np.asarray(values)

        if arr.ndim <= 1:
            return arr

        leading = arr.shape[0]
        flat = arr.reshape(leading, -1)

        if component == -2:  # Magnitude
            return np.linalg.norm(flat, axis=1)

        if component < 0 or component >= flat.shape[1]:
            return flat[:, 0]

        return flat[:, component]


class StatusFormatter:
    """Formats status messages for display."""

    @staticmethod
    def format_min_max(values: np.ndarray | None) -> str:
        """Format min/max values with scientific notation."""
        if values is None:
            return "n/a"

        arr = np.asarray(values)
        if arr.size == 0:
            return "n/a"

        finite_values = arr[np.isfinite(arr)]
        if finite_values.size == 0:
            return "n/a"

        return (
            f"Min={float(np.min(finite_values)):.3e}, "
            f"Max={float(np.max(finite_values)):.3e}"
        )

    @staticmethod
    def build_status_text(
        file_name: str,
        step: int,
        step_count: int,
        current_limits: str,
        global_limits: str,
    ) -> str:
        """Build multi-line status text."""
        return (
            f"{file_name}\n"
            f"Step {step + 1}/{max(1, step_count)}\n"
            f"{current_limits}\n"
            f"{global_limits} (All Steps)"
        )

    @staticmethod
    def scalar_bar_title(field_name: str, component_label: str) -> str:
        """Build scalar bar title with component."""
        if not component_label or component_label.lower() == "scalar":
            return field_name
        return f"{field_name} {component_label}"


class MeshRenderer:
    """Encapsulates all mesh rendering logic."""

    def __init__(self, plotter: QtInteractor) -> None:
        self.plotter = plotter
        self.display_mesh: pv.UnstructuredGrid | None = None
        self.display_cell_centers: np.ndarray | None = None
        self.state = DisplayState()

    def set_status_overlay(self, text: str) -> None:
        """Show the current status text in the top-left corner of the plot."""
        self.plotter.add_text(
            text,
            position="upper_left",
            font_size=10,
            color="#1f2937",
            shadow=False,
            name="status-overlay",
            viewport=False,
            render=False,
        )

    def prepare_mesh_for_display(
        self,
        mesh: pv.UnstructuredGrid,
        warp_vector_name: str | None = None,
        warp_factor: float = 1.0,
    ) -> pv.UnstructuredGrid:
        """Prepare mesh for display and apply warp if specified."""

        mesh_to_plot = mesh.copy(deep=True)

        # Apply warp transformation only if a vector field is explicitly selected
        if warp_vector_name:
            try:
                mesh_to_plot = mesh_to_plot.warp_by_vector(
                    warp_vector_name,
                    factor=warp_factor,
                )
            except Exception:
                pass  # Warp vector not available

        self.display_mesh = mesh_to_plot
        return mesh_to_plot

    def attach_scalars(
        self,
        mesh_to_plot: pv.UnstructuredGrid,
        scalars: np.ndarray | None,
        location: str,
    ) -> None:
        """Attach scalars to mesh and cache cell centers if needed."""
        if scalars is None:
            return

        if location == "point":
            mesh_to_plot.point_data["__active_scalars__"] = scalars
        else:
            mesh_to_plot.cell_data["__active_scalars__"] = scalars
            mesh_to_plot.set_active_scalars("__active_scalars__", preference="cell")
            self.display_cell_centers = mesh_to_plot.cell_centers().points

    def render_frame(
        self,
        mesh_to_plot: pv.UnstructuredGrid,
        scalars: np.ndarray | None = None,
        scalar_name: str = "",
        color_limits: tuple[float, float] | None = None,
        location: str = "point",
        config: RenderConfig | None = None,
        log_scale: bool = False,
    ) -> None:
        """Render mesh with optional scalars and apply view."""
        if config is None:
            config = RenderConfig()

        self.plotter.clear()

        if scalars is None:
            self.plotter.add_mesh(
                mesh_to_plot,
                name="result-mesh",
                color="lightgray",
                show_edges=config.show_edges,
                show_scalar_bar=False,
                reset_camera=False,
                render=False,
            )
        else:
            add_kwargs = {"preference": location}
            if color_limits is not None:
                add_kwargs["clim"] = color_limits
            add_kwargs.update(config.to_plotter_kwargs())

            self.plotter.add_mesh(
                mesh_to_plot,
                name="result-mesh",
                scalars="__active_scalars__",
                cmap=config.colormap,
                flip_scalars=config.invert_cmap,
                show_edges=config.show_edges,
                scalar_bar_args={"title": scalar_name or ""},
                show_scalar_bar=True,
                reset_camera=False,
                log_scale=log_scale,
                render=False,
                **add_kwargs,
            )

        if not self.state.axes_added:
            self.plotter.add_axes()
            self.state.axes_added = True

        if not self.state.camera_is_initialized:
            self.plotter.reset_camera()
            self.state.camera_is_initialized = True

    def reset_state(self) -> None:
        """Reset display state."""
        self.state = DisplayState()
        self.display_mesh = None
        self.display_cell_centers = None


class HoverTooltipManager:
    """Manages hover tooltip rendering and picking logic."""

    def __init__(
        self,
        plotter: QtInteractor,
        cell_picker: vtkCellPicker,
    ) -> None:
        self.plotter = plotter
        self.cell_picker = cell_picker
        self.state = HoverState()
        self.label_offset = np.array([0.05, 0.05, 0.05], dtype=float)
        self._mouse_move_callback = None

    def set_mouse_move_callback(self, callback) -> None:
        """Set callback for mouse move events."""
        self._mouse_move_callback = callback

    def enable(self) -> None:
        """Enable hover tooltip."""
        if self.state.observer_id is not None:
            return
        self.state.observer_id = self.plotter.iren.add_observer(
            "MouseMoveEvent", self._on_mouse_move
        )
        self.state.enabled = True

    def disable(self) -> None:
        """Disable hover tooltip."""
        if self.state.observer_id is not None:
            self.plotter.iren.remove_observer(self.state.observer_id)
            self.state.observer_id = None
        self.clear()
        self.state.enabled = False

    def clear(self) -> None:
        """Clear current hover tooltip."""
        self.state.last_target = None
        self.plotter.remove_actor("hover-tooltip", render=False)
        self.plotter.remove_actor("hover-cell-highlight", render=False)
        self.plotter.remove_actor("hover-point-highlight", render=False)

    def _on_mouse_move(self, _obj, _event) -> None:
        """Handle mouse move event."""
        if self._mouse_move_callback is not None:
            self._mouse_move_callback()

    def get_label_position(
        self, position: np.ndarray, mesh: pv.UnstructuredGrid | None
    ) -> np.ndarray:
        """Calculate label position with offset based on mesh bounds."""
        if mesh is None or mesh.n_points == 0:
            return np.asarray(position, dtype=float)

        bounds = np.asarray(mesh.bounds, dtype=float)
        span = np.array(
            [bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]],
            dtype=float,
        )
        diagonal = float(np.linalg.norm(span))
        if not np.isfinite(diagonal) or diagonal == 0.0:
            diagonal = 1.0

        return np.asarray(position, dtype=float) + self.label_offset * diagonal

    @staticmethod
    def format_value(value: float) -> str:
        """Format scalar value for display."""
        if not np.isfinite(value):
            return "nan"
        return f"{value:.6g}"


# ============================================================================
# Main GUI Application
# ============================================================================

class XdmfGuiApp(QMainWindow):
    """Main window for browsing and plotting XDMF time-series results."""

    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle(f"XDMF Viewer v{__version__}")
        self.resize(1400, 900)

        # Core state
        self.file_state = XDMFFileState()
        self.field_meta: dict[tuple[str, str], FieldMeta] = {}
        self.active_location: str | None = None
        self.active_field_name: str | None = None

        # Caches
        self._global_limits_cache: dict[tuple[str, str, int], tuple[float, float]] = {}
        self._mesh_cache: dict[int, pv.UnstructuredGrid] = {}

        # UI components
        self.plotter = QtInteractor(self)
        self.renderer = MeshRenderer(self.plotter)
        self.render_config = RenderConfig()

        # Hover management
        self._hover_cell_picker = vtkCellPicker()
        self.hover_manager = HoverTooltipManager(
            self.plotter, self._hover_cell_picker
        )
        self.hover_manager.set_mouse_move_callback(self._on_mouse_move)

        # Animation
        self._timer = QTimer(self)
        self._timer.setInterval(250)
        self._timer.timeout.connect(self._advance_step)

        # UI elements (set later in _build_ui)
        self.btn_open: QPushButton | None = None
        self.btn_close: QPushButton | None = None
        self.btn_play: QPushButton | None = None
        self.point_fields: QListWidget | None = None
        self.cell_fields: QListWidget | None = None
        self.component_combo: QComboBox | None = None
        self.lock_limits: QCheckBox | None = None
        self.cache_steps: QCheckBox | None = None
        self.view_combo: QComboBox | None = None
        self.background_combo: QComboBox | None = None
        self.hover_values: QCheckBox | None = None
        self.discrete_cmap: QCheckBox | None = None
        self.log_scale: QCheckBox | None = None
        self.invert_cmap: QCheckBox | None = None
        self.cmap_levels: QSpinBox | None = None
        self.frame_interval: QSpinBox | None = None
        self.warp_vector: QComboBox | None = None
        self.time_slider: QSlider | None = None
        self.btn_export_screenshot: QPushButton | None = None
        self.export_fps: QSpinBox | None = None
        self.btn_export_animation: QPushButton | None = None

        self._build_ui()
        self._apply_background()
        self._set_idle_state()

    def _build_ui(self) -> None:
        """Build main UI layout."""
        central = QWidget(self)
        self.setCentralWidget(central)

        layout = QHBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        splitter = QSplitter(Qt.Horizontal, self)
        layout.addWidget(splitter)

        sidebar = self._build_sidebar()
        sidebar_scroll = QScrollArea(self)
        sidebar_scroll.setWidgetResizable(True)
        sidebar_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        sidebar_scroll.setWidget(sidebar)

        splitter.addWidget(sidebar_scroll)
        splitter.addWidget(self.plotter)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([380, 1020])

    def _build_sidebar(self) -> QWidget:
        """Build sidebar with all control groups."""
        sidebar = QWidget(self)
        sidebar.setObjectName("sidebarPanel")
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(10, 10, 10, 10)
        sidebar_layout.setSpacing(10)

        sidebar.setStyleSheet(self._get_sidebar_stylesheet())

        sidebar_layout.addWidget(self._build_file_group())
        sidebar_layout.addWidget(self._build_data_group())
        sidebar_layout.addWidget(self._build_view_group())
        sidebar_layout.addWidget(self._build_animation_group())
        sidebar_layout.addWidget(self._build_export_group())

        sidebar_layout.addStretch(1)

        return sidebar

    @staticmethod
    def _get_sidebar_stylesheet() -> str:
        """Return sidebar stylesheet."""
        return """
            QWidget#sidebarPanel {
                background-color: #f4f6f8;
                border: 1px solid #d9dee5;
                border-radius: 8px;
            }
            QGroupBox {
                font-weight: 600;
                border: 1px solid #d9dee5;
                border-radius: 6px;
                margin-top: 12px;
                padding: 10px 8px 8px 8px;
                background-color: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
                color: #2f3b4a;
            }
            QLabel.section-label {
                color: #465566;
                font-size: 11px;
                font-weight: 500;
                margin-top: 4px;
            }
            """

    def _build_file_group(self) -> QGroupBox:
        """Build File group."""
        group = QGroupBox("File", self)
        layout = QVBoxLayout(group)
        layout.setSpacing(6)

        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(6)

        self.btn_open = QPushButton("Load XDMF")
        self.btn_open.clicked.connect(self.open_file)
        buttons_layout.addWidget(self.btn_open)

        self.btn_close = QPushButton("Close XDMF")
        self.btn_close.clicked.connect(self.close_file)
        buttons_layout.addWidget(self.btn_close)

        layout.addLayout(buttons_layout)
        return group

    def _build_data_group(self) -> QGroupBox:
        """Build Data group."""
        group = QGroupBox("Data", self)
        layout = QVBoxLayout(group)
        layout.setSpacing(6)

        point_label = QLabel("Point data")
        point_label.setProperty("class", "section-label")
        layout.addWidget(point_label)
        self.point_fields = QListWidget(self)
        self.point_fields.itemSelectionChanged.connect(self._on_point_field_changed)
        layout.addWidget(self.point_fields)

        cell_label = QLabel("Cell data")
        cell_label.setProperty("class", "section-label")
        layout.addWidget(cell_label)
        self.cell_fields = QListWidget(self)
        self.cell_fields.itemSelectionChanged.connect(self._on_cell_field_changed)
        layout.addWidget(self.cell_fields)

        component_label = QLabel("Component")
        component_label.setProperty("class", "section-label")
        layout.addWidget(component_label)
        self.component_combo = QComboBox(self)
        self.component_combo.currentIndexChanged.connect(self._on_component_changed)
        layout.addWidget(self.component_combo)

        return group

    def _build_view_group(self) -> QGroupBox:
        """Build Display/View group."""
        group = QGroupBox("Display", self)
        layout = QVBoxLayout(group)
        layout.setSpacing(6)

        self.lock_limits = QCheckBox("Constant limits over all steps")
        self.lock_limits.toggled.connect(self._on_scalar_limits_mode_changed)
        layout.addWidget(self.lock_limits)

        self.cache_steps = QCheckBox("Cache time steps in RAM")
        self.cache_steps.setChecked(True)
        self.cache_steps.toggled.connect(self._on_cache_steps_toggled)
        layout.addWidget(self.cache_steps)

        view_label = QLabel("View")
        view_label.setProperty("class", "section-label")
        layout.addWidget(view_label)
        self.view_combo = QComboBox(self)
        self.view_combo.addItem("XY", "xy")
        self.view_combo.addItem("YZ", "yz")
        self.view_combo.addItem("XZ", "xz")
        self.view_combo.addItem("3D", "3d")
        self.view_combo.currentIndexChanged.connect(self._on_view_changed)
        layout.addWidget(self.view_combo)

        background_label = QLabel("Background")
        background_label.setProperty("class", "section-label")
        layout.addWidget(background_label)
        self.background_combo = QComboBox(self)
        self.background_combo.addItem("Gradient", "gradient")
        self.background_combo.addItem("White", "white")
        self.background_combo.currentIndexChanged.connect(self._on_background_changed)
        layout.addWidget(self.background_combo)

        self.hover_values = QCheckBox("Show hover tooltip")
        self.hover_values.setChecked(False)
        self.hover_values.toggled.connect(self._on_hover_toggled)
        layout.addWidget(self.hover_values)

        self.discrete_cmap = QCheckBox("Use discrete colormap")
        self.discrete_cmap.setChecked(False)
        self.discrete_cmap.toggled.connect(self._on_discrete_cmap_toggled)
        layout.addWidget(self.discrete_cmap)

        self.log_scale = QCheckBox("Use logarithmic color scale")
        self.log_scale.setChecked(False)
        self.log_scale.toggled.connect(self._on_color_mapping_changed)
        layout.addWidget(self.log_scale)

        self.invert_cmap = QCheckBox("Invert color scale")
        self.invert_cmap.setChecked(False)
        self.invert_cmap.toggled.connect(self._on_color_mapping_changed)
        layout.addWidget(self.invert_cmap)

        levels_label = QLabel("Colormap levels")
        levels_label.setProperty("class", "section-label")
        layout.addWidget(levels_label)
        self.cmap_levels = QSpinBox(self)
        self.cmap_levels.setRange(2, 256)
        self.cmap_levels.setSingleStep(1)
        self.cmap_levels.setValue(10)
        self.cmap_levels.valueChanged.connect(self._on_discrete_levels_changed)
        self.cmap_levels.setEnabled(False)
        layout.addWidget(self.cmap_levels)

        warp_label = QLabel("Warp by Vector")
        warp_label.setProperty("class", "section-label")
        layout.addWidget(warp_label)
        self.warp_vector = QComboBox(self)
        self.warp_vector.currentIndexChanged.connect(self._on_warp_vector_changed)
        layout.addWidget(self.warp_vector)

        return group

    def _build_animation_group(self) -> QGroupBox:
        """Build Animation group."""
        group = QGroupBox("Animation", self)
        layout = QVBoxLayout(group)
        layout.setSpacing(6)

        self.btn_play = QPushButton("Play")
        self.btn_play.clicked.connect(self._toggle_playback)
        layout.addWidget(self.btn_play)

        frame_label = QLabel("Frame interval [ms]")
        frame_label.setProperty("class", "section-label")
        layout.addWidget(frame_label)
        self.frame_interval = QSpinBox(self)
        self.frame_interval.setRange(20, 2000)
        self.frame_interval.setSingleStep(10)
        self.frame_interval.setValue(250)
        self.frame_interval.valueChanged.connect(self._on_frame_interval_changed)
        layout.addWidget(self.frame_interval)

        step_label = QLabel("Time step")
        step_label.setProperty("class", "section-label")
        layout.addWidget(step_label)
        self.time_slider = QSlider(Qt.Horizontal, self)
        self.time_slider.valueChanged.connect(self._on_time_slider_changed)
        layout.addWidget(self.time_slider)

        return group

    def _build_export_group(self) -> QGroupBox:
        """Build Export group."""
        group = QGroupBox("Export", self)
        layout = QVBoxLayout(group)
        layout.setSpacing(6)

        self.btn_export_screenshot = QPushButton("Screenshot")
        self.btn_export_screenshot.clicked.connect(self._export_screenshot)
        layout.addWidget(self.btn_export_screenshot)

        fps_label = QLabel("Animation FPS")
        fps_label.setProperty("class", "section-label")
        layout.addWidget(fps_label)
        self.export_fps = QSpinBox(self)
        self.export_fps.setRange(1, 120)
        self.export_fps.setSingleStep(1)
        self.export_fps.setValue(10)
        layout.addWidget(self.export_fps)

        self.btn_export_animation = QPushButton("Animation")
        self.btn_export_animation.clicked.connect(self._export_animation)
        layout.addWidget(self.btn_export_animation)

        return group

    def _set_idle_state(self) -> None:
        """Disable all controls when no file is loaded."""
        controls = [
            self.btn_close, self.btn_play, self.time_slider,
            self.component_combo, self.lock_limits, self.cache_steps,
            self.view_combo, self.background_combo, self.hover_values, self.discrete_cmap,
            self.log_scale, self.invert_cmap,
            self.cmap_levels, self.frame_interval, self.warp_vector,
            self.btn_export_screenshot, self.export_fps, self.btn_export_animation,
        ]
        for ctrl in controls:
            if ctrl:
                ctrl.setEnabled(False)
        self.renderer.set_status_overlay("No file loaded.")

    def open_file(self) -> None:
        """Open XDMF file dialog."""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open XDMF file", "", "XDMF files (*.xdmf)",
        )
        if not filename:
            return
        self._load_xdmf(filename)

    def _load_xdmf(self, filename: str) -> None:
        """Load XDMF file and initialize."""
        try:
            reader = pv.XdmfReader(filename)
            times = self._extract_times(filename)
            if not times:
                times = [0.0]

            self.file_state.filename = filename
            self.file_state.reader = reader
            self.file_state.times = times
            self.file_state.current_step = 0
            self.file_state.field_signature = None
            self._global_limits_cache.clear()
            self._mesh_cache.clear()

            self._load_mesh_for_step(0, refresh_fields=True)
            self._set_default_view_for_mesh(self.file_state.mesh)

            self._enable_controls_on_load()
            self.renderer.set_status_overlay(
                StatusFormatter.build_status_text(
                    Path(filename).name,
                    self.file_state.current_step,
                    len(self.file_state.times),
                    current_limits="n/a",
                    global_limits="n/a",
                )
            )
            self._render()

        except Exception as exc:
            QMessageBox.critical(
                self, "Failed to open XDMF",
                f"Could not open file:\n{filename}\n\n{exc}",
            )
            self.close_file()

    def _enable_controls_on_load(self) -> None:
        """Enable controls after file is loaded."""
        self.btn_close.setEnabled(True)
        self.btn_play.setEnabled(len(self.file_state.times) > 1)
        self.time_slider.setEnabled(True)
        self.time_slider.blockSignals(True)
        self.time_slider.setMinimum(0)
        self.time_slider.setMaximum(max(0, len(self.file_state.times) - 1))
        self.time_slider.setValue(0)
        self.time_slider.blockSignals(False)
        self.lock_limits.setEnabled(True)
        self.cache_steps.setEnabled(True)
        self.view_combo.setEnabled(True)
        self.background_combo.setEnabled(True)
        self.hover_values.setEnabled(True)
        self.discrete_cmap.setEnabled(True)
        self.log_scale.setEnabled(True)
        self.invert_cmap.setEnabled(True)
        self.cmap_levels.setEnabled(self.discrete_cmap.isChecked())
        self.frame_interval.setEnabled(True)
        self.warp_vector.setEnabled(True)
        self.btn_export_screenshot.setEnabled(True)
        self.export_fps.setEnabled(True)
        self.btn_export_animation.setEnabled(len(self.file_state.times) > 0)

    def close_file(self) -> None:
        """Close file and reset UI."""
        self._timer.stop()
        self.btn_play.setText("Play")

        self.file_state.clear()
        self.field_meta.clear()
        self.active_location = None
        self.active_field_name = None
        self._global_limits_cache.clear()
        self._mesh_cache.clear()

        self.point_fields.clear()
        self.cell_fields.clear()
        self.component_combo.clear()
        self.lock_limits.setChecked(False)
        self.cache_steps.setChecked(True)
        self.view_combo.blockSignals(True)
        self.view_combo.setCurrentIndex(3)
        self.view_combo.blockSignals(False)
        self.background_combo.blockSignals(True)
        self.background_combo.setCurrentIndex(0)
        self.background_combo.blockSignals(False)
        self.hover_values.setChecked(False)
        self.discrete_cmap.setChecked(False)
        self.log_scale.setChecked(False)
        self.invert_cmap.setChecked(False)
        self.cmap_levels.setValue(10)
        self.warp_vector.blockSignals(True)
        self.warp_vector.clear()
        self.warp_vector.blockSignals(False)

        self._apply_background()

        self.plotter.clear()
        self.renderer.reset_state()

        self.renderer.set_status_overlay("No file loaded.")
        self._set_idle_state()

    @staticmethod
    def _extract_times(filename: str) -> list[float]:
        """Extract time values from XDMF file."""
        tree = ET.parse(filename)
        root = tree.getroot()
        times: list[float] = []

        for elem in root.iter():
            tag = elem.tag
            if "}" in tag:
                tag = tag.rsplit("}", 1)[1]
            if tag != "Time":
                continue

            value = elem.attrib.get("Value")
            if value is None:
                continue

            try:
                times.append(float(value))
            except ValueError:
                continue

        return sorted(set(times)) if times else []

    def _mesh_signature(self, mesh: pv.UnstructuredGrid) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Get field signature (point_data, cell_data keys)."""
        return tuple(mesh.point_data.keys()), tuple(mesh.cell_data.keys())

    def _get_mesh_for_step(self, step: int) -> pv.UnstructuredGrid:
        """Load mesh for given step, with caching."""
        if self.file_state.reader is None:
            raise RuntimeError("Reader is not initialized.")

        if self.cache_steps.isChecked() and step in self._mesh_cache:
            return self._mesh_cache[step]

        time_value = self.file_state.times[step]
        self.file_state.reader.set_active_time_value(time_value)
        mesh = self.file_state.reader.read()[0]

        if self.cache_steps.isChecked():
            self._mesh_cache[step] = mesh

        return mesh

    def _load_mesh_for_step(self, step: int, refresh_fields: bool = False) -> None:
        """Load mesh for step and optionally refresh field lists."""
        if self.file_state.reader is None:
            return

        step = max(0, min(step, len(self.file_state.times) - 1))
        self.file_state.mesh = self._get_mesh_for_step(step)
        self.file_state.current_step = step

        preferred_warp_vector = self.warp_vector.currentText() if self.warp_vector is not None else None

        if refresh_fields:
            signature = self._mesh_signature(self.file_state.mesh)
            if signature != self.file_state.field_signature:
                self.file_state.field_signature = signature
                self._refresh_field_lists_preserve_selection(render=False)
        
        self._populate_warp_vectors(preferred_warp_vector)

    def _refresh_field_lists(self) -> None:
        """Populate field lists from current mesh."""
        self.field_meta.clear()
        self.point_fields.clear()
        self.cell_fields.clear()

        if self.file_state.mesh is None:
            return

        for name in self.file_state.mesh.point_data.keys():
            kind, flat_size = self._detect_kind(self.file_state.mesh.point_data[name])
            self.field_meta[("point", name)] = FieldMeta(
                location="point", name=name, kind=kind, flat_size=flat_size
            )
            item = QListWidgetItem(name)
            item.setData(Qt.UserRole, name)
            self.point_fields.addItem(item)

        for name in self.file_state.mesh.cell_data.keys():
            kind, flat_size = self._detect_kind(self.file_state.mesh.cell_data[name])
            self.field_meta[("cell", name)] = FieldMeta(
                location="cell", name=name, kind=kind, flat_size=flat_size
            )
            item = QListWidgetItem(name)
            item.setData(Qt.UserRole, name)
            self.cell_fields.addItem(item)

    def _select_default_field(self, render: bool = True) -> None:
        """Select first available field."""
        if self.point_fields.count() > 0:
            self.point_fields.blockSignals(not render)
            self.point_fields.setCurrentRow(0)
            self.point_fields.blockSignals(False)
            items = self.point_fields.selectedItems()
            if items:
                name = items[0].data(Qt.UserRole)
                self._set_active_field("point", name, render=render)
        elif self.cell_fields.count() > 0:
            self.cell_fields.blockSignals(not render)
            self.cell_fields.setCurrentRow(0)
            self.cell_fields.blockSignals(False)
            items = self.cell_fields.selectedItems()
            if items:
                name = items[0].data(Qt.UserRole)
                self._set_active_field("cell", name, render=render)
        else:
            self.active_location = None
            self.active_field_name = None
            self.component_combo.clear()
            self.component_combo.setEnabled(False)

    @staticmethod
    def _detect_kind(values: np.ndarray) -> tuple[str, int]:
        """Detect field kind (scalar, vector, tensor, array)."""
        arr = np.asarray(values)

        if arr.ndim <= 1:
            return "scalar", 1

        shape_tail = arr.shape[1:]
        flat_size = int(np.prod(shape_tail))

        if flat_size == 1:
            return "scalar", 1
        if flat_size == 3:
            return "vector", 3
        if flat_size == 9:
            return "tensor", 9

        return "array", flat_size

    def _on_point_field_changed(self) -> None:
        """Handle point field selection."""
        items = self.point_fields.selectedItems()
        if not items:
            return

        self.cell_fields.blockSignals(True)
        self.cell_fields.clearSelection()
        self.cell_fields.blockSignals(False)

        name = items[0].data(Qt.UserRole)
        self._set_active_field("point", name)

    def _on_cell_field_changed(self) -> None:
        """Handle cell field selection."""
        items = self.cell_fields.selectedItems()
        if not items:
            return

        self.point_fields.blockSignals(True)
        self.point_fields.clearSelection()
        self.point_fields.blockSignals(False)

        name = items[0].data(Qt.UserRole)
        self._set_active_field("cell", name)

    def _set_active_field(self, location: str, name: str, render: bool = True) -> None:
        """Set active field and update component options."""
        self.active_location = location
        self.active_field_name = name

        meta = self.field_meta.get((location, name))
        self._populate_component_options(meta)
        if render:
            self._render()

    def _populate_component_options(self, meta: FieldMeta | None) -> None:
        """Populate component combo based on field kind."""
        self.component_combo.blockSignals(True)
        self.component_combo.clear()

        if meta is None:
            self.component_combo.setEnabled(False)
            self.component_combo.blockSignals(False)
            return

        if meta.kind == "scalar":
            self.component_combo.addItem("", -1)
            self.component_combo.setEnabled(False)
        elif meta.kind == "vector":
            self.component_combo.addItem("Magnitude", -2)
            for label, index in (("1", 0), ("2", 1), ("3", 2)):
                self.component_combo.addItem(label, index)
            self.component_combo.setEnabled(True)
        elif meta.kind == "tensor":
            tensor_map = (
                ("11", 0), ("12", 1), ("13", 2),
                ("21", 3), ("22", 4), ("23", 5),
                ("31", 6), ("32", 7), ("33", 8),
            )
            for label, index in tensor_map:
                self.component_combo.addItem(label, index)
            self.component_combo.setEnabled(True)
        else:
            self.component_combo.addItem("flat index 0", 0)
            self.component_combo.setEnabled(False)

        self.component_combo.setCurrentIndex(0)
        self.component_combo.blockSignals(False)

    def _on_component_changed(self) -> None:
        """Handle component selection change."""
        if self.file_state.mesh is None:
            return
        self._render()

    def _on_cache_steps_toggled(self) -> None:
        """Handle cache steps toggle."""
        if not self.cache_steps.isChecked():
            self._mesh_cache.clear()

    def _on_view_changed(self, *_args) -> None:
        """Handle view combo change."""
        if self.file_state.mesh is None:
            return
        self._apply_selected_view()
        self.plotter.render()

    def _on_background_changed(self) -> None:
        """Handle background mode change."""
        self._apply_background()
        if self.file_state.mesh is not None:
            self.plotter.render()

    def _apply_background(self) -> None:
        """Apply selected plot background mode."""
        mode = "gradient"
        if self.background_combo is not None:
            selected = self.background_combo.currentData()
            if selected in {"gradient", "white"}:
                mode = selected

        if mode == "white":
            self.plotter.set_background("white")
        else:
            # top slightly darker than bottom for a subtle light-gray gradient
            self.plotter.set_background("#fbfbfb", top="#dfdfdf")

    def _apply_selected_view(self) -> None:
        """Apply selected view to plotter."""
        if self.file_state.mesh is None:
            return

        view_key = self.view_combo.currentData()
        if view_key == "xy":
            self.plotter.view_xy()
        elif view_key == "yz":
            self.plotter.view_yz()
        elif view_key == "xz":
            self.plotter.view_xz()
        else:
            self.plotter.view_isometric()

    def _set_default_view_for_mesh(self, mesh: pv.UnstructuredGrid | None) -> None:
        """Set default view based on mesh geometry."""
        if mesh is None or mesh.n_points == 0:
            return

        points = np.asarray(mesh.points)
        if points.ndim != 2 or points.shape[1] < 3:
            default_index = 3
        elif np.allclose(points[:, 2], 0.0):
            default_index = 0
        else:
            default_index = 3

        self.view_combo.blockSignals(True)
        self.view_combo.setCurrentIndex(default_index)
        self.view_combo.blockSignals(False)

    def _on_hover_toggled(self, enabled: bool) -> None:
        """Handle hover tooltip toggle."""
        if enabled:
            self.hover_manager.enable()
        else:
            self.hover_manager.disable()

    def _on_discrete_cmap_toggled(self, enabled: bool) -> None:
        """Handle discrete colormap toggle."""
        self.cmap_levels.setEnabled(enabled and self.discrete_cmap.isEnabled())
        if self.file_state.mesh is None:
            return
        self._render()

    def _on_discrete_levels_changed(self) -> None:
        """Handle colormap levels change."""
        if not self.discrete_cmap.isChecked() or self.file_state.mesh is None:
            return
        self._render()

    def _on_color_mapping_changed(self, *_args) -> None:
        """Handle color mapping option changes."""
        if self.file_state.mesh is None:
            return
        self._render()

    def _export_screenshot(self) -> None:
        """Export current view as screenshot."""
        if self.file_state.mesh is None:
            QMessageBox.information(self, "Export", "No mesh loaded.")
            return

        filename, _ = QFileDialog.getSaveFileName(
            self, "Export screenshot", "", "PNG image (*.png)",
        )
        if not filename:
            return

        file_path = Path(filename)
        if file_path.suffix.lower() != ".png":
            file_path = file_path.with_suffix(".png")

        try:
            self.plotter.screenshot(str(file_path))
            self.renderer.set_status_overlay(f"Exported screenshot:\n{file_path.name}")
        except Exception as exc:
            QMessageBox.critical(
                self, "Export failed",
                f"Could not export screenshot:\n{file_path}\n\n{exc}",
            )

    def _export_animation(self) -> None:
        """Export time series as animation."""
        if self.file_state.reader is None or not self.file_state.times:
            QMessageBox.information(self, "Export", "No time-series data loaded.")
            return

        filename, selected_filter = QFileDialog.getSaveFileName(
            self, "Export animation", "",
            "GIF animation (*.gif);;Video (*.mp4)",
        )
        if not filename:
            return

        file_path = Path(filename)
        export_kind = "video" if selected_filter.startswith("Video") else "gif"

        if file_path.suffix.lower() not in (".gif", ".mp4"):
            file_path = file_path.with_suffix(".mp4" if export_kind == "video" else ".gif")
        elif file_path.suffix.lower() == ".mp4":
            export_kind = "video"
        elif file_path.suffix.lower() == ".gif":
            export_kind = "gif"

        saved_step = self.file_state.current_step
        timer_was_active = self._timer.isActive()
        export_succeeded = False
        if timer_was_active:
            self._timer.stop()
            self.btn_play.setText("Play")

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            if export_kind == "video":
                self.plotter.open_movie(str(file_path), framerate=int(self.export_fps.value()))
            else:
                self.plotter.open_gif(str(file_path), fps=int(self.export_fps.value()))
            try:
                for step in range(len(self.file_state.times)):
                    self.time_slider.blockSignals(True)
                    self.time_slider.setValue(step)
                    self.time_slider.blockSignals(False)
                    self._load_mesh_for_step(step, refresh_fields=True)
                    self._render()
                    self.plotter.write_frame()
                export_succeeded = True
            finally:
                movie_writer = getattr(self.plotter, "mwriter", None)
                if movie_writer is not None:
                    movie_writer.close()
                    self.plotter.mwriter = None
        except Exception as exc:
            QMessageBox.critical(
                self, "Export failed",
                f"Could not export animation:\n{file_path}\n\n{exc}",
            )
        finally:
            QApplication.restoreOverrideCursor()
            self.time_slider.blockSignals(True)
            self.time_slider.setValue(saved_step)
            self.time_slider.blockSignals(False)
            self._load_mesh_for_step(saved_step, refresh_fields=True)
            self._render()

            if timer_was_active:
                self._timer.start()
                self.btn_play.setText("Pause")

        if export_succeeded:
            self.renderer.set_status_overlay(
                f"Exported animation:\n{file_path.name}\n"
                f"Frames: {len(self.file_state.times)}\nFPS: {self.export_fps.value()}"
            )

    def _on_mouse_move(self) -> None:
        """Handle mouse move for hover tooltip."""
        if not self.hover_manager.state.enabled:
            return

        if (
            self.renderer.display_mesh is None
            or self.active_location is None
            or self.active_field_name is None
        ):
            self.hover_manager.clear()
            return

        scalars = self.renderer.display_mesh.get_array(
            "__active_scalars__", preference=self.active_location
        )
        if scalars is None:
            self.hover_manager.clear()
            return

        x, y = self.plotter.iren.get_event_position()
        renderer = self.plotter.renderer

        if self.active_location == "point":
            # Use cell picker with ray casting (respects occlusion)
            if self._hover_cell_picker.Pick(float(x), float(y), 0.0, renderer) == 0:
                self.hover_manager.clear()
                return

            cell_id = self._hover_cell_picker.GetCellId()
            if cell_id < 0:
                self.hover_manager.clear()
                return

            # Find nearest point in picked cell to pick position
            pick_position = self._hover_cell_picker.GetPickPosition()
            mesh = self.renderer.display_mesh

            # Get all points of the picked cell
            cell = mesh.extract_cells([cell_id])
            if cell.n_points == 0:
                self.hover_manager.clear()
                return

            # Find nearest point to pick position
            cell_points = np.asarray(cell.points)
            pick_pos_array = np.asarray(pick_position, dtype=float)
            distances = np.linalg.norm(cell_points - pick_pos_array, axis=1)
            nearest_local_idx = int(np.argmin(distances))
            nearest_point_position = cell_points[nearest_local_idx]

            # Map back to original mesh point_id
            # Search for exact match in original mesh points
            original_points = np.asarray(mesh.points)
            point_id = None
            for i, pt in enumerate(original_points):
                if np.allclose(pt, nearest_point_position, atol=1e-6):
                    point_id = i
                    break

            if point_id is None or point_id >= len(scalars):
                self.hover_manager.clear()
                return

            target = ("point", int(point_id))
            if target == self.hover_manager.state.last_target:
                return

            value = float(scalars[point_id])
            position = mesh.points[point_id]
            label_position = self.hover_manager.get_label_position(
                position, mesh
            )
            label = (
                f"point {point_id}\n"
                f"{self.active_field_name} = {HoverTooltipManager.format_value(value)}"
            )
            self.plotter.remove_actor("hover-cell-highlight", render=False)
            self.plotter.add_mesh(
                pv.PolyData(np.asarray([position])),
                name="hover-point-highlight",
                style="points",
                color="magenta",
                point_size=16,
                render_points_as_spheres=True,
                pickable=False,
                reset_camera=False,
                render=False,
            )
        else:
            if self._hover_cell_picker.Pick(float(x), float(y), 0.0, renderer) == 0:
                self.hover_manager.clear()
                return

            cell_id = self._hover_cell_picker.GetCellId()
            if cell_id < 0 or cell_id >= len(scalars):
                self.hover_manager.clear()
                return

            if self.renderer.display_cell_centers is None:
                self.hover_manager.clear()
                return

            target = ("cell", int(cell_id))
            if target == self.hover_manager.state.last_target:
                return

            value = float(scalars[cell_id])
            position = self.renderer.display_cell_centers[cell_id]
            label_position = self.hover_manager.get_label_position(
                position, self.renderer.display_mesh
            )
            label = (
                f"cell {cell_id}\n"
                f"{self.active_field_name} = {HoverTooltipManager.format_value(value)}"
            )

            self.plotter.remove_actor("hover-point-highlight", render=False)

            selected_cell = self.renderer.display_mesh.extract_cells([cell_id])
            self.plotter.add_mesh(
                selected_cell,
                name="hover-cell-highlight",
                style="wireframe",
                color="magenta",
                line_width=5,
                render_lines_as_tubes=True,
                pickable=False,
                reset_camera=False,
                render=False,
            )

        self.hover_manager.state.last_target = target
        self.plotter.remove_actor("hover-tooltip", render=False)

        try:
            self.plotter.add_point_labels(
                np.asarray([label_position]),
                [label],
                name="hover-tooltip",
                show_points=False,
                always_visible=True,
                shape="rounded_rect",
                fill_shape=True,
                font_size=12,
                render=False,
            )
        except TypeError:
            self.plotter.add_point_labels(
                np.asarray([label_position]),
                [label],
                name="hover-tooltip",
                show_points=False,
                always_visible=True,
                font_size=12,
                render=False,
            )

        self.plotter.render()

    def _on_frame_interval_changed(self, value: int) -> None:
        """Handle frame interval change."""
        self._timer.setInterval(int(value))

    def _on_scalar_limits_mode_changed(self) -> None:
        """Handle lock limits change."""
        if self.file_state.mesh is None:
            return
        self._render()

    def _on_warp_vector_changed(self) -> None:
        """Handle warp vector selection."""
        if self.file_state.mesh is None:
            return
        self._render()

    def _populate_warp_vectors(self, preferred_warp_vector: str | None = None) -> None:
        """Populate warp vector combo from mesh."""
        if self.file_state.mesh is None:
            return

        self.warp_vector.blockSignals(True)
        self.warp_vector.clear()
        self.warp_vector.addItem("")

        available_vectors: list[str] = []
        for name in self.file_state.mesh.point_data.keys():
            arr = np.asarray(self.file_state.mesh.point_data[name])
            if arr.ndim > 1 and int(np.prod(arr.shape[1:])) == 3:
                self.warp_vector.addItem(name)
                available_vectors.append(name)

        if preferred_warp_vector and preferred_warp_vector in available_vectors:
            self.warp_vector.setCurrentText(preferred_warp_vector)
        else:
            self.warp_vector.setCurrentIndex(0)

        self.warp_vector.blockSignals(False)

    def _on_time_slider_changed(self, step: int) -> None:
        """Handle time slider change."""
        if self.file_state.reader is None or not self.file_state.times:
            return

        self._load_mesh_for_step(step, refresh_fields=True)
        self._render()

    def _refresh_field_lists_preserve_selection(self, render: bool = True) -> None:
        """Refresh field lists preserving selection."""
        old_location = self.active_location
        old_name = self.active_field_name

        self._refresh_field_lists()

        if old_location == "point" and old_name in self.file_state.mesh.point_data:
            self.point_fields.blockSignals(not render)
            self._select_item_by_name(self.point_fields, old_name)
            self.point_fields.blockSignals(False)
            self._set_active_field("point", old_name, render=render)
            return

        if old_location == "cell" and old_name in self.file_state.mesh.cell_data:
            self.cell_fields.blockSignals(not render)
            self._select_item_by_name(self.cell_fields, old_name)
            self.cell_fields.blockSignals(False)
            self._set_active_field("cell", old_name, render=render)
            return

        self._select_default_field(render=render)

    @staticmethod
    def _select_item_by_name(widget: QListWidget, name: str) -> None:
        """Select item by field name."""
        for idx in range(widget.count()):
            item = widget.item(idx)
            if item.data(Qt.UserRole) == name:
                widget.setCurrentRow(idx)
                return

    def _toggle_playback(self) -> None:
        """Toggle animation playback."""
        if not self.file_state.times or len(self.file_state.times) <= 1:
            return

        if self._timer.isActive():
            self._timer.stop()
            self.btn_play.setText("Play")
        else:
            self._timer.start()
            self.btn_play.setText("Pause")

    def _advance_step(self) -> None:
        """Advance to next time step."""
        if not self.file_state.times:
            return

        next_step = (self.file_state.current_step + 1) % len(self.file_state.times)
        self.time_slider.blockSignals(True)
        self.time_slider.setValue(next_step)
        self.time_slider.blockSignals(False)

        self._load_mesh_for_step(next_step, refresh_fields=True)
        self._render()

    def _compute_global_limits(
        self,
        location: str,
        field_name: str,
        component: int,
    ) -> tuple[float, float] | None:
        """Compute global min/max across all steps."""
        if self.file_state.reader is None or not self.file_state.times:
            return None

        cache_key = (location, field_name, component)
        if cache_key in self._global_limits_cache:
            return self._global_limits_cache[cache_key]

        saved_step = self.file_state.current_step
        data_min = np.inf
        data_max = -np.inf

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            for step, _time_value in enumerate(self.file_state.times):
                mesh = self._get_mesh_for_step(step)

                if location == "point":
                    values = mesh.point_data.get(field_name)
                else:
                    values = mesh.cell_data.get(field_name)

                if values is None:
                    continue

                comp_values = ScalarExtractor.extract_component(values, component)
                if comp_values.size == 0:
                    continue

                finite_values = comp_values[np.isfinite(comp_values)]
                if finite_values.size == 0:
                    continue

                local_min = float(np.min(finite_values))
                local_max = float(np.max(finite_values))
                if local_min < data_min:
                    data_min = local_min
                if local_max > data_max:
                    data_max = local_max

            if not np.isfinite(data_min) or not np.isfinite(data_max):
                return None

            if data_min == data_max:
                eps = max(1.0, abs(data_min)) * 1e-12
                data_min -= eps
                data_max += eps

            limits = (data_min, data_max)
            self._global_limits_cache[cache_key] = limits
            return limits
        finally:
            QApplication.restoreOverrideCursor()
            if self.file_state.times:
                self._load_mesh_for_step(saved_step, refresh_fields=False)

    def _render(self) -> None:
        """Main rendering pipeline."""
        if self.file_state.mesh is None:
            return

        # Update render config from UI
        self.render_config.warp_vector_name = (
            self.warp_vector.currentText()
            if self.warp_vector.count() > 0 else None
        )
        self.render_config.discrete_cmap = self.discrete_cmap.isChecked()
        self.render_config.cmap_levels = int(self.cmap_levels.value())
        self.render_config.lock_limits = self.lock_limits.isChecked()
        self.render_config.log_scale = self.log_scale.isChecked()
        self.render_config.invert_cmap = self.invert_cmap.isChecked()

        # Prepare mesh
        camera_was_initialized = self.renderer.state.camera_is_initialized
        mesh_to_plot = self.renderer.prepare_mesh_for_display(
            self.file_state.mesh,
            warp_vector_name=self.render_config.warp_vector_name,
            warp_factor=1.0,
        )

        # Prepare scalars
        scalars = None
        scalar_name = ""
        if (
            self.active_location is not None
            and self.active_field_name is not None
            and (self.active_location, self.active_field_name) in self.field_meta
        ):
            if self.active_location == "point":
                values = self.file_state.mesh.point_data[self.active_field_name]
            else:
                values = self.file_state.mesh.cell_data[self.active_field_name]

            component = self.component_combo.currentData()
            if component is None:
                component = -1

            scalars = ScalarExtractor.extract_component(values, int(component))
            meta = self.field_meta.get((self.active_location, self.active_field_name))
            component_label = self.component_combo.currentText().strip()
            if meta is not None and meta.kind == "scalar":
                scalar_name = self.active_field_name
            else:
                scalar_name = StatusFormatter.scalar_bar_title(self.active_field_name, component_label)

            self.renderer.attach_scalars(mesh_to_plot, scalars, self.active_location)

        # Compute color limits
        color_limits = None
        if (
            scalars is not None
            and self.render_config.lock_limits
            and self.active_location is not None
            and self.active_field_name is not None
        ):
            component_value = self.component_combo.currentData()
            if component_value is None:
                component_value = -1
            color_limits = self._compute_global_limits(
                self.active_location,
                self.active_field_name,
                int(component_value),
            )

        log_scale = False
        if scalars is not None and self.render_config.log_scale:
            finite_scalars = np.asarray(scalars)
            finite_scalars = finite_scalars[np.isfinite(finite_scalars)]
            log_scale = finite_scalars.size > 0 and bool(np.all(finite_scalars > 0))

        # Render
        self.renderer.render_frame(
            mesh_to_plot,
            scalars=scalars,
            scalar_name=scalar_name,
            color_limits=color_limits,
            location=self.active_location or "point",
            config=self.render_config,
            log_scale=log_scale,
        )

        # Apply preset views only; keep manual 3D camera orientation intact.
        view_key = self.view_combo.currentData()
        if view_key == "3d":
            if not camera_was_initialized:
                self.plotter.view_isometric()
        else:
            self._apply_selected_view()

        if not self.hover_manager.state.enabled:
            self.hover_manager.clear()

        # Update status
        file_name = Path(self.file_state.filename).name if self.file_state.filename else "<none>"

        current_limits = "n/a"
        global_limits = "n/a"
        if scalars is not None:
            current_limits = StatusFormatter.format_min_max(scalars)

            if self.active_location is not None and self.active_field_name is not None:
                component_value = self.component_combo.currentData()
                if component_value is None:
                    component_value = -1
                global_limits_values = self._compute_global_limits(
                    self.active_location,
                    self.active_field_name,
                    int(component_value),
                )
                if global_limits_values is not None:
                    global_limits = StatusFormatter.format_min_max(np.asarray(global_limits_values))

        status_text = StatusFormatter.build_status_text(
            file_name,
            self.file_state.current_step,
            len(self.file_state.times),
            current_limits,
            global_limits,
        )
        self.renderer.set_status_overlay(status_text)
        self.plotter.render()

    def apply_live_frame(self, mesh, times):
        """Integration point for future live streaming from Job.evaluate."""
        self.file_state.mesh = mesh
        self.file_state.times = list(times)
        self.file_state.current_step = min(self.file_state.current_step, max(0, len(self.file_state.times) - 1))
        self._refresh_field_lists_preserve_selection(render=False)
        self._render()

def run(app: QApplication, splash: QWidget | None = None) -> int:
    """Build and run the main window."""
    window = XdmfGuiApp()
    window.showMaximized()
    app.processEvents()
    if splash is not None:
        try:
            splash.finish(window)
        except Exception:
            splash.close()
    return app.exec_()


def main() -> int:
    """Create a local QApplication and run the GUI."""
    app = QApplication(sys.argv)
    return run(app)


if __name__ == "__main__":
    sys.exit(main())
