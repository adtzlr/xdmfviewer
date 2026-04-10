# xdmfviewer

A standalone desktop viewer for XDMF time-series results based on PyVista and Qt.

## Features

<video src="https://github.com/user-attachments/assets/621a77e2-c160-4be3-a67b-ecdaabe35109" width="100%" controls></video>

- Interactive visualization of XDMF time steps
- Point and cell field selection
- Scalar component selection (scalar, vector, tensor)
- Warp-by-vector rendering (manual opt-in)
- Animation playback and export (GIF/MP4)
- Screenshot export
- Hover tooltip inspection

## Installation

Install from PyPI:

```bash
pip install xdmfviewer[qt]
```

For development, clone the repository and install in editable mode:

```bash
git clone https://github.com/adtzlr/xdmfviewer.git
cd xdmfviewer
pip install --editable ".[qt,dev]"
```

### Windows release

Tagged releases publish a ZIP archive as a GitHub release asset.
Download the `.zip` file from the release page, extract it, and start
`xdmfviewer.exe` from the extracted folder.

The release workflow is triggered by version tags such as `v0.1.0`.

## Usage

After installation:

```bash
xdmfviewer
```

Or via module:

```bash
python -m xdmfviewer
```

Then open an `.xdmf` file from the GUI.

## Dependencies

Core runtime dependencies are declared in `pyproject.toml`:

- `numpy`
- `pyvista`
- `pyvistaqt`
- `qtpy`

You also need a Qt binding, for example one of:

- `PySide6`
- `PyQt6`
- `PyQt5`

The recommended install extra for this project is `qt`, which currently
pulls in `PySide6`.

## Acknowledgments

This project was developed with assistance from Claude (Anthropic) and GitHub Copilot.

## License

This project is licensed under the GNU Lesser General Public License v3 or later (LGPL-3.0-or-later). See [LICENSE](LICENSE).
