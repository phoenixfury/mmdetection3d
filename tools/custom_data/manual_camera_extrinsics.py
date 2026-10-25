#!/usr/bin/env python3
"""Interactively adjust camera extrinsics and preview LiDAR projections."""

from __future__ import annotations

import argparse
import base64
import colorsys
import copy
import csv
import io
import json
import tempfile
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, font as tkfont, messagebox, ttk

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFont

from project_lidar import CAMERAS, project


PARAMETERS = (
    ('dx', 'BASELINK dx [m]', -0.50, 0.50, 0.001),
    ('dy', 'BASELINK dy [m]', -0.50, 0.50, 0.001),
    ('dz', 'BASELINK dz [m]', -0.50, 0.50, 0.001),
    ('rx', 'Local Rx / image pitch [deg]', -5.0, 5.0, 0.01),
    ('ry', 'Local Ry / image yaw [deg]', -5.0, 5.0, 0.01),
    ('rz', 'Local Rz / image roll [deg]', -5.0, 5.0, 0.01),
)


def rotation_x(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def rotation_y(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def rotation_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def adjusted_extrinsic(initial: np.ndarray, values: dict[str, float]) -> np.ndarray:
    """Apply BASELINK translation and camera-local fixed-axis rotations."""
    result = initial.copy()
    result[:3, 3] += [values['dx'], values['dy'], values['dz']]
    radians = {key: np.deg2rad(values[key]) for key in ('rx', 'ry', 'rz')}
    # Column vectors encounter Rx, then Ry, then Rz. All rotations are relative
    # to the initial optical camera frame, not the vehicle frame.
    delta = (rotation_z(radians['rz']) @ rotation_y(radians['ry'])
             @ rotation_x(radians['rx']))
    result[:3, :3] = initial[:3, :3] @ delta
    return result


def overlay(image: Image.Image, points: np.ndarray, intrinsic: dict,
            lidar_to_camera: np.ndarray, max_range: float, stride: int,
            radius: int) -> tuple[Image.Image, int]:
    pixels, depth = project(points[::stride], intrinsic, lidar_to_camera,
                            'fisheye', max_range)
    width, height = image.size
    visible = (np.all(np.isfinite(pixels), axis=1)
               & (pixels[:, 0] >= 0) & (pixels[:, 0] < width)
               & (pixels[:, 1] >= 0) & (pixels[:, 1] < height))
    pixels = np.rint(pixels[visible]).astype(np.int32)
    depth = depth[visible]
    normalized = np.clip(depth / max_range, 0, 1)
    colors = [tuple(round(channel * 255) for channel in
                    colorsys.hsv_to_rgb((2/3) * value, 1.0, 1.0))
              for value in normalized]
    order = np.argsort(depth)[::-1]
    result = image.copy()
    draw = ImageDraw.Draw(result)
    for index in order:
        x, y = pixels[index]
        draw.ellipse((x-radius, y-radius, x+radius, y+radius),
                     fill=colors[index])
    return result, len(pixels)


def photo_image(master: tk.Misc, image: Image.Image) -> tk.PhotoImage:
    """Convert a Pillow image for Tk 8.x/9.x without requiring PIL.ImageTk."""
    encoded = io.BytesIO()
    image.save(encoded, format='PNG')
    data = base64.b64encode(encoded.getvalue()).decode('ascii')
    try:
        return tk.PhotoImage(master=master, data=data, format='png')
    except tk.TclError:
        # Some older distro Tk builds reject inline PNG data but accept files.
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(encoded.getvalue())
        try:
            return tk.PhotoImage(master=master, file=str(temporary), format='png')
        finally:
            temporary.unlink(missing_ok=True)


def preview_font() -> ImageFont.ImageFont:
    for candidate in (
            'DejaVuSans-Bold.ttf',
            '/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf',
            '/usr/share/fonts/liberation-sans-fonts/LiberationSans-Bold.ttf'):
        try:
            return ImageFont.truetype(candidate, 42)
        except OSError:
            pass
    return ImageFont.load_default()


def configure_tk_fonts(root: tk.Tk, size: int) -> None:
    """Set explicit font sizes; Tk scaling alone is unreliable across builds."""
    normal = tkfont.Font(root=root, name='ExtrinsicsUIFont',
                         family='Liberation Sans', size=size)
    fixed = tkfont.Font(root=root, name='ExtrinsicsFixedFont',
                        family='Liberation Mono', size=max(12, round(size * 0.85)))
    # Keep Python references as well as Tcl names for older Tk builds.
    root._extrinsics_fonts = (normal, fixed)  # type: ignore[attr-defined]
    root.option_add('*Font', str(normal))
    style = ttk.Style(root)
    style.configure('.', font=str(normal))
    style.configure('TButton', padding=(8, 5))
    style.configure('TEntry', padding=(4, 3))


class ExtrinsicsEditor:
    def __init__(self, root: tk.Tk, args: argparse.Namespace) -> None:
        self.root = root
        self.args = args
        self.converted = args.converted
        self.initial_document = json.loads(args.calibration.read_text())
        self.initial_extrinsics = {
            camera: np.asarray(
                self.initial_document['extrinsics'][camera]['matrix'], dtype=float)
            for camera in CAMERAS
        }
        self.lidar_to_base = np.asarray(
            self.initial_document['extrinsics']['CLIDAR']['matrix'], dtype=float)
        self.values = {
            camera: {name: 0.0 for name, *_ in PARAMETERS} for camera in CAMERAS
        }
        sync = args.sync or self.converted / 'synchronization.csv'
        with sync.open(newline='') as stream:
            self.samples = list(csv.DictReader(stream))
        if not self.samples:
            raise ValueError(f'no synchronized samples in {sync}')
        if not 0 <= args.sample_index < len(self.samples):
            raise IndexError(
                f'--sample-index must be in [0, {len(self.samples)-1}]')

        self.camera = tk.StringVar(value=args.camera)
        self.active_camera = args.camera
        self.sample_index = tk.IntVar(value=args.sample_index)
        self.status = tk.StringVar()
        self.matrix_text = tk.StringVar()
        self.slider_variables = {
            name: tk.DoubleVar(value=0.0) for name, *_ in PARAMETERS
        }
        self.render_job: str | None = None
        self.resize_job: str | None = None
        self.photo: tk.PhotoImage | None = None
        self.full_preview: Image.Image | None = None
        self.display_size = (0, 0)
        self.loaded_sample_index = -1
        self.points: np.ndarray
        self.images: dict[str, Image.Image] = {}

        self._build_ui()
        self.root.bind('<Configure>', self._window_resized)
        self._load_sample()
        self.render()

    def _build_ui(self) -> None:
        self.root.title(
            f'Manual camera extrinsics editor — {self.active_camera} '
            '(independent parameters)')
        self.root.minsize(900, 650)
        controls = ttk.Frame(self.root, padding=8)
        controls.pack(fill=tk.X)

        ttk.Label(controls, text='Camera').grid(row=0, column=0, sticky='w')
        camera_box = ttk.Combobox(
            controls, textvariable=self.camera, values=CAMERAS,
            state='readonly', width=9)
        camera_box.grid(row=0, column=1, padx=(4, 14))
        camera_box.bind('<<ComboboxSelected>>', self._camera_changed)
        ttk.Label(controls, text='(independent parameters per camera)').grid(
            row=0, column=2, sticky='w', padx=(0, 18))

        ttk.Label(controls, text='Sample').grid(row=0, column=3, sticky='w')
        ttk.Button(controls, text='◀', width=3,
                   command=lambda: self._move_sample(-1)).grid(row=0, column=4)
        sample_spin = ttk.Spinbox(
            controls, from_=0, to=len(self.samples)-1,
            textvariable=self.sample_index, width=7, command=self._sample_changed)
        sample_spin.grid(row=0, column=5, padx=3)
        sample_spin.bind('<Return>', self._sample_changed)
        ttk.Button(controls, text='▶', width=3,
                   command=lambda: self._move_sample(1)).grid(row=0, column=6)

        ttk.Button(controls, text='Reset camera', command=self.reset_camera).grid(
            row=0, column=7, padx=(18, 4))
        ttk.Button(controls, text='Reset all', command=self.reset_all).grid(
            row=0, column=8, padx=4)
        ttk.Button(controls, text='Save overlay', command=self.save_overlay).grid(
            row=0, column=9, padx=(18, 4))
        ttk.Button(controls, text='Save calibration', command=self.save).grid(
            row=0, column=10, padx=4)

        sliders = ttk.Frame(self.root, padding=(8, 0, 8, 5))
        sliders.pack(fill=tk.X)
        for row, (name, label, minimum, maximum, resolution) in enumerate(PARAMETERS):
            ttk.Label(sliders, text=label, width=31).grid(
                row=row, column=0, sticky='w')
            scale = tk.Scale(
                sliders, variable=self.slider_variables[name], from_=minimum,
                to=maximum, resolution=resolution, orient=tk.HORIZONTAL,
                showvalue=False, length=780, command=self._slider_changed)
            scale.grid(row=row, column=1, sticky='ew')
            entry = ttk.Entry(sliders, textvariable=self.slider_variables[name],
                              width=10, justify='right')
            entry.grid(row=row, column=2, padx=(10, 2), sticky='ew')
            entry.bind('<Return>', self._slider_changed)
            entry.bind('<FocusOut>', self._slider_changed)
        sliders.columnconfigure(1, weight=1)

        ttk.Label(self.root, textvariable=self.matrix_text,
                  font='ExtrinsicsFixedFont',
                  padding=(8, 0)).pack(fill=tk.X)
        ttk.Label(self.root, textvariable=self.status, padding=(8, 3)).pack(fill=tk.X)
        self.image_label = ttk.Label(self.root, anchor='center')
        self.image_label.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

    def _current_values(self) -> dict[str, float]:
        return {name: variable.get()
                for name, variable in self.slider_variables.items()}

    def _store_current(self) -> None:
        self.values[self.active_camera] = self._current_values()

    def _camera_changed(self, _event: object = None) -> None:
        self._store_current()
        self.active_camera = self.camera.get()
        self.root.title(
            f'Manual camera extrinsics editor — {self.active_camera} '
            '(independent parameters)')
        camera_values = self.values[self.active_camera]
        for name, variable in self.slider_variables.items():
            variable.set(camera_values[name])
        self.schedule_render()

    def _slider_changed(self, _value: str = '') -> None:
        self._store_current()
        self.schedule_render()

    def _move_sample(self, delta: int) -> None:
        value = max(0, min(len(self.samples)-1,
                           self.sample_index.get() + delta))
        self.sample_index.set(value)
        self._sample_changed()

    def _sample_changed(self, _event: object = None) -> None:
        try:
            value = int(self.sample_index.get())
        except (tk.TclError, ValueError):
            return
        self.sample_index.set(max(0, min(len(self.samples)-1, value)))
        self._load_sample()
        self.schedule_render()

    def _load_sample(self) -> None:
        index = self.sample_index.get()
        if index == self.loaded_sample_index:
            return
        sample = self.samples[index]
        raw = np.fromfile(self.converted / sample['lidar_path'], dtype='<f4')
        if raw.size % 4:
            raise ValueError(f"{sample['lidar_path']} is not float32 XYZI")
        self.points = raw.reshape(-1, 4)
        self.images = {
            camera: Image.open(
                self.converted / sample[f'{camera.lower()}_path']).convert('RGB')
            for camera in CAMERAS
        }
        self.loaded_sample_index = index

    def schedule_render(self) -> None:
        if self.render_job is not None:
            self.root.after_cancel(self.render_job)
        self.render_job = self.root.after(100, self.render)

    def _matrix(self, camera: str, original: bool = False) -> np.ndarray:
        if original:
            return self.initial_extrinsics[camera]
        return adjusted_extrinsic(self.initial_extrinsics[camera], self.values[camera])

    def _render_overlay(self, original: bool) -> tuple[Image.Image, int]:
        camera = self.camera.get()
        camera_to_base = self._matrix(camera, original)
        lidar_to_camera = np.linalg.inv(camera_to_base) @ self.lidar_to_base
        return overlay(
            self.images[camera], self.points,
            self.initial_document['intrinsics'][camera], lidar_to_camera,
            self.args.max_range, self.args.point_stride, self.args.point_radius)

    def render(self) -> None:
        self.render_job = None
        self._store_current()
        original, original_count = self._render_overlay(True)
        adjusted, adjusted_count = self._render_overlay(False)
        difference = np.asarray(ImageChops.difference(original, adjusted))
        changed_pixels = np.count_nonzero(np.any(difference > 8, axis=2))
        changed_percent = 100.0 * changed_pixels / (original.width * original.height)
        draw_original = ImageDraw.Draw(original)
        draw_adjusted = ImageDraw.Draw(adjusted)
        label_font = preview_font()
        draw_original.text((20, 20), 'INITIAL', fill='white', stroke_width=2,
                           stroke_fill='black', font=label_font)
        draw_adjusted.text((20, 20), 'ADJUSTED', fill='white', stroke_width=2,
                           stroke_fill='black', font=label_font)
        combined = Image.new('RGB', (original.width * 2, original.height))
        combined.paste(original, (0, 0))
        combined.paste(adjusted, (original.width, 0))
        self.full_preview = combined
        # The dimensions usually remain constant while slider values change,
        # but the pixels do not. Force Tk to receive the newly rendered image.
        self._display_preview(force=True)

        matrix = self._matrix(self.camera.get())
        self.matrix_text.set(
            'Adjusted T_BASELINK_CAMERA:\n' +
            '\n'.join('  '.join(f'{value:+.8f}' for value in row)
                      for row in matrix))
        timestamp = self.samples[self.sample_index.get()]['timestamp_ns']
        modified = [camera for camera in CAMERAS
                    if any(abs(value) > 1e-12
                           for value in self.values[camera].values())]
        modified_text = ','.join(modified) if modified else 'none'
        self.status.set(
            f'camera={self.camera.get()}  sample={self.sample_index.get()}/'
            f'{len(self.samples)-1}  timestamp={timestamp}  '
            f'visible points: initial={original_count}, adjusted={adjusted_count}  '
            f'overlay change={changed_percent:.2f}%  '
            f'modified cameras={modified_text}')

    def _window_resized(self, event: tk.Event) -> None:
        if event.widget is not self.root or self.full_preview is None:
            return
        if self.resize_job is not None:
            self.root.after_cancel(self.resize_job)
        self.resize_job = self.root.after(80, self._display_preview)

    def _display_preview(self, force: bool = False) -> None:
        self.resize_job = None
        if self.full_preview is None:
            return
        self.root.update_idletasks()
        available_width = max(320, self.image_label.winfo_width() - 8)
        available_height = max(220, self.image_label.winfo_height() - 8)
        source_width, source_height = self.full_preview.size
        ratio = min(available_width / source_width,
                    available_height / source_height)
        target = (max(1, round(source_width * ratio)),
                  max(1, round(source_height * ratio)))
        if target == self.display_size and not force:
            return
        preview = self.full_preview
        if target != preview.size:
            preview = preview.resize(target, Image.Resampling.LANCZOS)
        self.photo = photo_image(self.root, preview)
        self.image_label.configure(image=self.photo)
        self.display_size = target

    def reset_camera(self) -> None:
        for variable in self.slider_variables.values():
            variable.set(0.0)
        self._store_current()
        self.schedule_render()

    def reset_all(self) -> None:
        for camera in CAMERAS:
            self.values[camera] = {name: 0.0 for name, *_ in PARAMETERS}
        for variable in self.slider_variables.values():
            variable.set(0.0)
        self.active_camera = self.camera.get()
        self.schedule_render()

    def save_overlay(self) -> None:
        destination = filedialog.asksaveasfilename(
            title='Save adjusted overlay', defaultextension='.jpg',
            filetypes=[('JPEG image', '*.jpg'), ('PNG image', '*.png')])
        if not destination:
            return
        adjusted, _ = self._render_overlay(False)
        adjusted.save(destination, quality=95)
        self.status.set(f'saved overlay: {destination}')

    def save(self) -> None:
        self._store_current()
        default = self.args.output or self.args.calibration.with_name(
            self.args.calibration.stem + '_manual.json')
        destination = filedialog.asksaveasfilename(
            title='Save adjusted calibration', initialdir=str(default.parent),
            initialfile=default.name, defaultextension='.json',
            filetypes=[('JSON calibration', '*.json')])
        if not destination:
            return
        document = copy.deepcopy(self.initial_document)
        adjustments = {}
        for camera in CAMERAS:
            matrix = self._matrix(camera)
            document['extrinsics'][camera]['matrix'] = matrix.tolist()
            document['lidar_to_camera'][camera] = (
                np.linalg.inv(matrix) @ self.lidar_to_base).tolist()
            adjustments[camera] = self.values[camera]
        document['manual_extrinsics_adjustments'] = {
            'starting_calibration': str(self.args.calibration),
            'translation_frame': 'BASELINK',
            'rotation_frame': 'initial camera optical frame',
            'rotation_order': 'Rx then Ry then Rz',
            'per_camera': adjustments,
        }
        Path(destination).write_text(json.dumps(document, indent=2) + '\n')
        self.status.set(f'saved calibration: {destination}')
        messagebox.showinfo('Calibration saved', destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('converted', type=Path,
                        help='converted directory containing images, LiDAR and synchronization.csv')
    parser.add_argument('calibration', type=Path,
                        help='initial calibration JSON; this file is never modified')
    parser.add_argument('--sync', type=Path)
    parser.add_argument('--camera', choices=CAMERAS, default='ARC_F')
    parser.add_argument('--sample-index', type=int, default=0)
    parser.add_argument('--output', type=Path,
                        help='suggested destination when saving calibration')
    parser.add_argument('--max-range', type=float, default=80.0)
    parser.add_argument('--point-stride', type=int, default=2,
                        help='draw every Nth point for faster preview')
    parser.add_argument('--point-radius', type=int, default=2)
    parser.add_argument('--ui-scale', type=float, default=1.4,
                        help='font and control scale factor (default: 1.4)')
    parser.add_argument('--font-size', type=int, default=16,
                        help='explicit UI font size in points (default: 16)')
    args = parser.parse_args()
    if args.point_stride < 1:
        parser.error('--point-stride must be at least 1')
    if args.ui_scale <= 0:
        parser.error('--ui-scale must be positive')
    if args.font_size < 8:
        parser.error('--font-size must be at least 8')
    root = tk.Tk()
    current_scaling = float(root.tk.call('tk', 'scaling'))
    root.tk.call('tk', 'scaling', current_scaling * args.ui_scale)
    configure_tk_fonts(root, args.font_size)
    try:
        root.state('zoomed')
    except tk.TclError:
        width = round(root.winfo_screenwidth() * 0.9)
        height = round(root.winfo_screenheight() * 0.9)
        root.geometry(f'{width}x{height}+0+0')
    ExtrinsicsEditor(root, args)
    root.mainloop()


if __name__ == '__main__':
    main()
