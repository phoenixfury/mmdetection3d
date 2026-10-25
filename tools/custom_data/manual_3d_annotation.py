#!/usr/bin/env python3
"""Edit fused-object 3D annotations using synchronized cameras and LiDAR.

The input is the ``fisheye3dod_infos_all.pkl`` produced by
prepare_fused_groundtruth.py. Each frame is saved independently as JSON, so an
interrupted labeling session is resumable. Export creates MMDetection3D PKLs.
"""

from __future__ import annotations

import argparse
import copy
import json
import tkinter as tk
from datetime import datetime, timezone
from pathlib import Path
from tkinter import font as tkfont, messagebox, ttk

import cv2
import mmengine
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from ocamcamera import OcamCamera


CLASSES = ('Car', 'Van', 'Truck', 'Bus', 'Pedestrian', 'Cyclist')
COLORS = {
    'Car': '#00bcd4', 'Van': '#2979ff', 'Truck': '#ff9100',
    'Bus': '#ff1744', 'Pedestrian': '#76ff03', 'Cyclist': '#d500f9',
}
DEFAULT_DIMS = {
    'Car': (4.3, 1.8, 1.6), 'Van': (4.8, 2.0, 2.1),
    'Truck': (8.0, 2.5, 3.2), 'Bus': (11.0, 2.6, 3.3),
    'Pedestrian': (0.6, 0.6, 1.7), 'Cyclist': (1.8, 0.7, 1.7),
}
PARAMETERS = (
    ('x', 'Center X [m]'), ('y', 'Center Y [m]'), ('z', 'Center Z [m]'),
    ('length', 'Length [m]'), ('width', 'Width [m]'),
    ('height', 'Height [m]'), ('yaw', 'Yaw [deg]'),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path,
                        help='prepared 800x800 Fisheye3DOD dataset directory')
    parser.add_argument('--info', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--start-index', type=int, default=0)
    parser.add_argument('--point-stride', type=int, default=3)
    parser.add_argument('--point-size', type=float, default=0.9)
    parser.add_argument('--range', type=float, default=35.0)
    parser.add_argument('--font-size', type=int, default=14)
    parser.add_argument('--ui-scale', type=float, default=1.25)
    parser.add_argument('--target-ocam', type=Path,
                        default=Path('data/Fisheye3DODdataset/calib_results.txt'))
    parser.add_argument('--export-only', action='store_true',
                        help='export saved manual frames without opening the UI')
    return parser.parse_args()


def configure_fonts(root: tk.Tk, size: int) -> None:
    normal = tkfont.Font(root=root, name='AnnotationUIFont',
                         family='Liberation Sans', size=size)
    fixed = tkfont.Font(root=root, name='AnnotationFixedFont',
                        family='Liberation Mono', size=max(11, size - 2))
    root._annotation_fonts = (normal, fixed)  # type: ignore[attr-defined]
    root.option_add('*Font', str(normal))
    style = ttk.Style(root)
    style.configure('.', font=str(normal))
    style.configure('TButton', padding=(8, 6))
    style.configure('TEntry', padding=(5, 4))


def load_points(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.float32)
    if len(raw) % 4:
        raise ValueError(f'{path}: expected float32 XYZI')
    points = raw.reshape(-1, 4)[:, :3]
    valid = np.isfinite(points).all(axis=1)
    valid &= np.linalg.norm(points, axis=1) > 0.1
    return points[valid]


def box_corners(box: np.ndarray) -> np.ndarray:
    half_l, half_w = box[3] / 2, box[4] / 2
    local = np.array([[half_l, half_w], [half_l, -half_w],
                      [-half_l, -half_w], [-half_l, half_w],
                      [half_l, half_w]])
    cosine, sine = np.cos(box[6]), np.sin(box[6])
    rotation = np.array([[cosine, -sine], [sine, cosine]])
    return local @ rotation.T + box[:2]


def box_corners_3d(box: np.ndarray) -> np.ndarray:
    half_l, half_w, half_h = box[3:6] / 2
    local = np.array([
        [sx * half_l, sy * half_w, sz * half_h]
        for sz in (-1, 1) for sy in (-1, 1) for sx in (-1, 1)])
    cosine, sine = np.cos(box[6]), np.sin(box[6])
    rotation = np.array([[cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1]])
    return local @ rotation.T + box[:3]


BOX_EDGES = ((0, 1), (0, 2), (1, 3), (2, 3),
             (4, 5), (4, 6), (5, 7), (6, 7),
             (0, 4), (1, 5), (2, 6), (3, 7))


def annotations_from_boxes(boxes: list[dict]) -> dict:
    return {
        'gt_boxes': np.asarray([item['box'] for item in boxes],
                               dtype=np.float32).reshape(-1, 7),
        'gt_names': np.asarray([item['class'] for item in boxes]),
        'gt_mesh': np.asarray(
            [f"manual.{item['class'].lower()}" for item in boxes]),
        'gt_ids': np.asarray([item.get('id', -1) for item in boxes],
                             dtype=np.int64),
    }


def export_manual_annotations(info_path: Path, output: Path) -> tuple[int, int, int]:
    info = mmengine.load(info_path)
    frames_dir = output / 'frames'
    saved = {path.stem: path for path in frames_dir.glob('*.json')}
    samples = {str(item['token']): item for item in info['data_list']}
    unknown = sorted(set(saved) - set(samples))
    if unknown:
        raise ValueError(f'{len(unknown)} saved annotation tokens are absent from {info_path}')

    exported = []
    for token, sample in samples.items():
        path = saved.get(token)
        if path is None:
            continue
        boxes = json.loads(path.read_text())['boxes']
        item = copy.deepcopy(sample)
        item['annos'] = annotations_from_boxes(boxes)
        exported.append(item)

    train = exported
    automatic_val = mmengine.load(
        info_path.parent / 'fisheye3dod_infos_val.pkl')['data_list']
    val = [copy.deepcopy(item) for item in automatic_val
           if str(item['token']) not in saved]

    metainfo = dict(info['metainfo'])
    metainfo['dataset'] = 'custom_manual_3d_annotations'
    output.mkdir(parents=True, exist_ok=True)
    mmengine.dump({'metainfo': metainfo, 'data_list': exported},
                  output / 'fisheye3dod_infos_manual_all.pkl')
    mmengine.dump({'metainfo': metainfo, 'data_list': train},
                  output / 'fisheye3dod_infos_manual_train.pkl')
    mmengine.dump({'metainfo': metainfo, 'data_list': val},
                  output / 'fisheye3dod_infos_manual_val.pkl')
    return len(exported), len(train), len(val)


class AnnotationEditor:
    def __init__(self, root: tk.Tk, args: argparse.Namespace) -> None:
        self.root, self.args = root, args
        self.dataset = args.dataset.resolve()
        self.info_path = (args.info or self.dataset / 'groundtruth' /
                          'fisheye3dod_infos_all.pkl').resolve()
        self.output = (args.output or self.dataset / 'manual_annotations').resolve()
        self.frames_dir = self.output / 'frames'
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.info = mmengine.load(self.info_path)
        self.ocam = OcamCamera(filename=str(args.target_ocam), fov=220.0)
        self.samples = self.info['data_list']
        if not self.samples:
            raise ValueError(f'no samples in {self.info_path}')
        if not 0 <= args.start_index < len(self.samples):
            raise IndexError('--start-index is outside the dataset')

        self.index = tk.IntVar(value=args.start_index)
        self.jump_index = tk.StringVar(value=str(args.start_index))
        self.class_name = tk.StringVar(value='Car')
        self.status = tk.StringVar()
        self.frame_label = tk.StringVar()
        self.mode_label = tk.StringVar(value='Mode: select/move')
        self.variables = {name: tk.StringVar() for name, _ in PARAMETERS}
        self.delta_variables = {
            'dx': tk.StringVar(value='0.000'),
            'dy': tk.StringVar(value='0.000'),
            'dyaw': tk.StringVar(value='0.000'),
        }
        self.boxes: list[dict] = []
        self.initial_boxes: list[dict] = []
        self.points = np.empty((0, 3), dtype=np.float32)
        self.images: list[np.ndarray] = []
        self.camera_data: list[dict] = []
        self.selected: int | None = None
        self.add_mode = False
        self.add_points: list[tuple[float, float]] = []
        self.deleted_ids: set[int] = set()
        self.next_manual_id = -1
        self.dragging = False
        self.drag_offset = np.zeros(2)
        self.dirty = False
        self.saved_tokens = {path.stem for path in self.frames_dir.glob('*.json')}

        self._build_ui()
        self._bind_keys()
        self.load_frame(args.start_index)
        self.root.protocol('WM_DELETE_WINDOW', self.close)

    def _build_ui(self) -> None:
        self.root.title('Manual 3D annotation — synchronized cameras + LiDAR')
        self.root.minsize(1200, 800)
        try:
            self.root.state('zoomed')
        except tk.TclError:
            pass

        main = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True)
        plot_frame = ttk.Frame(main)
        controls = ttk.Frame(main, padding=10, width=360)
        main.add(plot_frame, weight=5)
        main.add(controls, weight=1)

        self.figure = Figure(figsize=(15, 9), dpi=90, constrained_layout=True)
        grid = self.figure.add_gridspec(
            2, 4, width_ratios=(1.35, 1.35, 1.0, 1.0))
        self.camera_axes = [
            self.figure.add_subplot(grid[0, 0]),
            self.figure.add_subplot(grid[0, 1]),
            self.figure.add_subplot(grid[1, 0]),
            self.figure.add_subplot(grid[1, 1]),
        ]
        self.bev_axis = self.figure.add_subplot(grid[:, 2:])
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect('button_press_event', self.on_press)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.canvas.mpl_connect('button_release_event', self.on_release)

        ttk.Label(controls, textvariable=self.frame_label,
                  font='AnnotationFixedFont').pack(fill=tk.X, pady=(0, 8))
        navigation = ttk.Frame(controls)
        navigation.pack(fill=tk.X)
        ttk.Button(navigation, text='◀ Previous', command=lambda: self.move(-1)).pack(
            side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(navigation, text='Next frame ▶', command=lambda: self.move(1)).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=(5, 0))
        jump = ttk.Frame(controls)
        jump.pack(fill=tk.X, pady=6)
        ttk.Label(jump, text='Frame').pack(side=tk.LEFT)
        spin = ttk.Spinbox(jump, from_=0, to=len(self.samples)-1,
                           textvariable=self.jump_index, width=9)
        spin.pack(side=tk.LEFT, padx=5)
        spin.bind('<Return>', lambda _event: self.jump())
        ttk.Button(jump, text='Go', command=self.jump).pack(side=tk.LEFT)

        ttk.Separator(controls).pack(fill=tk.X, pady=8)
        ttk.Label(controls, textvariable=self.mode_label).pack(fill=tk.X)
        ttk.Label(controls, text='Class').pack(anchor='w', pady=(8, 2))
        class_box = ttk.Combobox(controls, textvariable=self.class_name,
                                 values=CLASSES, state='readonly')
        class_box.pack(fill=tk.X)
        class_box.bind('<<ComboboxSelected>>', self.class_changed)

        parameter_frame = ttk.Frame(controls)
        parameter_frame.pack(fill=tk.X, pady=8)
        for row, (name, label) in enumerate(PARAMETERS):
            ttk.Label(parameter_frame, text=label).grid(row=row, column=0, sticky='w')
            entry = ttk.Entry(parameter_frame, textvariable=self.variables[name],
                              width=13, justify='right')
            entry.grid(row=row, column=1, sticky='ew', padx=(8, 0), pady=2)
            entry.bind('<Return>', self.apply_fields)
            entry.bind('<space>', self.space_pressed)
        parameter_frame.columnconfigure(1, weight=1)
        ttk.Button(controls, text='Apply edited values',
                   command=self.apply_fields).pack(fill=tk.X)

        ttk.Label(controls, text='Per-frame movement when advancing',
                  padding=(0, 10, 0, 2)).pack(anchor='w')
        delta_frame = ttk.Frame(controls)
        delta_frame.pack(fill=tk.X)
        for row, (name, label) in enumerate((
                ('dx', 'Next ΔX [m]'), ('dy', 'Next ΔY [m]'),
                ('dyaw', 'Next Δyaw [deg]'))):
            ttk.Label(delta_frame, text=label).grid(row=row, column=0, sticky='w')
            entry = ttk.Entry(delta_frame, textvariable=self.delta_variables[name],
                              width=11, justify='right')
            entry.grid(row=row, column=1, sticky='ew', padx=(8, 0), pady=2)
            entry.bind('<Return>', self.apply_delta)
            entry.bind('<space>', self.space_pressed)
        delta_frame.columnconfigure(1, weight=1)
        ttk.Button(controls, text='Apply movement to selected box',
                   command=self.apply_delta).pack(fill=tk.X)
        delta_buttons = ttk.Frame(controls)
        delta_buttons.pack(fill=tk.X, pady=(4, 0))
        for column, (label, dx, dy) in enumerate((
                ('+X', 0.10, 0.0), ('−X', -0.10, 0.0),
                ('+Y', 0.0, 0.10), ('−Y', 0.0, -0.10))):
            ttk.Button(delta_buttons, text=label,
                       command=lambda x=dx, y=dy: self.nudge_delta(x, y)).grid(
                           row=0, column=column, sticky='ew')
            delta_buttons.columnconfigure(column, weight=1)

        editing = ttk.Frame(controls)
        editing.pack(fill=tk.X, pady=8)
        ttk.Button(editing, text='Add box (A)', command=self.begin_add).grid(
            row=0, column=0, sticky='ew')
        ttk.Button(editing, text='Duplicate', command=self.duplicate).grid(
            row=0, column=1, sticky='ew', padx=(5, 0))
        ttk.Button(editing, text='Delete (Del)', command=self.delete).grid(
            row=1, column=0, sticky='ew', pady=(5, 0))
        ttk.Button(editing, text='Reset frame', command=self.reset_frame).grid(
            row=1, column=1, sticky='ew', padx=(5, 0), pady=(5, 0))
        editing.columnconfigure((0, 1), weight=1)

        ttk.Separator(controls).pack(fill=tk.X, pady=8)
        ttk.Button(controls, text='Save frame (Ctrl+S)',
                   command=self.save_frame).pack(fill=tk.X)
        self.save_next_button = ttk.Button(
            controls, text='Save and next (Space)', command=self.save_and_next)
        self.save_next_button.pack(fill=tk.X, pady=5)
        ttk.Button(controls, text='Export training PKLs',
                   command=self.export).pack(fill=tk.X)
        ttk.Label(controls, textvariable=self.status, wraplength=340,
                  justify=tk.LEFT).pack(fill=tk.X, pady=(10, 0))

    def _bind_keys(self) -> None:
        self.root.bind('<Control-s>', lambda _event: self.save_frame())
        self.root.bind('<space>', self.space_pressed)
        self.root.bind('<Delete>', lambda _event: self.delete())
        self.root.bind('a', lambda _event: self.begin_add())
        self.root.bind('<Prior>', lambda _event: self.move(-1))
        self.root.bind('<Next>', lambda _event: self.move(1))
        self.root.bind('<Escape>', lambda _event: self.cancel_mode())

    def space_pressed(self, event: object) -> str | None:
        # A focused ttk button already handles Space through its class binding.
        # Running the global shortcut too would save/advance twice.
        if event.widget.winfo_class() == 'TButton':
            return None
        self.save_and_next()
        return 'break'

    @property
    def sample(self) -> dict:
        return self.samples[self.index.get()]

    @property
    def token(self) -> str:
        return str(self.sample['token'])

    def frame_path(self, token: str | None = None) -> Path:
        return self.frames_dir / f'{token or self.token}.json'

    def boxes_from_info(self, sample: dict) -> list[dict]:
        annos = sample.get('annos', {})
        boxes = np.asarray(annos.get('gt_boxes', []), dtype=float).reshape(-1, 7)
        names = annos.get('gt_names', [])
        ids = annos.get('gt_ids', np.arange(len(boxes)))
        return [{'class': str(name), 'box': box.tolist(), 'id': int(track_id),
                 'source': 'automatic_fused_lidar',
                 'next_delta': [0.0, 0.0, 0.0]}
                for name, box, track_id in zip(names, boxes, ids)]

    @staticmethod
    def normalize_box(item: dict) -> dict:
        item = copy.deepcopy(item)
        item.setdefault('id', -1)
        item.setdefault('source', 'manual')
        item.setdefault('next_delta', [0.0, 0.0, 0.0])
        return item

    @staticmethod
    def same_manual_box(first: dict, second: dict) -> bool:
        a, b = np.asarray(first['box'], dtype=float), np.asarray(second['box'], dtype=float)
        return (first['class'] == second['class'] and
                np.linalg.norm(a[:2] - b[:2]) < 0.75 and
                np.linalg.norm(a[3:6] - b[3:6]) < 1.0)

    def load_frame(self, index: int, propagated: list[dict] | None = None,
                   deleted_ids: set[int] | None = None) -> None:
        self.index.set(max(0, min(len(self.samples)-1, index)))
        self.jump_index.set(str(self.index.get()))
        sample = self.sample
        self.points = load_points((self.dataset / sample['pc_info']['pts_path']).resolve())
        camera_info = sample['cam_info']['cam_fisheye']
        self.images = []
        self.camera_data = []
        for name in ('fisheye_camera_front', 'fisheye_camera_left',
                     'fisheye_camera_right', 'fisheye_camera_rear'):
            from PIL import Image
            self.images.append(np.asarray(Image.open(
                self.dataset / camera_info[name]['cam_path']).convert('RGB')))
            self.camera_data.append(camera_info[name])
        self.initial_boxes = self.boxes_from_info(sample)
        path = self.frame_path()
        propagation_changed = False
        incoming_deleted = set(deleted_ids or ())
        if path.exists():
            document = json.loads(path.read_text())
            saved_boxes = [self.normalize_box(item) for item in document['boxes']]
            self.deleted_ids = set(document.get('deleted_ids', [])) | incoming_deleted
            if propagated is not None:
                # The current frame is authoritative. Start with every carried
                # box, then preserve genuinely additional boxes from the saved
                # target frame. Negative IDs from older sessions may collide,
                # so geometry—not ID alone—deduplicates manual boxes.
                self.boxes = [self.normalize_box(item) for item in propagated
                              if int(item.get('id', -1)) not in self.deleted_ids]
                for saved in saved_boxes:
                    track_id = int(saved['id'])
                    if track_id in self.deleted_ids:
                        continue
                    if track_id >= 0:
                        duplicate = any(int(item['id']) == track_id
                                        for item in self.boxes)
                    else:
                        same_id = [item for item in self.boxes
                                   if int(item['id']) == track_id]
                        duplicate = any(self.same_manual_box(item, saved)
                                        for item in same_id)
                    if not duplicate:
                        self.boxes.append(saved)
                propagation_changed = True
            else:
                self.boxes = [item for item in saved_boxes
                              if int(item['id']) not in self.deleted_ids]
                propagation_changed = len(self.boxes) != len(saved_boxes)
        elif propagated is not None:
            self.deleted_ids = incoming_deleted
            self.boxes = [self.normalize_box(item) for item in propagated]
            propagation_changed = True
        else:
            self.boxes = copy.deepcopy(self.initial_boxes)
            self.deleted_ids = set()

        # Merge new positive-ID fused tracks for this timestamp.
        if propagated is not None:
            existing_positive = {int(item['id']) for item in self.boxes
                                 if int(item['id']) >= 0}
            for item in self.initial_boxes:
                track_id = int(item['id'])
                if track_id not in existing_positive and track_id not in self.deleted_ids:
                    self.boxes.append(item)
                    existing_positive.add(track_id)

        # Allocate negative IDs from a frame-specific range and repair legacy
        # collisions, preventing one manual box from suppressing another.
        candidate = -(self.index.get() + 1) * 100000 - 1
        used: set[int] = set()
        for item in self.boxes:
            track_id = int(item['id'])
            if track_id < 0 and track_id in used:
                while candidate in used:
                    candidate -= 1
                item['id'] = candidate
                track_id = candidate
                candidate -= 1
                propagation_changed = True
            used.add(track_id)
        while candidate in used:
            candidate -= 1
        self.next_manual_id = candidate
        self.selected = 0 if self.boxes else None
        self.dirty = propagation_changed
        self.cancel_mode(render=False)
        self.sync_fields()
        self.render()

    def render(self) -> None:
        camera_titles = ('FRONT', 'LEFT', 'RIGHT', 'REAR')
        for axis, image, title, camera in zip(
                self.camera_axes, self.images, camera_titles, self.camera_data):
            axis.clear(); axis.imshow(image); axis.set_title(title); axis.axis('off')
            self.draw_camera_boxes(axis, camera, image.shape[1], image.shape[0])
        axis = self.bev_axis
        axis.clear()
        limit = self.args.range
        visible = ((np.abs(self.points[:, 0]) <= limit) &
                   (np.abs(self.points[:, 1]) <= limit))
        draw = self.points[visible][::max(1, self.args.point_stride)]
        axis.scatter(draw[:, 0], draw[:, 1], s=self.args.point_size, c=draw[:, 2],
                     cmap='gray', vmin=-3, vmax=2, alpha=0.7)
        for index, item in enumerate(self.boxes):
            box = np.asarray(item['box'], dtype=float)
            polygon = box_corners(box)
            selected = index == self.selected
            color = COLORS.get(item['class'], '#ffffff')
            axis.plot(polygon[:, 0], polygon[:, 1], color=color,
                      linewidth=3.2 if selected else 1.7)
            axis.fill(polygon[:, 0], polygon[:, 1], color=color,
                      alpha=0.14 if selected else 0.045)
            axis.plot(box[0], box[1], 'o', color=color,
                      markersize=7 if selected else 3)
            axis.text(box[0], box[1], f"{index}: {item['class']}", color=color,
                      fontsize=9, weight='bold' if selected else 'normal')
        if self.add_points:
            clicks = np.asarray(self.add_points)
            axis.plot(clicks[:, 0], clicks[:, 1], 'o-', color='yellow',
                      markersize=9, linewidth=1.5)
        axis.set(xlim=(-limit, limit), ylim=(-limit, limit), aspect='equal',
                 xlabel='LiDAR X [m]', ylabel='LiDAR Y [m]',
                 title='BEV — click a box to select; drag its center to move')
        axis.grid(alpha=0.15)
        marker = '*' if self.dirty else ''
        saved = 'saved' if self.token in self.saved_tokens else 'automatic draft'
        self.frame_label.set(
            f'Frame {self.index.get()} / {len(self.samples)-1}{marker}\n'
            f'Token {self.token}\n{len(self.boxes)} boxes · {saved}\n'
            f'Progress {len(self.saved_tokens)} / {len(self.samples)}')
        self.canvas.draw_idle()

    def draw_camera_boxes(self, axis: object, camera: dict,
                          width: int, height: int) -> None:
        lidar_to_camera = np.asarray(camera['lidar2cam'], dtype=float)
        for index, item in enumerate(self.boxes):
            corners = box_corners_3d(np.asarray(item['box'], dtype=float))
            homogeneous = np.column_stack((corners, np.ones(8)))
            camera_xyz = (lidar_to_camera @ homogeneous.T)[:3]
            pixels = self.ocam.world2cam(camera_xyz).T
            color = COLORS.get(item['class'], '#ffffff')
            selected = index == self.selected
            for first, second in BOX_EDGES:
                segment = pixels[[first, second]]
                valid = np.isfinite(segment).all() and (segment >= 0).all()
                valid &= (segment[:, 0] < width).all() and (segment[:, 1] < height).all()
                # Do not draw a seam-crossing edge across the whole fisheye image.
                valid &= np.linalg.norm(segment[0] - segment[1]) < width * 0.6
                if valid:
                    axis.plot(segment[:, 0], segment[:, 1], color=color,
                              linewidth=3.5 if selected else 1.8,
                              alpha=1.0 if selected else 0.72)
            visible = ((pixels[:, 0] >= 0) & (pixels[:, 0] < width) &
                       (pixels[:, 1] >= 0) & (pixels[:, 1] < height))
            if np.any(visible):
                anchor = pixels[np.flatnonzero(visible)[0]]
                axis.text(anchor[0], anchor[1], f"{index}: {item['class']}",
                          color=color, fontsize=10,
                          weight='bold' if selected else 'normal',
                          bbox=dict(facecolor='black', alpha=0.45, pad=1,
                                    edgecolor='none'))

    def nearest_box(self, x: float, y: float) -> int | None:
        if not self.boxes:
            return None
        inside = []
        for index, item in enumerate(self.boxes):
            box = np.asarray(item['box'], dtype=float)
            dx, dy = x - box[0], y - box[1]
            cosine, sine = np.cos(box[6]), np.sin(box[6])
            local_x = cosine * dx + sine * dy
            local_y = -sine * dx + cosine * dy
            if (abs(local_x) <= box[3] / 2 + 0.45 and
                    abs(local_y) <= box[4] / 2 + 0.45):
                inside.append((box[3] * box[4], index))
        if inside:
            return min(inside)[1]
        centers = np.asarray([item['box'][:2] for item in self.boxes])
        distances = np.linalg.norm(centers - [x, y], axis=1)
        index = int(np.argmin(distances))
        return index if distances[index] <= 2.0 else None

    def on_press(self, event: object) -> None:
        if event.inaxes is not self.bev_axis or event.xdata is None:
            return
        x, y = float(event.xdata), float(event.ydata)
        if self.add_mode:
            self.add_click(x, y)
            return
        selected = self.nearest_box(x, y)
        if selected is None:
            self.selected = None
            self.sync_fields(); self.render(); return
        self.selected = selected
        center = np.asarray(self.boxes[selected]['box'][:2])
        self.drag_offset = center - [x, y]
        self.dragging = True
        self.sync_fields(); self.render()

    def on_motion(self, event: object) -> None:
        if not self.dragging or self.selected is None or event.inaxes is not self.bev_axis:
            return
        if event.xdata is None:
            return
        box = self.boxes[self.selected]['box']
        box[0] = float(event.xdata + self.drag_offset[0])
        box[1] = float(event.ydata + self.drag_offset[1])
        self.dirty = True
        self.sync_fields(); self.render()

    def on_release(self, _event: object) -> None:
        self.dragging = False

    def begin_add(self) -> None:
        self.add_mode = True; self.add_points = []
        self.mode_label.set('ADD RECTANGLE: click four footprint corners')
        self.status.set('Click four approximate corners in any order; a strict rectangle is fitted.')

    def add_click(self, x: float, y: float) -> None:
        self.add_points.append((x, y))
        if len(self.add_points) < 4:
            self.mode_label.set(
                f'ADD RECTANGLE: {len(self.add_points)}/4 corners selected')
            self.render()
            return
        points = np.asarray(self.add_points, dtype=np.float32)
        (cx, cy), (rect_w, rect_h), angle = cv2.minAreaRect(points)
        if rect_w >= rect_h:
            length, width, yaw = rect_w, rect_h, np.deg2rad(angle)
        else:
            length, width, yaw = rect_h, rect_w, np.deg2rad(angle + 90.0)
        length, width = max(float(length), 0.2), max(float(width), 0.2)
        name = self.class_name.get()
        _, _, height = DEFAULT_DIMS[name]
        local = self.points[np.hypot(self.points[:, 0]-cx,
                                     self.points[:, 1]-cy) < 2.0, 2]
        ground = float(np.percentile(local, 15)) if len(local) >= 5 else -1.5
        self.boxes.append({'class': name,
                           'box': [cx, cy, ground + height/2,
                                   length, width, height, float(yaw)],
                           'id': self.next_manual_id, 'source': 'manual',
                           'next_delta': [0.0, 0.0, 0.0]})
        self.next_manual_id -= 1
        self.selected = len(self.boxes) - 1
        self.dirty = True; self.cancel_mode(render=False)
        self.sync_fields(); self.render()

    def cancel_mode(self, render: bool = True) -> None:
        self.add_mode = False; self.add_points = []
        self.mode_label.set('Mode: select/move')
        if render and self.images:
            self.render()

    def sync_fields(self) -> None:
        if self.selected is None or self.selected >= len(self.boxes):
            for variable in self.variables.values(): variable.set('')
            for variable in self.delta_variables.values(): variable.set('0.000')
            return
        item = self.boxes[self.selected]
        self.class_name.set(item['class'])
        box = item['box']
        values = box[:6] + [np.rad2deg(box[6])]
        for (name, _), value in zip(PARAMETERS, values):
            self.variables[name].set(f'{value:.4f}')
        delta = item.get('next_delta', [0.0, 0.0, 0.0])
        self.delta_variables['dx'].set(f'{delta[0]:.4f}')
        self.delta_variables['dy'].set(f'{delta[1]:.4f}')
        self.delta_variables['dyaw'].set(f'{np.rad2deg(delta[2]):.4f}')

    def apply_delta(self, _event: object = None) -> bool:
        if self.selected is None:
            return True
        try:
            dx = float(self.delta_variables['dx'].get())
            dy = float(self.delta_variables['dy'].get())
            dyaw = np.deg2rad(float(self.delta_variables['dyaw'].get()))
        except ValueError:
            messagebox.showerror('Invalid movement', 'Next-frame movement must be numeric.')
            return False
        self.boxes[self.selected]['next_delta'] = [dx, dy, float(dyaw)]
        self.dirty = True
        self.status.set('Per-frame movement saved on the selected box.')
        self.render()
        return True

    def nudge_delta(self, dx: float, dy: float) -> None:
        if self.selected is None:
            return
        try:
            current_x = float(self.delta_variables['dx'].get())
            current_y = float(self.delta_variables['dy'].get())
        except ValueError:
            current_x = current_y = 0.0
        self.delta_variables['dx'].set(f'{current_x + dx:.3f}')
        self.delta_variables['dy'].set(f'{current_y + dy:.3f}')
        self.apply_delta()

    def apply_fields(self, _event: object = None) -> bool:
        if self.selected is None:
            return True
        try:
            values = [float(self.variables[name].get()) for name, _ in PARAMETERS]
        except ValueError:
            messagebox.showerror('Invalid value', 'All box fields must be numeric.')
            return False
        if min(values[3:6]) <= 0:
            messagebox.showerror('Invalid dimensions', 'Length, width, and height must be positive.')
            return False
        values[6] = float(np.deg2rad(values[6]))
        self.boxes[self.selected]['box'] = values
        self.boxes[self.selected]['class'] = self.class_name.get()
        self.boxes[self.selected]['source'] = 'manual_edit'
        self.dirty = True; self.render()
        return True

    def class_changed(self, _event: object = None) -> None:
        if self.selected is not None:
            self.boxes[self.selected]['class'] = self.class_name.get()
            self.boxes[self.selected]['source'] = 'manual_edit'
            self.dirty = True; self.render()

    def delete(self) -> None:
        if self.selected is None:
            return
        self.deleted_ids.add(int(self.boxes[self.selected].get('id', -1)))
        del self.boxes[self.selected]
        self.selected = min(self.selected, len(self.boxes)-1) if self.boxes else None
        self.dirty = True; self.sync_fields(); self.render()

    def duplicate(self) -> None:
        if self.selected is None:
            return
        item = copy.deepcopy(self.boxes[self.selected])
        item['box'][0] += 1.0; item['box'][1] += 1.0
        item['id'] = self.next_manual_id; item['source'] = 'manual'
        self.next_manual_id -= 1
        self.boxes.append(item); self.selected = len(self.boxes)-1
        self.dirty = True; self.sync_fields(); self.render()

    def reset_frame(self) -> None:
        if not messagebox.askyesno('Reset frame', 'Restore automatic boxes for this frame?'):
            return
        self.boxes = copy.deepcopy(self.initial_boxes)
        self.deleted_ids = set()
        self.next_manual_id = -(self.index.get() + 1) * 100000 - 1
        self.selected = 0 if self.boxes else None
        self.dirty = True; self.sync_fields(); self.render()

    def save_frame(self) -> bool:
        if not self.apply_fields() or not self.apply_delta():
            return False
        document = {
            'token': self.token, 'sample_idx': int(self.sample['sample_idx']),
            'updated_utc': datetime.now(timezone.utc).isoformat(),
            'source_info': str(self.info_path), 'boxes': self.boxes,
            'deleted_ids': sorted(self.deleted_ids),
        }
        destination = self.frame_path()
        temporary = destination.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(document, indent=2) + '\n')
        temporary.replace(destination)
        self.saved_tokens.add(self.token); self.dirty = False
        self.status.set(f'Saved all {len(self.boxes)} boxes to {destination}')
        self.render()
        return True

    def save_and_next(self) -> None:
        if self.save_frame():
            self.move(1, save=False, propagate=True)

    def propagated_boxes(self) -> list[dict]:
        propagated = copy.deepcopy(self.boxes)
        for item in propagated:
            delta = item.get('next_delta', [0.0, 0.0, 0.0])
            item['box'][0] += float(delta[0])
            item['box'][1] += float(delta[1])
            item['box'][6] = float(
                (item['box'][6] + float(delta[2]) + np.pi) % (2*np.pi) - np.pi)
            item['source'] = 'propagated_manual_track'
        return propagated

    def move(self, delta: int, save: bool = True,
             propagate: bool = False) -> None:
        if save and self.dirty:
            if not self.save_frame():
                return
        target = max(0, min(len(self.samples)-1, self.index.get()+delta))
        if target != self.index.get():
            propagated = self.propagated_boxes() if propagate and delta == 1 else None
            self.load_frame(target, propagated=propagated,
                            deleted_ids=(set(self.deleted_ids)
                                         if propagated is not None else None))

    def jump(self) -> None:
        try:
            target = int(self.jump_index.get())
        except ValueError:
            self.jump_index.set(str(self.index.get()))
            return
        target = max(0, min(len(self.samples)-1, target))
        if self.dirty:
            if not self.save_frame():
                return
        self.load_frame(target)

    def annotations_for(self, sample: dict) -> dict:
        token = str(sample['token'])
        path = self.frame_path(token)
        boxes = json.loads(path.read_text())['boxes'] if path.exists() else self.boxes_from_info(sample)
        return annotations_from_boxes(boxes)

    def export(self) -> None:
        if self.dirty:
            if not self.save_frame():
                return
        total, train, val = export_manual_annotations(self.info_path, self.output)
        self.status.set(f'Exported {total} reviewed frames ({train} train, {val} val)')
        messagebox.showinfo(
            'Export complete', f'Exported {total} reviewed frames\n'
            f'{train} train / {val} validation\n\n{self.output}')

    def close(self) -> None:
        if self.dirty:
            answer = messagebox.askyesnocancel('Unsaved frame', 'Save this frame before closing?')
            if answer is None:
                return
            if answer:
                if not self.save_frame():
                    return
        self.root.destroy()


def main() -> None:
    args = parse_args()
    if args.export_only:
        dataset = args.dataset.resolve()
        info_path = (args.info or dataset / 'groundtruth' /
                     'fisheye3dod_infos_all.pkl').resolve()
        output = (args.output or dataset / 'manual_annotations').resolve()
        total, train, val = export_manual_annotations(info_path, output)
        print(f'exported {total} reviewed frames: {train} train, {val} validation')
        return
    root = tk.Tk()
    root.tk.call('tk', 'scaling', args.ui_scale)
    configure_fonts(root, args.font_size)
    AnnotationEditor(root, args)
    root.mainloop()


if __name__ == '__main__':
    main()
