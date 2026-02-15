"""FreeSurfer-related Slicer services (registration, surfaces, annotations)."""

import os
import shutil
import subprocess
import tempfile

try:
    import numpy as np
except ImportError:
    np = None

try:
    import nibabel as nib
except ImportError:
    nib = None

from __main__ import slicer, vtk


class FreeSurferService:
    """Service layer for FreeSurfer integration inside Slicer."""

    def __init__(self, module_dir):
        """Store module root directory for bundled resources lookup."""
        self.module_dir = module_dir

    def _find_node_by_name(self, node_name, class_name):
        """Return first node with exact name and class, or None."""
        for node in slicer.util.getNodesByClass(class_name):
            if node.GetName() == node_name:
                return node
        return None

    def _cli_success(self, cli_node):
        """Return True when a CLI node finished without errors."""
        if cli_node is None:
            return False
        status = (cli_node.GetStatusString() or "").lower()
        return ("completed" in status) and ("error" not in status)

    def run_brainsfit_rigid_registration(
        self,
        fixed_volume_node,
        moving_volume_node,
        output_transform_node,
        initialize_mode="useGeometryAlign",
        logger=None,
    ):
        """Run BRAINSFit rigid registration for moving->fixed and return transform node."""

        def log(msg):
            if logger:
                logger(msg)

        if fixed_volume_node is None or moving_volume_node is None:
            raise ValueError("Fixed and moving volumes are required.")
        if output_transform_node is None:
            raise ValueError("Output transform node is required.")
        if not hasattr(slicer.modules, "brainsfit"):
            raise RuntimeError("BRAINSFit module is not available in this Slicer install.")

        identity = vtk.vtkMatrix4x4()
        output_transform_node.SetMatrixTransformToParent(identity)

        base = {
            "fixedVolume": fixed_volume_node.GetID(),
            "movingVolume": moving_volume_node.GetID(),
            "initializeTransformMode": initialize_mode or "useGeometryAlign",
            "samplingPercentage": 0.02,
            "minimumStepLength": 0.001,
            "maximumStepLength": 0.2,
        }
        variants = [
            dict(base, linearTransform=output_transform_node.GetID(), useRigid=True),
            dict(base, outputTransform=output_transform_node.GetID(), useRigid=True),
            dict(base, linearTransform=output_transform_node.GetID(), transformType="Rigid"),
            dict(base, outputTransform=output_transform_node.GetID(), transformType="Rigid"),
        ]

        last_error = "Unknown BRAINSFit error"
        for i, params in enumerate(variants, start=1):
            cli_node = None
            try:
                cli_node = slicer.cli.runSync(slicer.modules.brainsfit, None, params)
                if self._cli_success(cli_node):
                    log(f"[fs] BRAINSFit variant {i}/{len(variants)} succeeded")
                    return output_transform_node
                status = cli_node.GetStatusString() if cli_node is not None else "no status"
                err_text = ""
                if cli_node is not None and hasattr(cli_node, "GetErrorText"):
                    err_text = cli_node.GetErrorText() or ""
                last_error = f"{status} {err_text}".strip()
                log(f"[fs] BRAINSFit variant {i}/{len(variants)} failed: {last_error}")
            except Exception as exc:
                last_error = str(exc)
                log(f"[fs] BRAINSFit variant {i}/{len(variants)} exception: {last_error}")
            finally:
                if cli_node is not None and cli_node.GetScene() is not None:
                    slicer.mrmlScene.RemoveNode(cli_node)

        raise RuntimeError(f"BRAINSFit rigid registration failed: {last_error}")

    def _resolve_freesurfer_surf_dir(self, subject_dir):
        """Resolve FreeSurfer surf directory from subject root or direct surf path."""
        root = os.path.abspath(subject_dir)
        if not os.path.isdir(root):
            raise ValueError(f"FreeSurfer path not found: {subject_dir}")
        if os.path.basename(root) == "surf":
            return root
        surf_dir = os.path.join(root, "surf")
        if os.path.isdir(surf_dir):
            return surf_dir
        raise ValueError(f"No surf/ directory found under: {subject_dir}")

    def _resolve_freesurfer_label_dir(self, subject_dir):
        """Resolve FreeSurfer label directory from subject root, surf/, or label/ path."""
        root = os.path.abspath(subject_dir)
        if not os.path.isdir(root):
            return None
        base = os.path.basename(root)
        if base == "label":
            return root
        if base == "surf":
            label_dir = os.path.join(os.path.dirname(root), "label")
            return label_dir if os.path.isdir(label_dir) else None
        label_dir = os.path.join(root, "label")
        if os.path.isdir(label_dir):
            return label_dir
        return None

    def _surface_filenames_for_set(self, surface_set):
        """Return expected FreeSurfer surface file names for one preset."""
        key = (surface_set or "pial").strip().lower()
        mapping = {
            "pial": ["lh.pial", "rh.pial"],
            "white": ["lh.white", "rh.white"],
            "pial+white": ["lh.pial", "rh.pial", "lh.white", "rh.white"],
            "inflated": ["lh.inflated", "rh.inflated"],
        }
        if key not in mapping:
            raise ValueError(f"Unknown surface set '{surface_set}'")
        return mapping[key]

    def _style_freesurfer_model(self, node, source_path):
        """Apply default display styling to loaded FreeSurfer model nodes."""
        if node is None:
            return
        node.CreateDefaultDisplayNodes()
        display = node.GetDisplayNode()
        if display is None:
            return
        base = os.path.basename(source_path).lower()
        if base.startswith("lh."):
            display.SetColor(0.90, 0.45, 0.45)
        elif base.startswith("rh."):
            display.SetColor(0.45, 0.60, 0.90)
        else:
            display.SetColor(0.80, 0.80, 0.80)
        display.SetOpacity(0.35)
        if hasattr(display, "SetVisibility2D"):
            display.SetVisibility2D(False)
        elif hasattr(display, "SetSliceIntersectionVisibility"):
            display.SetSliceIntersectionVisibility(False)

    def _load_model_node(self, path):
        """Load one model path using multiple reader entrypoints."""
        node = None
        try:
            result = slicer.util.loadModel(path, returnNode=True)
            if isinstance(result, tuple):
                ok, node = result
                node = node if ok else None
            else:
                node = result
        except Exception:
            node = None
        if node is not None:
            return node

        for file_type in ("ModelFile", "FreeSurferModelFile", "FreeSurfer model"):
            try:
                node = slicer.util.loadNodeFromFile(path, file_type)
            except Exception:
                node = None
            if node is not None:
                return node
        return None

    def _annotation_path_for_surface(self, source_surface_path, annotation_name, label_dir):
        """Return annotation file path for one surface/hemisphere when available."""
        if not annotation_name or not label_dir or not os.path.isdir(label_dir):
            return None
        base = os.path.basename(source_surface_path)
        hemi = base.split(".", 1)[0].strip().lower()
        if hemi not in ("lh", "rh"):
            return None
        name = str(annotation_name).strip()
        if not name:
            return None
        if os.path.isabs(name) and os.path.isfile(name):
            return name
        if name.endswith(".annot"):
            if name.startswith("lh.") or name.startswith("rh."):
                filename = f"{hemi}.{name.split('.', 1)[1]}"
            else:
                filename = name
        else:
            if "{hemi}" in name:
                filename = f"{name.format(hemi=hemi)}.annot"
            else:
                filename = f"{hemi}.{name}.annot"
        path = os.path.join(label_dir, filename)
        if os.path.isfile(path):
            return path
        return None

    def _default_freesurfer_color_node(self):
        """Find a likely FreeSurfer color node already present in the scene."""
        candidates = []
        try:
            candidates = list(slicer.util.getNodesByClass("vtkMRMLColorTableNode"))
        except Exception:
            candidates = []
        preferred_names = ["FreeSurferLabels", "FreeSurferParcellation", "FreeSurferColorLUT"]
        for target in preferred_names:
            for node in candidates:
                if (node.GetName() or "") == target:
                    return node
        for node in candidates:
            name = (node.GetName() or "").lower()
            if "freesurfer" in name or "aparc" in name:
                return node
        return None

    def _bundled_freesurfer_lut_path(self):
        """Return path to bundled FreeSurfer LUT shipped with the module."""
        path = os.path.join(self.module_dir, "Resources", "freesurfer", "FreeSurferColorLUT20120827.txt")
        if os.path.isfile(path):
            return path
        return None

    def _load_color_table_node(self, path, logger=None):
        """Load a color table node from file and return it."""

        def log(msg):
            if logger:
                logger(msg)

        if not path:
            return None
        lut_path = os.path.abspath(path)
        if not os.path.isfile(lut_path):
            raise ValueError(f"Color LUT file not found: {path}")
        if lut_path.lower().endswith(".annot"):
            raise ValueError(
                f"Annotation file provided as LUT ({path}). Please select a color table text file (for example FreeSurferColorLUT.txt)."
            )

        for node in slicer.util.getNodesByClass("vtkMRMLColorTableNode"):
            storage = node.GetStorageNode()
            if storage is None:
                continue
            file_name = storage.GetFileName() or ""
            if file_name and os.path.abspath(file_name) == lut_path:
                return node

        node = None
        try:
            if hasattr(slicer.util, "loadColorTable"):
                node = slicer.util.loadColorTable(lut_path)
        except Exception:
            node = None
        if node is None:
            try:
                node = slicer.util.loadNodeFromFile(lut_path, "ColorTableFile")
            except Exception:
                node = None
        if node is None:
            raise RuntimeError(f"Failed to load color table: {path}")
        log(f"[fs] loaded LUT: {lut_path}")
        return node

    def _preferred_annotation_color_node(self, color_lut_path=None, logger=None):
        """Resolve preferred color node: user LUT -> FreeSurferLabels -> bundled LUT -> fallback."""

        def log(msg):
            if logger:
                logger(msg)

        if color_lut_path:
            try:
                return self._load_color_table_node(color_lut_path, logger=logger)
            except Exception as exc:
                log(f"[fs] LUT warning: {exc}")

        for node in slicer.util.getNodesByClass("vtkMRMLColorTableNode"):
            if (node.GetName() or "") == "FreeSurferLabels":
                return node

        bundled = self._bundled_freesurfer_lut_path()
        if bundled:
            try:
                node = self._load_color_table_node(bundled, logger=logger)
                log(f"[fs] using bundled LUT: {bundled}")
                return node
            except Exception as exc:
                log(f"[fs] bundled LUT warning: {exc}")

        return self._default_freesurfer_color_node()

    def _read_annot_nibabel(self, annot_path):
        """Read FreeSurfer .annot labels with nibabel and return normalized arrays."""
        if nib is None:
            raise RuntimeError("Nibabel is not available in this Slicer Python environment.")
        labels, ctab, names = nib.freesurfer.io.read_annot(annot_path, orig_ids=False)
        labels = np.asarray(labels, dtype=np.int32).reshape(-1)
        ctab = np.asarray(ctab) if ctab is not None else np.empty((0, 5), dtype=np.int32)
        decoded = []
        for name in names or []:
            if isinstance(name, bytes):
                decoded.append(name.decode("utf-8", errors="ignore"))
            else:
                decoded.append(str(name))
        return labels, ctab, decoded

    def _ensure_color_table_from_annot(self, node_name, ctab, names):
        """Create/update a color table node from annotation colortable."""
        color_node = self._find_node_by_name(node_name, "vtkMRMLColorTableNode")
        if color_node is None:
            color_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLColorTableNode", node_name)
        color_node.SetTypeToUser()

        entry_count = max(len(names), int(ctab.shape[0]) if ctab is not None else 0)
        color_node.SetNumberOfColors(entry_count + 1)
        color_node.SetColor(0, "Unknown", 0.5, 0.5, 0.5, 0.0)

        for idx in range(entry_count):
            label_name = names[idx] if idx < len(names) else f"Label_{idx}"
            if ctab is not None and idx < int(ctab.shape[0]) and int(ctab.shape[1]) >= 4:
                r = float(ctab[idx, 0]) / 255.0
                g = float(ctab[idx, 1]) / 255.0
                b = float(ctab[idx, 2]) / 255.0
                t = float(ctab[idx, 3])
                a = max(0.0, min(1.0, 1.0 - t / 255.0))
            else:
                r = g = b = 0.7
                a = 1.0
            color_node.SetColor(idx + 1, label_name, r, g, b, a)

        return color_node

    def _apply_annotation_labels_with_nibabel(self, model_node, annot_path, color_node=None):
        """Attach nibabel-loaded annotation labels as model point scalars and enable display."""
        if model_node is None:
            return False, None
        poly = model_node.GetPolyData()
        if poly is None:
            return False, None
        point_count = int(poly.GetNumberOfPoints())
        labels, ctab, names = self._read_annot_nibabel(annot_path)
        if int(labels.shape[0]) != point_count:
            raise RuntimeError(
                f"Annot vertex count mismatch: annot={int(labels.shape[0])}, surface={point_count}"
            )

        scalar_values = labels + 1
        array_name = f"FSAnnot_{os.path.basename(annot_path)}"
        vtk_arr = vtk.vtkIntArray()
        vtk_arr.SetName(array_name)
        vtk_arr.SetNumberOfComponents(1)
        vtk_arr.SetNumberOfTuples(point_count)
        for i in range(point_count):
            vtk_arr.SetValue(i, int(scalar_values[i]))

        pdata = poly.GetPointData()
        existing = pdata.GetArray(array_name)
        if existing is not None:
            pdata.RemoveArray(array_name)
        pdata.AddArray(vtk_arr)
        pdata.SetActiveScalars(array_name)
        poly.Modified()
        model_node.Modified()

        if color_node is None:
            hemi = (os.path.basename(annot_path).split(".", 1)[0] or "hemi").lower()
            annot_base = os.path.basename(annot_path).replace(".annot", "")
            if annot_base.startswith("lh.") or annot_base.startswith("rh."):
                annot_base = annot_base.split(".", 1)[1]
            color_node = self._ensure_color_table_from_annot(
                f"FS_{hemi}_{annot_base}_LUT",
                ctab=ctab,
                names=names,
            )

        display = model_node.GetDisplayNode()
        if display is None:
            model_node.CreateDefaultDisplayNodes()
            display = model_node.GetDisplayNode()
        if display is None:
            return False, color_node

        if hasattr(display, "SetScalarVisibility"):
            display.SetScalarVisibility(True)
        if hasattr(display, "SetActiveScalarName"):
            display.SetActiveScalarName(array_name)
        if color_node is not None and hasattr(display, "SetAndObserveColorNodeID"):
            display.SetAndObserveColorNodeID(color_node.GetID())
        if hasattr(display, "SetOpacity"):
            display.SetOpacity(1.0)
        if hasattr(display, "SetScalarRangeFlag"):
            try:
                display.SetScalarRangeFlag(display.UseColorNodeScalarRange)
            except Exception:
                pass
        return True, color_node

    def _select_annotation_scalar_array_name(self, model_node):
        """Pick a scalar array name likely containing annotation labels."""
        if model_node is None:
            return None
        poly = model_node.GetPolyData()
        if poly is None:
            return None
        pdata = poly.GetPointData()
        if pdata is None:
            return None
        n_arrays = int(pdata.GetNumberOfArrays())
        if n_arrays <= 0:
            return None

        preferred = []
        fallback = []
        n_points = int(poly.GetNumberOfPoints())
        for idx in range(n_arrays):
            arr = pdata.GetArray(idx)
            if arr is None:
                continue
            name = arr.GetName() or f"Array_{idx}"
            tuples = int(arr.GetNumberOfTuples())
            comps = int(arr.GetNumberOfComponents())
            if tuples <= 0:
                continue
            lower = name.lower()
            if tuples == n_points and comps == 1:
                if ("annot" in lower) or ("label" in lower) or ("parc" in lower):
                    preferred.append(name)
                else:
                    fallback.append(name)
        if preferred:
            return preferred[0]
        if fallback:
            return fallback[0]
        return None

    def _apply_annotation_display(self, model_node, color_node=None):
        """Enable scalar annotation display on one model node when available."""
        if model_node is None:
            return False
        scalar_name = self._select_annotation_scalar_array_name(model_node)
        if not scalar_name:
            return False
        display = model_node.GetDisplayNode()
        if display is None:
            model_node.CreateDefaultDisplayNodes()
            display = model_node.GetDisplayNode()
        if display is None:
            return False
        if hasattr(display, "SetScalarVisibility"):
            display.SetScalarVisibility(True)
        if hasattr(display, "SetActiveScalarName"):
            display.SetActiveScalarName(scalar_name)
        if color_node is not None and hasattr(display, "SetAndObserveColorNodeID"):
            display.SetAndObserveColorNodeID(color_node.GetID())
        if hasattr(display, "SetOpacity"):
            display.SetOpacity(1.0)
        if color_node is not None and hasattr(display, "SetScalarRangeFlag"):
            try:
                display.SetScalarRangeFlag(display.UseColorNodeScalarRange)
            except Exception:
                pass
        elif hasattr(display, "SetScalarRangeFlagFromData"):
            display.SetScalarRangeFlagFromData()
        return True

    def _has_freesurfer_extension(self):
        """Return True when a FreeSurfer-related Slicer module appears available."""
        names = []
        try:
            names = dir(slicer.modules)
        except Exception:
            names = []
        for name in names:
            if "freesurfer" in str(name).lower():
                return True
        return False

    def _apply_annotation_with_slicerfreesurfer(self, model_node, annot_path, color_node=None, logger=None):
        """Best-effort annotation import via SlicerFreeSurfer reader pipeline."""

        def log(msg):
            if logger:
                logger(msg)

        if model_node is None or not annot_path or not os.path.isfile(annot_path):
            return False
        if not self._has_freesurfer_extension():
            return False
        if not hasattr(slicer.modules, "freesurferimporter"):
            return False

        poly = model_node.GetPolyData()
        pdata = poly.GetPointData() if poly is not None else None
        before_arrays = int(pdata.GetNumberOfArrays()) if pdata is not None else 0

        logic = slicer.modules.freesurferimporter.logic()
        if logic is None:
            return False

        loaded_any = False
        try:
            loaded_any = bool(logic.LoadFreeSurferScalarOverlay(annot_path, model_node))
        except Exception:
            loaded_any = False
        if not loaded_any:
            try:
                collection = vtk.vtkCollection()
                collection.AddItem(model_node)
                loaded_any = bool(logic.LoadFreeSurferScalarOverlay(annot_path, collection))
            except Exception:
                loaded_any = False

        poly = model_node.GetPolyData()
        pdata = poly.GetPointData() if poly is not None else None
        after_arrays = int(pdata.GetNumberOfArrays()) if pdata is not None else 0
        added_arrays = after_arrays > before_arrays
        display_applied = self._apply_annotation_display(model_node, color_node=color_node)
        success = bool(added_arrays or display_applied)
        if success:
            log(f"[fs] applied annotation via SlicerFreeSurfer: {annot_path}")
        return success

    def _find_mris_convert(self):
        """Resolve mris_convert executable path if available."""
        tool = shutil.which("mris_convert")
        if tool:
            return tool
        default_path = "/Applications/freesurfer/7.4.1/bin/mris_convert"
        if os.path.isfile(default_path):
            return default_path
        return None

    def _convert_freesurfer_surface_to_vtk(self, source_path, annot_path=None):
        """Convert a FreeSurfer surface to VTK polydata using mris_convert."""
        tool = self._find_mris_convert()
        if not tool:
            raise RuntimeError("mris_convert not found (FreeSurfer is required for conversion fallback).")

        stem = os.path.basename(source_path).replace(".", "_")
        if annot_path:
            annot_stem = os.path.basename(annot_path).replace(".", "_")
            stem = f"{stem}_{annot_stem}"
        out_dir = os.path.join(tempfile.gettempdir(), "rosa_helper_fs_cache")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{stem}.vtk")
        cmd = [tool]
        if annot_path:
            cmd.extend(["--annot", annot_path])
        cmd.extend([source_path, out_path])
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0 or not os.path.isfile(out_path):
            err = (proc.stderr or proc.stdout or "").strip()
            extra = f" annot={annot_path}" if annot_path else ""
            raise RuntimeError(err or f"mris_convert failed for {source_path}{extra}")
        return out_path

    def load_freesurfer_surfaces(
        self,
        subject_dir,
        surface_set="pial",
        annotation_name=None,
        color_lut_path=None,
        logger=None,
    ):
        """Load FreeSurfer surface models and optional annotation scalars/LUT."""

        def log(msg):
            if logger:
                logger(msg)

        surf_dir = self._resolve_freesurfer_surf_dir(subject_dir)
        label_dir = self._resolve_freesurfer_label_dir(subject_dir)
        names = self._surface_filenames_for_set(surface_set)
        loaded_nodes = []
        missing_paths = []
        failed_paths = []
        missing_annotation_paths = []
        annotated_nodes = 0
        color_node = None
        if annotation_name:
            color_node = self._preferred_annotation_color_node(color_lut_path=color_lut_path, logger=logger)
            if color_node is None:
                log("[fs] no LUT color node available; annotations will still load but colors may be generic")

        for name in names:
            path = os.path.join(surf_dir, name)
            if not os.path.isfile(path):
                missing_paths.append(path)
                continue

            annot_path = None
            if annotation_name:
                annot_path = self._annotation_path_for_surface(path, annotation_name, label_dir)
                if annot_path is None:
                    hemi = os.path.basename(path).split(".", 1)[0]
                    missing_annotation_paths.append(
                        os.path.join(label_dir or "<missing_label_dir>", f"{hemi}.{annotation_name}.annot")
                    )

            node = None
            try:
                node = self._load_model_node(path)
            except Exception:
                node = None

            if node is None:
                try:
                    vtk_path = self._convert_freesurfer_surface_to_vtk(path, annot_path=annot_path)
                    node = self._load_model_node(vtk_path)
                    if node is not None:
                        log(f"[fs] converted and loaded: {path} -> {vtk_path}")
                except Exception as exc:
                    log(f"[fs] convert fallback failed for {path}: {exc}")
                    if "license" in str(exc).lower():
                        log(
                            "[fs] FreeSurfer license is required for mris_convert-based annotation conversion. "
                            "Set FS_LICENSE or install .license."
                        )

            if node is None:
                failed_paths.append(path)
                continue

            node.SetName(f"FS_{name}")
            self._style_freesurfer_model(node, path)

            annotation_applied = False
            if annotation_name and annot_path and nib is not None and np is not None:
                try:
                    annotation_applied, inferred_color_node = self._apply_annotation_labels_with_nibabel(
                        node,
                        annot_path,
                        color_node=color_node,
                    )
                    if color_node is None and inferred_color_node is not None:
                        color_node = inferred_color_node
                    if annotation_applied:
                        log(f"[fs] applied annotation via nibabel: {annot_path}")
                except Exception as exc:
                    log(f"[fs] nibabel annot apply failed for {path}: {exc}")
            elif annotation_name and annot_path and nib is None:
                log("[fs] nibabel not available; cannot read .annot directly in this environment.")

            if annotation_name and annot_path and not annotation_applied:
                try:
                    annotation_applied = self._apply_annotation_with_slicerfreesurfer(
                        node,
                        annot_path,
                        color_node=color_node,
                        logger=logger,
                    )
                except Exception as exc:
                    log(f"[fs] SlicerFreeSurfer annot apply failed for {path}: {exc}")

            if annotation_name and not annotation_applied:
                annotation_applied = self._apply_annotation_display(node, color_node=color_node)

            if annotation_applied:
                annotated_nodes += 1

            loaded_nodes.append(node)
            log(f"[fs] loaded surface: {path}")

        return {
            "loaded_nodes": loaded_nodes,
            "missing_surface_paths": missing_paths,
            "failed_surface_paths": failed_paths,
            "missing_annotation_paths": sorted(set(missing_annotation_paths)),
            "annotated_nodes": int(annotated_nodes),
            "color_node_name": color_node.GetName() if color_node is not None else "",
        }

    def apply_transform_to_model_nodes(self, model_nodes, transform_node, harden=False):
        """Apply one transform node to all model nodes and optionally harden."""
        if transform_node is None:
            raise ValueError("Transform node is required.")
        for node in model_nodes or []:
            if node is None:
                continue
            node.SetAndObserveTransformNodeID(transform_node.GetID())
            if harden:
                slicer.vtkSlicerTransformLogic().hardenTransform(node)
