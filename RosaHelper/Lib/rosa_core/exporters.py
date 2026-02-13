import json

from .transforms import apply_affine, lps_to_ras_point


def build_markups_lines(trajectories, to_ras=True, display_to_dicom=None):
    markups = []
    for traj in trajectories:
        start = list(traj["start"])
        end = list(traj["end"])

        if display_to_dicom is not None:
            start = apply_affine(display_to_dicom, start)
            end = apply_affine(display_to_dicom, end)

        if to_ras:
            start = lps_to_ras_point(start)
            end = lps_to_ras_point(end)
            coord = "RAS"
        else:
            coord = "LPS"

        name = traj["name"]
        markups.append(
            {
                "type": "Line",
                "name": name,
                "coordinateSystem": coord,
                "locked": False,
                "fixedNumberOfControlPoints": True,
                "labelFormat": "%N",
                "lastUsedControlPointNumber": 2,
                "controlPoints": [
                    {
                        "id": f"{name}_start",
                        "label": f"{name}_start",
                        "position": start,
                        "orientation": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                        "selected": True,
                        "locked": False,
                        "visibility": True,
                    },
                    {
                        "id": f"{name}_end",
                        "label": f"{name}_end",
                        "position": end,
                        "orientation": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                        "selected": True,
                        "locked": False,
                        "visibility": True,
                    },
                ],
                "measurements": [],
                "display": {
                    "visibility": True,
                    "opacity": 1.0,
                    "color": [0.9, 0.2, 0.2],
                    "selectedColor": [1.0, 0.6, 0.2],
                    "propertiesLabelVisibility": False,
                    "pointLabelsVisibility": False,
                    "glyphType": "Sphere3D",
                    "glyphScale": 1.0,
                    "textScale": 1.0,
                    "lineThickness": 0.2,
                },
            }
        )

    return markups


def build_markups_document(markups):
    return {
        "@schema": "https://raw.githubusercontent.com/slicer/slicer/master/Modules/Loadable/Markups/Resources/Schema/markups-schema-v1.0.0.json",
        "markups": markups,
    }


def save_markups_json(path, markups):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(build_markups_document(markups), f, indent=2)


def build_fcsv_rows(trajectories, to_ras=True, same_label_pair=False):
    rows = []
    for traj in trajectories:
        start = list(traj["start"])
        end = list(traj["end"])
        if to_ras:
            start = lps_to_ras_point(start)
            end = lps_to_ras_point(end)
        name = traj["name"]

        if same_label_pair:
            rows.append({"label": name, "xyz": start})
            rows.append({"label": name, "xyz": end})
        else:
            rows.append({"label": f"{name}_entry", "xyz": start})
            rows.append({"label": f"{name}_target", "xyz": end})
    return rows


def save_fcsv(path, rows):
    header = [
        "# Markups fiducial file version = 4.11",
        "# CoordinateSystem = 0",
        "# columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID",
    ]
    with open(path, "w", encoding="utf-8") as f:
        for line in header:
            f.write(line + "\n")
        for idx, row in enumerate(rows, start=1):
            x, y, z = row["xyz"]
            values = [
                str(idx),
                f"{x:.6f}",
                f"{y:.6f}",
                f"{z:.6f}",
                "0",
                "0",
                "0",
                "1",
                "1",
                "1",
                "0",
                row["label"],
                "",
                "",
            ]
            f.write(",".join(values) + "\n")
