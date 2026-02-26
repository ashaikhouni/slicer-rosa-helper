"""Parse ROSA `.ros` text into display and trajectory records.

ROSA files are tokenized as bracketed headers followed by free-form payload:
`[TOKEN]` then one or more lines until the next token.
This parser extracts only the fields needed by ROSA Helper:
- display transforms (`TRdicomRdisplay` + `VOLUME`)
- display metadata (`IMAGERY_NAME`, `SERIE_UID`, `IMAGERY_3DREF`)
- trajectories (`TRAJECTORY` and `ELLIPS`)
"""

import re
from pathlib import Path


_FLOAT_RE = r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?"


def extract_tokens(text):
    """Return ordered token blocks from a ROSA text blob.

    Parameters
    ----------
    text: str
        Full `.ros` file content.

    Returns
    -------
    list[dict]
        Each item has `token` and `content` keys.
    """
    tokens = []
    matches = list(re.finditer(r"\[(.*?)\]", text))
    for i in range(len(matches) - 1):
        token = matches[i].group(1).strip()
        start = matches[i].end()
        end = matches[i + 1].start()
        content = text[start:end]
        tokens.append({"token": token, "content": content})
    return tokens


def _parse_display_matrix(block):
    """Parse a 4x4 matrix from a `TRdicomRdisplay` token payload."""
    nums = re.findall(_FLOAT_RE, block)
    if len(nums) < 16:
        return None
    vals = [float(v) for v in nums[-16:]]
    return [vals[0:4], vals[4:8], vals[8:12], vals[12:16]]


def _parse_volume_path(block):
    """Parse full ROSA volume path from a `VOLUME` payload."""
    lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
    if not lines:
        return None
    return lines[0].replace("\\", "/")


def _parse_volume_name(block):
    """Parse trailing volume name from a `VOLUME` payload path."""
    volume_path = _parse_volume_path(block)
    if not volume_path:
        return None
    return volume_path.split("/")[-1]


def _parse_name_field(block):
    """Parse first non-empty line from a simple token payload."""
    lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
    return lines[0] if lines else None


def _parse_int_field(block):
    """Parse integer payload value; return `None` if not valid."""
    value = _parse_name_field(block)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _parse_trajectory(block):
    """Parse a single trajectory payload into name/start/end points."""
    lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
    if len(lines) < 2:
        return None
    fields = [f for f in lines[1].split(" ") if f]
    if len(fields) < 11:
        return None
    try:
        name = fields[0]
        start = [float(fields[4]), float(fields[5]), float(fields[6])]
        end = [float(fields[8]), float(fields[9]), float(fields[10])]
    except ValueError:
        return None
    return {"name": name, "start": start, "end": end}


def parse_ros_text(text):
    """Parse ROSA text and return displays + trajectories.

    Output dictionary schema:
    - `displays`: ordered list of display dictionaries
    - `trajectories`: ordered list of trajectory dictionaries
    """
    tokens = extract_tokens(text)

    displays = []
    for i, tok in enumerate(tokens):
        if tok["token"] != "TRdicomRdisplay":
            continue
        if i + 1 >= len(tokens) or tokens[i + 1]["token"] != "VOLUME":
            continue
        matrix = _parse_display_matrix(tok["content"])
        volume_name = _parse_volume_name(tokens[i + 1]["content"])
        volume_path = _parse_volume_path(tokens[i + 1]["content"])
        if matrix and volume_name:
            displays.append(
                {
                    "volume": volume_name,
                    "volume_path": volume_path,
                    "matrix": matrix,
                }
            )

    imagery_names = []
    serie_uids = []
    imagery_refs = []
    for tok in tokens:
        if tok["token"] == "IMAGERY_NAME":
            name = _parse_name_field(tok["content"])
            if name:
                imagery_names.append(name)
        if tok["token"] == "SERIE_UID":
            uid = _parse_name_field(tok["content"])
            if uid:
                serie_uids.append(uid)
        if tok["token"] == "IMAGERY_3DREF":
            ref = _parse_int_field(tok["content"])
            if ref is not None:
                imagery_refs.append(ref)

    for i, disp in enumerate(displays):
        disp["index"] = i
        if i < len(imagery_names):
            disp["imagery_name"] = imagery_names[i]
        if i < len(serie_uids):
            disp["serie_uid"] = serie_uids[i]
        if i < len(imagery_refs):
            disp["imagery_3dref"] = imagery_refs[i]

    trajectories = []
    for tok in tokens:
        if tok["token"] not in ("TRAJECTORY", "ELLIPS"):
            continue
        traj = _parse_trajectory(tok["content"])
        if traj:
            trajectories.append(traj)

    return {
        "displays": displays,
        "trajectories": trajectories,
    }


def parse_ros_file(ros_path):
    """Read and parse a `.ros` file path."""
    path = Path(ros_path)
    text = path.read_text(encoding="utf-8", errors="ignore")
    parsed = parse_ros_text(text)
    parsed["ros_path"] = str(path)
    return parsed
