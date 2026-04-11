"""Candidate merging, suppression, and deduplication for Deep Core."""

try:
    import numpy as np
except ImportError:
    np = None


class DeepCoreCandidateDedupMixin:
    """Merge and suppress overlapping Deep Core candidate proposals."""

    def _merge_collinear_token_line_candidates(
        self,
        pts_ras,
        blob_ids,
        token_atom_ids,
        candidates,
        inlier_radius_mm=2.0,
        axial_bin_mm=2.5,
        min_span_mm=16.0,
        min_inlier_count=5,
        elongation_map=None,
        parent_blob_id_map=None,
    ):
        ordered = sorted(list(candidates or []), key=self._token_line_candidate_quality, reverse=True)
        merged = list(ordered)
        seen_pairs = set()
        extra = []
        for idx_a, cand_a in enumerate(ordered[:-1]):
            source_a = str(cand_a.get("seed_source_blob_class") or cand_a.get("source_blob_class") or "")
            if source_a != "complex_blob":
                continue
            axis_a = np.asarray(cand_a.get("axis_ras") or [0.0, 0.0, 1.0], dtype=float).reshape(3)
            axis_a = axis_a / max(float(np.linalg.norm(axis_a)), 1e-6)
            start_a = np.asarray(cand_a.get("start_ras") or [0.0, 0.0, 0.0], dtype=float).reshape(3)
            end_a = np.asarray(cand_a.get("end_ras") or [0.0, 0.0, 0.0], dtype=float).reshape(3)
            for cand_b in ordered[(idx_a + 1):]:
                pair_key = (
                    tuple(sorted(int(v) for v in list(cand_a.get("token_indices") or [])[:8])),
                    tuple(sorted(int(v) for v in list(cand_b.get("token_indices") or [])[:8])),
                )
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                axis_b = np.asarray(cand_b.get("axis_ras") or [0.0, 0.0, 1.0], dtype=float).reshape(3)
                axis_b = axis_b / max(float(np.linalg.norm(axis_b)), 1e-6)
                axis_dot = abs(float(np.dot(axis_a, axis_b)))
                if axis_dot < 0.92:
                    continue
                start_b = np.asarray(cand_b.get("start_ras") or [0.0, 0.0, 0.0], dtype=float).reshape(3)
                end_b = np.asarray(cand_b.get("end_ras") or [0.0, 0.0, 0.0], dtype=float).reshape(3)
                endpoint_pairs = [
                    (start_a, start_b),
                    (start_a, end_b),
                    (end_a, start_b),
                    (end_a, end_b),
                ]
                endpoint_gap = min(float(np.linalg.norm(p0 - p1)) for p0, p1 in endpoint_pairs)
                if endpoint_gap > 24.0:
                    continue
                lateral_ab = float(
                    self._radial_distance_to_line(
                        np.vstack((start_b.reshape(1, 3), end_b.reshape(1, 3))),
                        np.asarray(cand_a.get("center_ras") or 0.5 * (start_a + end_a), dtype=float).reshape(3),
                        axis_a,
                    ).mean()
                )
                lateral_ba = float(
                    self._radial_distance_to_line(
                        np.vstack((start_a.reshape(1, 3), end_a.reshape(1, 3))),
                        np.asarray(cand_b.get("center_ras") or 0.5 * (start_b + end_b), dtype=float).reshape(3),
                        axis_b,
                    ).mean()
                )
                if min(lateral_ab, lateral_ba) > 4.0:
                    continue
                token_union = sorted(
                    set(int(v) for v in list(cand_a.get("token_indices") or []))
                    | set(int(v) for v in list(cand_b.get("token_indices") or []))
                )
                if len(token_union) <= max(
                    len(list(cand_a.get("token_indices") or [])),
                    len(list(cand_b.get("token_indices") or [])),
                ):
                    continue
                base = dict(cand_a if self._token_line_candidate_quality(cand_a) >= self._token_line_candidate_quality(cand_b) else cand_b)
                base["token_indices"] = [int(v) for v in token_union]
                base["atom_id_list"] = sorted(
                    set(int(v) for v in list(cand_a.get("atom_id_list") or []))
                    | set(int(v) for v in list(cand_b.get("atom_id_list") or []))
                )
                base["proposal_family"] = "merged_collinear"
                completed = self._complete_token_line_candidate(
                    pts_ras=pts_ras,
                    blob_ids=blob_ids,
                    token_atom_ids=token_atom_ids,
                    candidate=base,
                    inlier_radius_mm=float(inlier_radius_mm),
                    axial_bin_mm=float(axial_bin_mm),
                    min_span_mm=float(min_span_mm),
                    min_inlier_count=int(min_inlier_count),
                    elongation_map=elongation_map,
                    parent_blob_id_map=parent_blob_id_map,
                )
                if completed is None:
                    continue
                if float(completed.get("span_mm", 0.0)) <= max(float(cand_a.get("span_mm", 0.0)), float(cand_b.get("span_mm", 0.0))) + 4.0:
                    continue
                extra.append(completed)
        if extra:
            merged.extend(extra)
        return merged

    def _nms_token_line_candidates(self, candidates, max_output=None):
        ordered = sorted(list(candidates or []), key=self._token_line_candidate_quality, reverse=True)
        survivors = []
        for cand in ordered:
            duplicate = False
            for prev in survivors:
                if self._token_line_candidates_overlap(cand, prev):
                    duplicate = True
                    break
            if not duplicate:
                survivors.append(cand)

        kept = []
        covered_tokens = set()
        covered_blobs = set()
        remaining = list(survivors)
        limit = None if max_output is None else int(max_output)
        if limit is not None and limit <= 0:
            limit = None
        while remaining and (limit is None or len(kept) < limit):
            best_idx = None
            best_gain = None
            for idx, cand in enumerate(remaining):
                token_set = set(int(v) for v in list(cand.get("token_indices") or []))
                blob_set = set(int(v) for v in list(cand.get("blob_id_list") or []))
                new_tokens = int(len(token_set - covered_tokens))
                new_blobs = int(len(blob_set - covered_blobs))
                reused_tokens = int(len(token_set & covered_tokens))
                gain = (
                    self._token_line_candidate_quality(cand)
                    + 1.4 * float(new_tokens)
                    + 6.0 * float(new_blobs)
                    - 0.6 * float(reused_tokens)
                )
                if best_gain is None or gain > best_gain:
                    best_idx = int(idx)
                    best_gain = float(gain)
            if best_idx is None:
                break
            best = remaining.pop(int(best_idx))
            kept.append(best)
            covered_tokens.update(int(v) for v in list(best.get("token_indices") or []))
            covered_blobs.update(int(v) for v in list(best.get("blob_id_list") or []))
        return kept

    def _resolve_contact_chain_stitched_claims(
        self,
        pts_ras,
        blob_ids,
        token_atom_ids,
        proposals,
        support_atoms=None,
        inlier_radius_mm=2.0,
        axial_bin_mm=2.5,
        min_span_mm=16.0,
        min_inlier_count=5,
        elongation_map=None,
        parent_blob_id_map=None,
        max_output=None,
    ):
        kept = [dict(p or {}) for p in list(proposals or [])]
        if not kept:
            return []
        pts = np.asarray(pts_ras, dtype=float).reshape(-1, 3)
        blob_ids = np.asarray(blob_ids, dtype=np.int32).reshape(-1)
        token_atom_ids = np.asarray(token_atom_ids, dtype=np.int32).reshape(-1)
        support_atom_map = {
            int(dict(atom or {}).get("atom_id", -1)): dict(atom or {})
            for atom in list(support_atoms or [])
            if int(dict(atom or {}).get("atom_id", -1)) > 0
        }
        stitched = []
        for proposal in kept:
            family = str(proposal.get("proposal_family") or "")
            atom_id_list = [int(v) for v in list(proposal.get("atom_id_list") or []) if int(v) > 0]
            if family in {"blob_connectivity", "merged_collinear"} or len(atom_id_list) >= 2:
                stitched.append(dict(proposal))
        if not stitched:
            return kept

        adjusted = []
        for proposal in kept:
            family = str(proposal.get("proposal_family") or "")
            source = str(proposal.get("seed_source_blob_class") or proposal.get("source_blob_class") or "")
            atom_id_list = [int(v) for v in list(proposal.get("atom_id_list") or []) if int(v) > 0]
            if not (family == "blob_axis" and source == "contact_chain" and len(atom_id_list) == 1):
                adjusted.append(dict(proposal))
                continue
            seed_atom_id = int(atom_id_list[0])
            seed_atom = dict(support_atom_map.get(int(seed_atom_id)) or {})
            raw_blob_members = {int(v) for v in list(seed_atom.get("parent_blob_id_list") or []) if int(v) > 0}
            if not raw_blob_members:
                adjusted.append(dict(proposal))
                continue
            claimed_raw = set()
            for stitched_prop in stitched:
                claimed_raw.update(int(v) for v in list(stitched_prop.get("blob_id_list") or []) if int(v) in raw_blob_members)
            if not claimed_raw:
                adjusted.append(dict(proposal))
                continue
            remaining_raw = set(int(v) for v in raw_blob_members if int(v) not in claimed_raw)
            if len(remaining_raw) <= 2:
                continue
            base_token_indices = [int(v) for v in list(proposal.get("token_indices") or []) if 0 <= int(v) < int(pts.shape[0])]
            trimmed_token_indices = [
                int(idx)
                for idx in base_token_indices
                if int(token_atom_ids[int(idx)]) == int(seed_atom_id) and int(blob_ids[int(idx)]) in remaining_raw
            ]
            if len(trimmed_token_indices) < 3:
                continue
            trimmed = dict(proposal)
            trimmed["token_indices"] = [int(v) for v in trimmed_token_indices]
            trimmed["blob_id_list"] = [int(v) for v in sorted(remaining_raw)]
            trimmed["parent_blob_id_list"] = [int(v) for v in sorted(remaining_raw)]
            completed = self._complete_token_line_candidate(
                pts_ras=pts,
                blob_ids=blob_ids,
                token_atom_ids=token_atom_ids,
                candidate=trimmed,
                inlier_radius_mm=float(inlier_radius_mm),
                axial_bin_mm=float(axial_bin_mm),
                min_span_mm=float(max(8.0, min_span_mm - 8.0)),
                min_inlier_count=int(min(3, len(trimmed_token_indices))),
                elongation_map=elongation_map,
                parent_blob_id_map=parent_blob_id_map,
            )
            if completed is None:
                continue
            completed["atom_id_list"] = [int(seed_atom_id)]
            adjusted.append(dict(completed))
        return self._nms_token_line_candidates(adjusted, max_output=max_output)

    def _suppress_stitched_subsumed_blob_axis_proposals(self, proposals):
        kept = [dict(p or {}) for p in list(proposals or [])]
        if not kept:
            return []
        stitched = []
        for proposal in kept:
            family = str(proposal.get("proposal_family") or "")
            atom_ids = [int(v) for v in list(proposal.get("atom_id_list") or []) if int(v) > 0]
            if family in {"blob_connectivity", "merged_collinear"} or len(atom_ids) >= 2:
                stitched.append(dict(proposal))
        if not stitched:
            return kept

        survivors = []
        for proposal in kept:
            family = str(proposal.get("proposal_family") or "")
            atom_ids = [int(v) for v in list(proposal.get("atom_id_list") or []) if int(v) > 0]
            if not (family == "blob_axis" and len(atom_ids) == 1):
                survivors.append(dict(proposal))
                continue
            seed_atom_id = int(atom_ids[0])
            covered = False
            for stitched_prop in stitched:
                stitched_atoms = {int(v) for v in list(stitched_prop.get("atom_id_list") or []) if int(v) > 0}
                if seed_atom_id not in stitched_atoms:
                    continue
                if not self._token_line_candidates_overlap(proposal, stitched_prop):
                    continue
                if float(stitched_prop.get("span_mm", 0.0)) < float(proposal.get("span_mm", 0.0)) + 4.0:
                    continue
                covered = True
                break
            if covered:
                continue
            survivors.append(dict(proposal))
        return self._suppress_family_level_duplicates(survivors)

    def _suppress_family_level_duplicates(self, proposals):
        kept = [dict(p or {}) for p in list(proposals or [])]
        if len(kept) < 2:
            return kept

        family_priority = {
            "blob_connectivity": 5,
            "merged_collinear": 4,
            "graph": 4,
            "blob_axis": 3,
        }

        ordered = sorted(
            kept,
            key=lambda p: (
                int(family_priority.get(str(p.get("proposal_family") or ""), 0)),
                float(p.get("span_mm", 0.0)),
                float(p.get("score", 0.0)),
            ),
            reverse=True,
        )
        survivors = []
        for proposal in ordered:
            family = str(proposal.get("proposal_family") or "")
            atom_ids = {int(v) for v in list(proposal.get("atom_id_list") or []) if int(v) > 0}
            drop = False
            for prev in survivors:
                prev_family = str(prev.get("proposal_family") or "")
                prev_atoms = {int(v) for v in list(prev.get("atom_id_list") or []) if int(v) > 0}
                lower_or_equal_priority = int(family_priority.get(family, 0)) <= int(family_priority.get(prev_family, 0))
                atom_subset = bool(atom_ids) and atom_ids.issubset(prev_atoms)
                if lower_or_equal_priority and atom_subset:
                    axis_a = np.asarray(proposal.get("axis_ras") or [0.0, 0.0, 1.0], dtype=float).reshape(3)
                    axis_b = np.asarray(prev.get("axis_ras") or [0.0, 0.0, 1.0], dtype=float).reshape(3)
                    dot = float(np.clip(abs(np.dot(axis_a, axis_b)), -1.0, 1.0))
                    angle_deg = float(np.degrees(np.arccos(dot)))
                    mid_a = np.asarray(proposal.get("midpoint_ras") or [0.0, 0.0, 0.0], dtype=float).reshape(3)
                    mid_b = np.asarray(prev.get("midpoint_ras") or [0.0, 0.0, 0.0], dtype=float).reshape(3)
                    lateral_a = self._radial_distance_to_line(mid_a.reshape(1, 3), np.asarray(prev.get("center_ras"), dtype=float), axis_b)[0]
                    lateral_b = self._radial_distance_to_line(mid_b.reshape(1, 3), np.asarray(proposal.get("center_ras"), dtype=float), axis_a)[0]
                    if angle_deg <= 12.0 and min(float(lateral_a), float(lateral_b)) <= 4.0:
                        drop = True
                        break
                if not self._token_line_candidates_overlap(proposal, prev):
                    continue
                same_extent = self._token_line_candidate_extent_similar(proposal, prev)
                loose_extent = self._token_line_candidate_extent_similar(
                    proposal,
                    prev,
                    endpoint_mm_thresh=16.0,
                    span_ratio_thresh=0.60,
                )
                if lower_or_equal_priority and (atom_subset or same_extent or loose_extent):
                    drop = True
                    break
            if not drop:
                survivors.append(dict(proposal))
        return survivors

    def _token_line_candidate_extent_similar(self, a, b, endpoint_mm_thresh=10.0, span_ratio_thresh=0.75):
        start_a = np.asarray(a.get("start_ras") or [0.0, 0.0, 0.0], dtype=float).reshape(3)
        end_a = np.asarray(a.get("end_ras") or [0.0, 0.0, 0.0], dtype=float).reshape(3)
        start_b = np.asarray(b.get("start_ras") or [0.0, 0.0, 0.0], dtype=float).reshape(3)
        end_b = np.asarray(b.get("end_ras") or [0.0, 0.0, 0.0], dtype=float).reshape(3)
        axis_a = np.asarray(a.get("axis_ras") or [0.0, 0.0, 1.0], dtype=float).reshape(3)
        axis_b = np.asarray(b.get("axis_ras") or [0.0, 0.0, 1.0], dtype=float).reshape(3)
        if float(np.dot(axis_a, axis_b)) < 0.0:
            start_b, end_b = end_b, start_b
        span_a = float(np.linalg.norm(end_a - start_a))
        span_b = float(np.linalg.norm(end_b - start_b))
        span_ratio = float(min(span_a, span_b) / max(1.0, max(span_a, span_b)))
        if span_ratio < float(span_ratio_thresh):
            return False
        start_err = float(np.linalg.norm(start_a - start_b))
        end_err = float(np.linalg.norm(end_a - end_b))
        return bool(start_err <= float(endpoint_mm_thresh) and end_err <= float(endpoint_mm_thresh))

    def _token_line_candidates_overlap(self, a, b, angle_deg_thresh=8.0, lateral_mm_thresh=3.0):
        tokens_a = set(int(v) for v in list(a.get("token_indices") or []))
        tokens_b = set(int(v) for v in list(b.get("token_indices") or []))
        overlap_ratio = 0.0
        if tokens_a and tokens_b:
            overlap = len(tokens_a & tokens_b)
            overlap_ratio = overlap / float(max(1, min(len(tokens_a), len(tokens_b))))
            if overlap / float(max(1, min(len(tokens_a), len(tokens_b)))) >= 0.35:
                if self._token_line_candidate_extent_similar(a, b):
                    return True
        blobs_a = set(int(v) for v in list(a.get("blob_id_list") or []))
        blobs_b = set(int(v) for v in list(b.get("blob_id_list") or []))
        shared_blob_count = len(blobs_a & blobs_b)
        axis_a = np.asarray(a.get("axis_ras") or [0.0, 0.0, 1.0], dtype=float).reshape(3)
        axis_b = np.asarray(b.get("axis_ras") or [0.0, 0.0, 1.0], dtype=float).reshape(3)
        dot = float(np.clip(abs(np.dot(axis_a, axis_b)), -1.0, 1.0))
        angle_deg = float(np.degrees(np.arccos(dot)))
        if angle_deg > float(angle_deg_thresh):
            return False
        mid_a = np.asarray(a.get("midpoint_ras") or [0.0, 0.0, 0.0], dtype=float).reshape(3)
        mid_b = np.asarray(b.get("midpoint_ras") or [0.0, 0.0, 0.0], dtype=float).reshape(3)
        lateral_a = self._radial_distance_to_line(mid_a.reshape(1, 3), np.asarray(b.get("center_ras"), dtype=float), axis_b)[0]
        lateral_b = self._radial_distance_to_line(mid_b.reshape(1, 3), np.asarray(a.get("center_ras"), dtype=float), axis_a)[0]
        if min(float(lateral_a), float(lateral_b)) > float(lateral_mm_thresh):
            return False
        # For deep-core debugging, preserve recall: do not suppress nearby lines
        # unless they reuse support tokens or parent blobs *and* represent the
        # same effective extent. Different-length ownership variants should
        # survive so debug recall does not collapse to stitched supersets.
        if shared_blob_count > 0 and self._token_line_candidate_extent_similar(a, b):
            return True
        shared_blob_ratio = float(shared_blob_count) / float(max(1, min(len(blobs_a), len(blobs_b))))
        span_a = float(a.get("span_mm", 0.0))
        span_b = float(b.get("span_mm", 0.0))
        if span_a <= span_b:
            shorter = a
            longer = b
        else:
            shorter = b
            longer = a
        shorter_span = float(shorter.get("span_mm", 0.0))
        longer_span = float(longer.get("span_mm", 0.0))
        shorter_cov = float(shorter.get("coverage", 0.0))
        longer_cov = float(longer.get("coverage", 0.0))
        shorter_gap = int(shorter.get("max_gap_bins", 0))
        longer_gap = int(longer.get("max_gap_bins", 0))
        # Suppress through-going variants that reuse the same core support but
        # append a sparse tail through a different corridor.
        if (
            max(float(overlap_ratio), float(shared_blob_ratio)) >= 0.70
            and longer_span >= 1.35 * max(1.0, shorter_span)
            and longer_cov + 0.20 <= shorter_cov
            and longer_gap >= max(6, shorter_gap + 4)
        ):
            return True
        return False
