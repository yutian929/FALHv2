#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import numpy as np
import cv2
from typing import List, Dict, Any, Optional, Tuple
from scipy.spatial.transform import Rotation as R

# 注意：语义分割器延迟导入（只有 obj.txt 找不到才 import）


# ============================
# 配置
# ============================
DATA_DIR = "/home/yutian/FALHv2/data/output"
PRIOR_ROOT = os.path.join(DATA_DIR, "prior")
OUT_ROOT = os.path.join(DATA_DIR, "prior_results")

SEMANTIC_SEGMENTOR_TYPE = "sam3"
SEG_OPTIONS = {
    "lang_sam": {
        "sam_type": "sam2.1_hiera_large",
        "box_threshold": 0.8,
        "text_threshold": 0.3,
    },
    "sam3": {}
}
BATCH_MAX_SIZE = 4

TOPK = 20  # 保存前 K 个命中结果；只要最优就设 1


# ============================
# utils
# ============================
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def _sanitize_name(s: str) -> str:
    s = s.strip()
    s = re.sub(r"[^\w\u4e00-\u9fff\-\. ]+", "_", s)  # 允许中文/字母数字/下划线/横线/点/空格
    s = re.sub(r"\s+", "_", s)
    return s[:80] if len(s) > 80 else s


def list_prior_indices(prior_root: str) -> List[int]:
    if not os.path.isdir(prior_root):
        return []
    idxs = []
    for name in os.listdir(prior_root):
        p = os.path.join(prior_root, name)
        if os.path.isdir(p) and name.isdigit():
            idxs.append(int(name))
    return sorted(idxs)


def read_obj_txt(prior_dir: str) -> List[str]:
    p = os.path.join(prior_dir, "obj.txt")
    if not os.path.exists(p):
        return []
    lines = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if not t or t.startswith("#"):
                continue
            lines.append(t)
    return lines


def match_query_in_obj(query: str, obj_lines: List[str]) -> Optional[str]:
    q = _norm(query)
    for o in obj_lines:
        oo = _norm(o)
        if not oo:
            continue
        if q == oo or q in oo or oo in q:
            return o
    return None


def load_rgb_from_prior(prior_dir: str) -> Optional[np.ndarray]:
    """
    返回 RGB uint8 (H,W,3), contiguous
    """
    npy = os.path.join(prior_dir, "rgb.npy")
    if os.path.exists(npy):
        rgb = np.load(npy)
    else:
        png = os.path.join(prior_dir, "rgb.png")
        if not os.path.exists(png):
            return None
        bgr = cv2.imread(png, cv2.IMREAD_COLOR)
        if bgr is None:
            return None
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    if rgb is None:
        return None
    if rgb.ndim == 3 and rgb.shape[0] == 3 and rgb.shape[-1] != 3:
        rgb = np.transpose(rgb, (1, 2, 0))
    if rgb.dtype != np.uint8:
        mx = float(np.max(rgb)) if rgb.size else 0.0
        if mx <= 1.5:
            rgb = rgb * 255.0
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    rgb = np.ascontiguousarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        return None
    return rgb


def load_pose_from_prior(prior_dir: str) -> Optional[np.ndarray]:
    npy = os.path.join(prior_dir, "cam_T_world.npy")
    if os.path.exists(npy):
        T = np.load(npy)
        if T.shape == (4, 4):
            return T.astype(np.float64)

    js = os.path.join(prior_dir, "cam_T_world.json")
    if os.path.exists(js):
        with open(js, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "cam_T_world" in data:
            T = np.array(data["cam_T_world"], dtype=np.float64)
            if T.shape == (4, 4):
                return T
    return None


def pose_pack(T: Optional[np.ndarray]) -> Dict[str, Any]:
    if T is None:
        return {"T": None, "t": None, "quat_xyzw": None, "euler_xyz_deg": None}
    t = T[:3, 3].astype(float)
    rot = R.from_matrix(T[:3, :3])
    quat = rot.as_quat().astype(float)   # [x,y,z,w]
    euler = rot.as_euler("xyz", degrees=True).astype(float)
    return {
        "T": T.tolist(),
        "t": t.tolist(),
        "quat_xyzw": quat.tolist(),
        "euler_xyz_deg": euler.tolist()
    }


def draw_text(img_bgr: np.ndarray, lines: List[str], org=(10, 25), line_h=22):
    x, y = org
    for i, s in enumerate(lines):
        yy = y + i * line_h
        cv2.putText(img_bgr, s, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img_bgr, s, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return img_bgr


def draw_boxes(img_bgr: np.ndarray, boxes: Any, label: str, scores: Any = None):
    if boxes is None:
        return img_bgr
    boxes = np.array(boxes)
    if boxes.ndim != 2 or boxes.shape[1] < 4:
        return img_bgr
    H, W = img_bgr.shape[:2]
    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = [int(round(v)) for v in b[:4]]
        x1 = max(0, min(W - 1, x1))
        x2 = max(0, min(W - 1, x2))
        y1 = max(0, min(H - 1, y1))
        y2 = max(0, min(H - 1, y2))
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

        s_txt = ""
        if scores is not None and len(scores) > i:
            try:
                s_txt = f" {float(scores[i]):.2f}"
            except Exception:
                pass
        cv2.putText(img_bgr, f"{label}{s_txt}", (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    return img_bgr


def save_hit(out_dir: str, rank: int, hit: Dict[str, Any]):
    """
    hit: {
      index, prior_dir, query, method,
      matched_name?, label?, score?, boxes?, scores?,
      pose: {...}
    }
    """
    os.makedirs(out_dir, exist_ok=True)

    idx = hit["index"]
    method = hit["method"]
    query = hit["query"]

    prior_dir = hit["prior_dir"]
    rgb = load_rgb_from_prior(prior_dir)
    if rgb is None:
        return

    img_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    if method == "semantic":
        img_bgr = draw_boxes(img_bgr, hit.get("boxes"), hit.get("label", query), hit.get("scores"))

    # overlay 文字信息
    lines = [
        f"rank={rank:03d}  index={idx}",
        f"query={query}",
        f"method={method}",
    ]
    if method == "obj.txt":
        lines.append(f"matched={hit.get('matched_name','')}")
    if method == "semantic":
        lines.append(f"label={hit.get('label','')} score={float(hit.get('score',0.0)):.3f}")

    pose = hit.get("pose", {})
    if pose and pose.get("t") is not None:
        t = pose["t"]
        q = pose["quat_xyzw"]
        e = pose["euler_xyz_deg"]
        lines.append(f"t=[{t[0]:.3f},{t[1]:.3f},{t[2]:.3f}]")
        lines.append(f"q=[{q[0]:.4f},{q[1]:.4f},{q[2]:.4f},{q[3]:.4f}]")
        lines.append(f"euler=[{e[0]:.1f},{e[1]:.1f},{e[2]:.1f}]")
    else:
        lines.append("pose=MISSING")

    img_bgr = draw_text(img_bgr, lines)

    img_name = f"hit_{rank:03d}_idx{idx}_{method}.png"
    json_name = f"hit_{rank:03d}_idx{idx}.json"

    cv2.imwrite(os.path.join(out_dir, img_name), img_bgr)
    with open(os.path.join(out_dir, json_name), "w", encoding="utf-8") as f:
        json.dump(hit, f, ensure_ascii=False, indent=2)


# ============================
# 主流程
# ============================
def main():
    query = input("请输入要查找的物体名称：").strip()
    if not query:
        print("Empty query. Exit.")
        return
    if not os.path.isdir(PRIOR_ROOT):
        print(f"Prior root not found: {PRIOR_ROOT}")
        return

    indices = list_prior_indices(PRIOR_ROOT)
    if not indices:
        print("No prior indices found.")
        return

    out_dir = os.path.join(OUT_ROOT, _sanitize_name(query))
    os.makedirs(out_dir, exist_ok=True)

    # 1) obj.txt 优先
    obj_hits = []
    for idx in indices:
        d = os.path.join(PRIOR_ROOT, str(idx))
        matched = match_query_in_obj(query, read_obj_txt(d))
        if matched is None:
            continue
        T = load_pose_from_prior(d)
        obj_hits.append({
            "index": idx,
            "prior_dir": d,
            "query": query,
            "method": "obj.txt",
            "matched_name": matched,
            "pose": pose_pack(T),
        })

    if obj_hits:
        # obj.txt 命中通常你只需要这些，不用跑语义
        obj_hits.sort(key=lambda x: x["index"])
        keep = obj_hits[:TOPK]
        for r, hit in enumerate(keep):
            save_hit(out_dir, r, hit)

        with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump({"query": query, "method": "obj.txt", "hits": keep}, f, ensure_ascii=False, indent=2)

        print(f"[DONE] Found in obj.txt: {len(obj_hits)} hits. Saved top {len(keep)} to: {out_dir}")
        return

    # 2) 语义分割（仅当 obj.txt 没找到）
    print("Not found in any obj.txt. Running semantic segmentation...")

    # 延迟导入，避免不必要的 torch 加载
    from falhv2.semantic_segmentor import SemanticSegmentQuery, SemanticSegmentor

    query_images = []
    pos2idx = []
    pos2dir = []

    for idx in indices:
        d = os.path.join(PRIOR_ROOT, str(idx))
        rgb = load_rgb_from_prior(d)
        if rgb is None:
            continue
        query_images.append(rgb)
        pos2idx.append(idx)
        pos2dir.append(d)

    if not query_images:
        print("No rgb found in prior.")
        return

    print(f"Collected {len(query_images)} images. Initializing segmentor...")
    segmentor = SemanticSegmentor(SEMANTIC_SEGMENTOR_TYPE, SEG_OPTIONS.get(SEMANTIC_SEGMENTOR_TYPE, {}))

    seg_query = SemanticSegmentQuery(query_images, [query])
    print(f"Running predict on {len(query_images)} images ...")
    response = segmentor.predict(seg_query, batch_max_size=BATCH_MAX_SIZE)
    results = response._dict_results()

    sem_hits = []
    for pos in range(len(query_images)):
        key = f"image_{pos}"
        if key not in results:
            continue
        per_img = results[key]

        best = None
        for label, data in per_img.items():
            boxes = data.get("boxes", [])
            if boxes is None or len(boxes) == 0:
                continue
            scores = data.get("scores", [])
            s = float(np.max(np.array(scores))) if scores is not None and len(scores) > 0 else 1.0
            if best is None or s > best["score"]:
                best = {
                    "label": label,
                    "score": s,
                    "boxes": np.array(boxes).tolist() if boxes is not None else None,
                    "scores": np.array(scores).tolist() if scores is not None else None,
                }

        if best is not None:
            idx = pos2idx[pos]
            d = pos2dir[pos]
            T = load_pose_from_prior(d)
            sem_hits.append({
                "index": idx,
                "prior_dir": d,
                "query": query,
                "method": "semantic",
                **best,
                "pose": pose_pack(T),
            })

    if not sem_hits:
        with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump({"query": query, "method": "semantic", "hits": []}, f, ensure_ascii=False, indent=2)
        print(f"[DONE] Semantic search: no detections. Summary saved to: {out_dir}")
        return

    sem_hits.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    keep = sem_hits[:TOPK]

    for r, hit in enumerate(keep):
        save_hit(out_dir, r, hit)

    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"query": query, "method": "semantic", "hits": keep}, f, ensure_ascii=False, indent=2)

    print(f"[DONE] Semantic hits: {len(sem_hits)}. Saved top {len(keep)} to: {out_dir}")


if __name__ == "__main__":
    main()
