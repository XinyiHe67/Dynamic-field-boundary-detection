from torch.utils.data import Dataset

class RSSubfieldDataset(Dataset):
    try:
        import cv2
        _HAS_CV2 = True
    except Exception:
        _HAS_CV2 = False

    def _load_xanylabeling_json(p):
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def _polygon_to_mask(H, W, poly_xy):
        m = np.zeros((H, W), dtype=np.uint8)
        if len(poly_xy) < 3:
            return m
        if _HAS_CV2:
            pts = np.array(poly_xy, dtype=np.float32).reshape(-1, 1, 2)
            cv2.fillPoly(m, [pts.astype(np.int32)], 1)
            return m
        # 兜底：纯 shapely（慢）
        poly = Polygon(poly_xy)
        yy, xx = np.indices((H, W))
        coords = np.stack([xx.ravel(), yy.ravel()], axis=1)
        inside = np.array([poly.contains(Point(x, y)) for x, y in coords], dtype=np.uint8)
        return inside.reshape(H, W)
    
    def _bbox_from_mask(m):
        ys, xs = np.where(m > 0)
        if ys.size == 0: return None
        ymin, ymax = ys.min(), ys.max()
        xmin, xmax = xs.min(), xs.max()
        return np.array([xmin, ymin, xmax + 1, ymax + 1], dtype=np.float32)