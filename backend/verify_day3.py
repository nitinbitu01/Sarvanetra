"""Day 3 verification script — run via: python backend/verify_day3.py"""
import sys, json
import numpy as np
sys.path.insert(0, '.')
from pathlib import Path
from backend import db
from backend.embedding_utils import encode_embedding, decode_embedding, normalize_l2, is_valid_embedding
from backend.faiss_index import FaissReIDIndex
import yaml

cfg = yaml.safe_load(open('config.yaml'))
db_path = str(Path('output/sentinel.db'))
print('=== DAY 3 FULL VERIFICATION ===\n')

# 1. Row counts
counts = db.get_table_counts(db_path)
n_cam = counts['cameras']
n_per = counts['watchlist_persons']
n_veh = counts['watchlist_vehicles']
assert n_cam == 6,  "Expected 6 cameras, got %d" % n_cam
assert n_per == 1,  "Expected 1 person, got %d"  % n_per
assert n_veh == 1,  "Expected 1 vehicle, got %d" % n_veh
print('[OK] Row counts: cameras=%d persons=%d vehicles=%d' % (n_cam, n_per, n_veh))

# 2. Embedding round-trip
row = db.get_active_watchlist_persons(db_path)[0]
blob = bytes(row['face_embedding'])
dim  = row['embedding_dim']
vec  = decode_embedding(blob, dim)
assert vec.shape == (512,),     "Shape wrong: %s" % str(vec.shape)
assert vec.dtype == np.float32, "Dtype wrong: %s" % str(vec.dtype)
assert is_valid_embedding(vec), "Embedding invalid (NaN/zeros/inf)"
print('[OK] Embedding round-trip: shape=%s dtype=%s' % (vec.shape, vec.dtype))
print('     First 3 values: %s' % str(vec[:3].tolist()))

# 3. Double round-trip
re_blob = encode_embedding(vec)
re_vec  = decode_embedding(re_blob, dim)
assert np.allclose(vec, re_vec), "Double round-trip changed values!"
print('[OK] Double round-trip: values unchanged')

# 4. FAISS loaded
idx = FaissReIDIndex.load(cfg)
assert idx.ntotal == 15, "Expected 15 FAISS vectors, got %d" % idx.ntotal
print('[OK] FAISS index: %d vectors, dim=%d' % (idx.ntotal, idx.dim))

# 5. FAISS self-queries — all 15 vectors
id_map = idx.get_id_map()
all_scores = []
for meta in id_map:
    npy = str(Path(meta['npy_path']))
    vec_i = np.load(npy).astype(np.float32).flatten()
    results = idx.search(vec_i, k=1)
    score = results[0]['score']
    returned_clip = results[0]['clip_id']
    assert abs(score - 1.0) < 0.001, "%s self-query score=%.6f (expected approx 1.0)" % (meta['clip_id'], score)
    assert returned_clip == meta['clip_id'], "Wrong clip: got %s expected %s" % (returned_clip, meta['clip_id'])
    all_scores.append(score)
print('[OK] FAISS self-queries (15/15): score range [%.6f, %.6f] all approx 1.0' % (min(all_scores), max(all_scores)))

# 6. normalize_l2 properties
v  = normalize_l2(vec)
v2 = normalize_l2(v)
assert np.allclose(v, v2, atol=1e-6), "normalize_l2 not idempotent!"
assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-6, "Not unit norm after normalize!"
print('[OK] normalize_l2: idempotent, unit-norm confirmed')

# 7. WAL mode
assert db.verify_wal_mode(db_path), "WAL mode NOT active!"
print('[OK] SQLite WAL mode: active')

# 8. Plate lookup
plate_row = db.is_plate_on_watchlist(db_path, 'GJ05AB1234')
assert plate_row is not None, "Plate GJ05AB1234 not found!"
assert plate_row['reason'] == 'stolen', "Wrong reason: %s" % plate_row['reason']
print('[OK] Plate lookup: GJ05AB1234 found, reason=%s' % plate_row['reason'])

# 9. Alert + metadata JSON round-trip
alert_id = db.insert_alert(
    db_path, alert_type='watchlist_vehicle',
    camera_id='CAM-01', score=0.97, severity='critical',
    metadata={'plate': 'GJ05AB1234', 'anpr_confidence': 0.94}
)
alerts = db.get_alerts(db_path, status='new')
matched = [a for a in alerts if a['id'] == alert_id]
assert matched, "Alert id=%d not found in get_alerts()" % alert_id
meta_back = json.loads(matched[0]['metadata'])
assert meta_back['plate'] == 'GJ05AB1234'
assert abs(meta_back['anpr_confidence'] - 0.94) < 1e-9
print('[OK] Alert insert + metadata JSON round-trip')

# 10. id_map.json structure
id_map_data = json.loads(open('output/faiss/id_map.json').read())
assert len(id_map_data) == 15, "Expected 15 id_map entries, got %d" % len(id_map_data)
required_keys = {'clip_id', 'label', 'camera_id', 'npy_path'}
for entry in id_map_data:
    missing = required_keys - set(entry.keys())
    assert not missing, "id_map entry missing keys: %s" % str(missing)
print('[OK] id_map.json: 15 entries, all required keys present')

print('\n' + '='*50)
print('ALL 10 VERIFICATION CHECKS PASSED')
print('Day 3 foundation ready for Day 6 (ReID) and Day 7 (face match).')
print('='*50)
