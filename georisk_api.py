"""Read-only serving of validated GeoRisk artifacts; never unpickle models."""
import hashlib
import json
import math
import struct
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

METADATA = 'ecosignal_georisk_v2_1m_metadata.json'
OVERLAY = 'ecosignal_georisk_v2_overlay.json'
PNG = 'ecosignal_georisk_v2_overlay.png'
MODEL = 'ecosignal_georisk_v2_1m.joblib'


def read_verified_artifacts(root: Path):
    try:
        metadata = json.loads((root / METADATA).read_text(encoding='utf-8'))
        overlay = json.loads((root / OVERLAY).read_text(encoding='utf-8'))
        png = (root / PNG).read_bytes()
        if png[:8] != b'\x89PNG\r\n\x1a\n' or png[12:16] != b'IHDR':
            raise ValueError('Invalid PNG')
        dimensions = list(struct.unpack('>II', png[16:24]))
        if dimensions != overlay['image_dimensions_px'] or min(dimensions) <= 0:
            raise ValueError('PNG dimensions mismatch')
        bounds = overlay['bounds_wgs84']
        west, south, east, north = (float(bounds[k]) for k in ('west', 'south', 'east', 'north'))
        if not all(math.isfinite(v) for v in (west, south, east, north)):
            raise ValueError('Nonfinite bounds')
        if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
            raise ValueError('Invalid geographic bounds')
        if overlay['leaflet_image_overlay_bounds'] != [[south, west], [north, east]]:
            raise ValueError('Overlay bounds disagree')
        if overlay['score_scale'] != [0.0, 1.0] or overlay['data_year'] != 2024:
            raise ValueError('Unexpected score contract or survey year')
        if overlay.get('display_projection') != 'EPSG:3857':
            raise ValueError('Display raster must match Web Mercator map coordinates')
        if metadata['latest_map_input_year'] != 2024 or metadata['vertical_units'] != 'meters':
            raise ValueError('Unexpected source semantics')
        if metadata['source_grid']['resolution_m'] != 1:
            raise ValueError('Expected native 1 m raster')
        if metadata['prediction_semantics'] != 'relative uncalibrated susceptibility to observed terrain lowering':
            raise ValueError('Expected uncalibrated susceptibility semantics')
        metrics = metadata['temporal_spatial_holdout_2022_2024']
        if metrics['model'] != metadata['model_type']:
            raise ValueError('Model and evaluation identity disagree')
        if metrics['threshold'] != metadata['frozen_classification_threshold']:
            raise ValueError('Frozen threshold and evaluation disagree')
        if metadata['model_file'] != MODEL or metadata['source_grid']['crs'] != 'EPSG:32119':
            raise ValueError('Unexpected model artifact or native grid')
        for key in ('training_input_year', 'training_label_period', 'test_input_year',
                    'development_2015_2022', 'vertical_reference_status'):
            if key not in metadata:
                raise ValueError('Incomplete provenance')
        for name in ('roc_auc', 'pr_auc', 'precision', 'recall', 'f1', 'eligible_sample_prevalence'):
            value = float(metrics[name])
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError('Invalid evaluation metric')
        if int(metrics['n']) <= 0 or metadata['test_label_period'] != '2022-2024':
            raise ValueError('Missing held-out evaluation evidence')
        if not metadata['critical_limitations']:
            raise ValueError('Scientific limitations must accompany the layer')
        return metadata, overlay, hashlib.sha256(png).hexdigest()
    except (OSError, ValueError, KeyError, TypeError, struct.error) as exc:
        raise HTTPException(503, 'Verified geomorphology results are not available yet.') from exc


def create_georisk_router(asset_directory: Path):
    root = asset_directory.resolve()
    router = APIRouter(prefix='/geomorphology', tags=['GeoRisk V2'])

    @router.get('/health')
    def health():
        metadata, _, fingerprint = read_verified_artifacts(root)
        return {'status': 'ok', 'dataset': 'GeoRisk V2', 'data_year': 2024,
                'semantics': metadata['prediction_semantics'], 'overlay_sha256': fingerprint}

    @router.get('/manifest')
    def manifest():
        metadata, overlay, fingerprint = read_verified_artifacts(root)
        return {**overlay, 'image_url': '/geomorphology/overlay.png',
                'statistics_url': '/geomorphology/statistics',
                'model_download_url': '/geomorphology/model',
                'overlay_sha256': fingerprint,
                'model_type': metadata['model_type'],
                'native_raster_resolution_m': 1,
                'display_semantics': 'Downsampled visualization of a 1 m source raster',
                'vertical_reference_status': metadata['vertical_reference_status'],
                'critical_limitations': metadata['critical_limitations']}

    @router.get('/statistics')
    def statistics():
        metadata, _, _ = read_verified_artifacts(root)
        return {key: metadata[key] for key in (
            'model_type', 'training_input_year', 'training_label_period',
            'test_input_year', 'test_label_period', 'latest_map_input_year',
            'development_2015_2022', 'temporal_spatial_holdout_2022_2024',
            'frozen_classification_threshold', 'prediction_semantics',
            'vertical_reference_status', 'critical_limitations')}

    @router.get('/overlay.png')
    def image():
        _, _, fingerprint = read_verified_artifacts(root)
        return FileResponse(root / PNG, media_type='image/png',
                            headers={'Cache-Control': 'public, max-age=3600',
                                     'ETag': f'"{fingerprint}"'})

    @router.get('/model')
    def model():
        read_verified_artifacts(root)
        if not (root / MODEL).is_file():
            raise HTTPException(503, 'The trained model download is not available yet.')
        return FileResponse(root / MODEL, filename=MODEL, media_type='application/octet-stream')

    return router
