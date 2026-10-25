"""Explicit user-selected evidence. No network fetching, lookup, or raw image logging."""
import base64
import io
import warnings
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from PIL import Image, ImageOps

MAX_IMAGE_BYTES = 700_000


class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False, strict=True)


class Selection(StrictModel):
    image: bool = False
    coverage: bool = False
    growth_form: bool = False
    satellite: bool = False
    geomorphology: bool = False
    elevation: bool = False
    location: bool = False


class Point(StrictModel):
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)


class ImageInput(StrictModel):
    mime_type: Literal['image/jpeg', 'image/png', 'image/webp']
    data: str = Field(min_length=4, max_length=940_000, repr=False)

    @model_validator(mode='after')
    def validate_and_strip_metadata(self):
        try:
            raw = base64.b64decode(self.data, validate=True)
            if not 0 < len(raw) <= MAX_IMAGE_BYTES:
                raise ValueError()
            with warnings.catch_warnings():
                warnings.simplefilter('error', Image.DecompressionBombWarning)
                with Image.open(io.BytesIO(raw)) as image:
                    if image.format not in {'JPEG', 'PNG', 'WEBP'} or image.width * image.height > 16_000_000:
                        raise ValueError()
                    expected = {'JPEG': 'image/jpeg', 'PNG': 'image/png', 'WEBP': 'image/webp'}[image.format]
                    if expected != self.mime_type:
                        raise ValueError()
                    image.load()
                    clean = ImageOps.exif_transpose(image).convert('RGB')
                    clean.thumbnail((1280, 1280))
                    for quality in (80, 65, 50):
                        output = io.BytesIO()
                        clean.save(output, format='JPEG', quality=quality)
                        if output.tell() <= MAX_IMAGE_BYTES:
                            break
                    if output.tell() > MAX_IMAGE_BYTES:
                        raise ValueError()
                    self.data = base64.b64encode(output.getvalue()).decode('ascii')
                    self.mime_type = 'image/jpeg'
        except Exception:
            # No raw bytes or underlying decoder exception in validation messages.
            raise ValueError('Choose a valid JPEG, PNG or WebP image under 700 KB after resizing.') from None
        return self


class Provenance(StrictModel):
    source: str = Field(min_length=1, max_length=160)
    observed_at: str | None = Field(default=None, max_length=80)
    version: str | None = Field(default=None, max_length=120)


class Coverage(Provenance):
    percent: float = Field(ge=0, le=100)
    confidence: float | None = Field(default=None, ge=0, le=1)


class GrowthForm(Provenance):
    label: str = Field(min_length=1, max_length=80)
    confidence: float | None = Field(default=None, ge=0, le=1)


class Satellite(Provenance):
    ndvi: float | None = Field(default=None, ge=-1, le=1)
    ndmi: float | None = Field(default=None, ge=-1, le=1)
    bsi: float | None = Field(default=None, ge=-1, le=1)
    mndwi: float | None = Field(default=None, ge=-1, le=1)
    resolution_m: float | None = Field(default=None, gt=0, le=10000)
    patch_pixels: int | None = Field(default=None, ge=1, le=10000)
    acquisition_dates: list[str] = Field(default_factory=list, max_length=20)

    @model_validator(mode='after')
    def nonempty(self):
        if all(getattr(self, key) is None for key in ('ndvi', 'ndmi', 'bsi', 'mndwi')):
            raise ValueError('No satellite indices supplied.')
        if any(len(value) > 80 for value in self.acquisition_dates):
            raise ValueError('Invalid acquisition date.')
        return self


class Geomorphology(Provenance):
    susceptibility: float = Field(ge=0, le=1)
    percentile: float | None = Field(default=None, ge=0, le=100)
    reference_year: int | None = Field(default=None, ge=1900, le=2100)
    sample_distance_m: float | None = Field(default=None, ge=0, le=100000)
    sample_spacing_m: float | None = Field(default=None, gt=0, le=100000)


class Elevation(Provenance):
    metres: float = Field(ge=-500, le=9000)
    resolution_m: float | None = Field(default=None, gt=0, le=10000)


class ObservationContext(StrictModel):
    observation_id: str = Field(min_length=1, max_length=128)
    selection: Selection
    image: ImageInput | None = None
    coverage: Coverage | None = None
    growth_form: GrowthForm | None = None
    satellite: Satellite | None = None
    geomorphology: Geomorphology | None = None
    elevation: Elevation | None = None
    location: Point | None = None

    @model_validator(mode='after')
    def enforce_explicit_selection(self):
        selected = self.selection.model_dump()
        if not any(selected.values()):
            raise ValueError('Select at least one item or send without an observation.')
        for key, enabled in selected.items():
            if enabled != (getattr(self, key) is not None):
                raise ValueError(f'Selection does not match supplied {key}.')
        return self

    @field_validator('observation_id')
    @classmethod
    def nonblank_id(cls, value):
        if not value.strip():
            raise ValueError('Choose a collected observation.')
        return value.strip()

    def receipt(self):
        return {'version':'observation-context-v1', 'observation_id':self.observation_id,
                'included':[key for key, enabled in self.selection.model_dump().items() if enabled],
                'image_received':self.image is not None}

    def text_payload(self):
        result = self.model_dump(exclude={'image'}, exclude_none=True)
        result['provenance_status'] = 'User-selected submitted observation; not independently verified'
        if self.image:
            result['image_attached'] = True
        return result

    def provider_image_part(self):
        if not self.image:
            return None
        return {'inline_data': {'mime_type': self.image.mime_type, 'data': base64.b64decode(self.image.data)}}

    def evidence_rows(self):
        source = f'User-selected observation {self.observation_id}; submitted, not independently verified'
        rows = []
        if self.image:
            rows.append({'label': 'Attached photograph', 'value': 'Supplied for visual description; not a measured prediction', 'source': source})
        labels = {'coverage': 'Visible coverage', 'growth_form': 'Growth form', 'satellite': 'Satellite indices',
                  'geomorphology': 'Geomorphology susceptibility', 'elevation': 'Elevation', 'location': 'Selected coordinates'}
        import json
        for key, label in labels.items():
            value = getattr(self, key)
            if value is not None:
                rows.append({'label': label, 'value': json.dumps(value.model_dump(exclude_none=True), ensure_ascii=False), 'source': source})
        return rows
