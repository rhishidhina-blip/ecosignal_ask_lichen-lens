"""Synthetic software fixtures only. No uploaded user photos or billable calls."""
import base64,io,json,contextlib,tempfile,unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
from PIL import Image
from pydantic import ValidationError
from google.genai import types
from fastapi.testclient import TestClient
import main as backend
from observation_context import ObservationContext, ImageInput
from semantic_checks import FakeGemini

KEYS=['image','coverage','growth_form','satellite','geomorphology','elevation','location']
def photo(progressive=False):
    i=Image.new('RGB',(32,24),(60,80,70));e=Image.Exif();e[270]='PRIVATE_METADATA_SENTINEL';e[274]=6
    out=io.BytesIO();i.save(out,format='JPEG',exif=e,progressive=progressive)
    return out.getvalue()+b'PRIVATE_TRAILING_SENTINEL'
def fields():
    return dict(image={'mime_type':'image/jpeg','data':base64.b64encode(photo()).decode()},
      coverage={'source':'Synthetic software test','percent':42.0},growth_form={'source':'Synthetic software test','label':'foliose'},
      satellite={'source':'Synthetic software test','ndvi':.5,'ndmi':.1,'bsi':-.2,'mndwi':-.4,'observed_at':'2026-09-28','resolution_m':20.0,'patch_pixels':3},
      geomorphology={'source':'Synthetic software test','susceptibility':.2,'percentile':35.0,'reference_year':2024,'sample_distance_m':5.0,'sample_spacing_m':20.0},
      elevation={'source':'Synthetic software test','metres':112.0,'resolution_m':90.0},location={'lat':35.8,'lon':-78.7})
def context(keys):
    values=fields()
    return {'observation_id':'SYNTHETIC-SOFTWARE-ONLY','selection':{k:k in keys for k in KEYS},**{k:values[k] for k in keys}}

class MultimodalTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();root=Path(self.temp.name)
        settings=replace(backend.SETTINGS,root=root,production_db=root/'none.db',knowledge_dir=root/'knowledge',chatbot_db=root/'audit.db')
        with contextlib.redirect_stdout(io.StringIO()):self.assistant=backend.AskLichenLens(api_key=None,settings=settings)
    def tearDown(self):self.temp.cleanup()
    def ask(self,ctx):
        fake=FakeGemini(intent='scan_explanation',needs=True);self.assistant.client=fake
        with contextlib.redirect_stdout(io.StringIO()):reply=self.assistant.ask(backend.AskRequest(question='Explain selected evidence',observation_context=ctx),audit=False)
        self.assertEqual(reply.status,'ok');self.assertEqual(len(fake.calls),1)
        return reply,fake.calls[0]
    def test_each_category_and_combined_selection_reach_one_provider_call(self):
        for keys in [[k] for k in KEYS]+[KEYS]:
            with self.subTest(keys=keys):
                ctx=ObservationContext.model_validate(context(keys));reply,call=self.ask(ctx)
                contents=call['contents'];payload=json.loads(contents if isinstance(contents,str) else contents[0]['parts'][0]['text'])
                selected=payload['selected_observation']
                for key in KEYS:
                    if key!='image':self.assertEqual(key in selected,key in keys)
                self.assertNotIn('data',selected)
                self.assertEqual(reply.context_receipt,ctx.receipt())
                self.assertFalse(self.assistant._answer_cache)
                if 'image'in keys:
                    part=contents[0]['parts'][1]
                    validated=types.Content.model_validate(contents[0])
                    self.assertIsInstance(validated.parts[1].inline_data.data,bytes)
                    self.assertNotIn(b'PRIVATE_METADATA_SENTINEL',part['inline_data']['data'])
                    self.assertNotIn(b'PRIVATE_TRAILING_SENTINEL',part['inline_data']['data'])
                    self.assertNotIn(base64.b64encode(photo()).decode(),json.dumps(payload))
    def test_all_selection_combinations_have_exact_receipt(self):
        for mask in range(1,128):
            keys=[k for i,k in enumerate(KEYS) if mask&(1<<i)]
            ctx=ObservationContext.model_validate(context(keys))
            self.assertEqual(ctx.receipt()['included'],keys)
            self.assertEqual(ctx.receipt()['image_received'],'image'in keys)
    def test_invalid_hidden_missing_unselected_numeric_and_blank_inputs_rejected(self):
        for mutate in [lambda c:c.update(hidden='private'),lambda c:c.pop('coverage'),lambda c:c.update(location=fields()['location']),lambda c:c['coverage'].update(percent=101.0),lambda c:c.update(observation_id=' '),lambda c:c['selection'].update(image='false')]:
            ctx=context(['coverage']);mutate(ctx)
            with self.assertRaises(ValidationError):ObservationContext.model_validate(ctx)
    def test_jpeg_orientation_progressive_metadata_trailing_and_valid_decode(self):
        for progressive in [False,True]:
            v=ImageInput(mime_type='image/jpeg',data=base64.b64encode(photo(progressive)).decode());data=base64.b64decode(v.data)
            self.assertNotIn(b'PRIVATE_',data)
            with Image.open(io.BytesIO(data))as image:self.assertEqual(image.size,(24,32));self.assertFalse(image.getexif())
    def test_invalid_image_mime_size_and_dimensions(self):
        for value in [{'mime_type':'image/jpeg','data':base64.b64encode(b'not image').decode()}, {'mime_type':'image/png','data':base64.b64encode(photo()).decode()}, {'mime_type':'image/jpeg','data':'a'*940001}]:
            with self.assertRaises(ValidationError):ImageInput(**value)
        image=Image.new('1',(5000,4000));out=io.BytesIO();image.save(out,format='PNG')
        with self.assertRaises(ValidationError):ImageInput(mime_type='image/png',data=base64.b64encode(out.getvalue()).decode())
    def test_api_receipt_and_safe_validation_error(self):
        self.assistant.client=FakeGemini(intent='scan_explanation',needs=True)
        with patch.object(backend,'assistant',self.assistant),TestClient(backend.api)as client,contextlib.redirect_stdout(io.StringIO()):
            r=client.post('/assistant/ask',json={'question':'Explain selected evidence','observation_context':context(['image'])})
            self.assertEqual(r.status_code,200);self.assertEqual(r.json()['context_receipt']['included'],['image'])
            invalid=context(['image']);invalid['image']['data']='PRIVATE_BAD_BASE64_SENTINEL'
            r=client.post('/assistant/ask',json={'question':'Question','observation_context':invalid})
            self.assertEqual(r.status_code,422);self.assertNotIn('PRIVATE_BAD_BASE64_SENTINEL',r.text)
    def test_streamed_body_limit_and_text_only_compatibility(self):
        self.assistant.client=FakeGemini()
        with patch.object(backend,'assistant',self.assistant),TestClient(backend.api)as client,contextlib.redirect_stdout(io.StringIO()):
            def chunks():
                for _ in range(13):yield b'x'*100000
            r=client.post('/assistant/ask',content=chunks(),headers={'content-type':'application/json'})
            self.assertEqual(r.status_code,413)
            r=client.post('/assistant/ask',json={'question':'What are lichens?'})
            self.assertEqual(r.status_code,200);self.assertIsNone(r.json()['context_receipt'])
            h=client.get('/assistant/health').json();self.assertTrue(h['image_context_supported'])
            self.assertEqual(client.get('/assistant/audit/verify').status_code,404)

if __name__=='__main__':unittest.main()
