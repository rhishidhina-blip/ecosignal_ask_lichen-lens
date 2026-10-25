"""Data-only regression tests; synthetic replies, no Gemini network requests."""
import contextlib,io,json,tempfile,unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
from pydantic import ValidationError
from fastapi.testclient import TestClient
import main as backend
from observation_context import ObservationContext
from semantic_checks import FakeGemini,ProviderError
KEYS=['coverage','growth_form','satellite','geomorphology','elevation','location']
FIELDS={'coverage':{'source':'Synthetic test','percent':42.0},'growth_form':{'source':'Synthetic test','label':'foliose'},'satellite':{'source':'Synthetic test','ndvi':.5},'geomorphology':{'source':'Synthetic test','susceptibility':.2},'elevation':{'source':'Synthetic test','metres':112.0},'location':{'lat':35.8,'lon':-78.7}}
def context(keys):return {'observation_id':'SYNTHETIC-ONLY','selection':{'image':False,**{k:k in keys for k in KEYS}},**{k:FIELDS[k] for k in keys}}
class DataOnlyTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();root=Path(self.temp.name)
  self.assistant=backend.AskLichenLens(replace(backend.SETTINGS,root=root,production_db=root/'none',chatbot_db=root/'audit.db'),api_key=None)
 def tearDown(self):self.temp.cleanup()
 def test_all_63_combinations_and_exact_receipts(self):
  for bits in range(1,64):
   keys=[k for i,k in enumerate(KEYS) if bits&(1<<i)];ctx=ObservationContext.model_validate(context(keys))
   self.assertEqual(ctx.receipt()['included'],keys);self.assertFalse(ctx.receipt()['image_received']);self.assertIsNone(ctx.provider_image_part())
 def test_images_rejected_even_with_data(self):
  for image in [{'mime_type':'image/jpeg','data':'PRIVATE'},'https://example.test/photo.jpg']:
   raw=context(['coverage']);raw['image']=image
   with self.assertRaises(ValidationError):ObservationContext.model_validate(raw)
  raw=context(['coverage']);raw['selection']['image']=True
  with self.assertRaises(ValidationError):ObservationContext.model_validate(raw)
 def test_data_and_history_reach_text_only_generation(self):
  f=FakeGemini(intent='scan_explanation',needs=True);self.assistant.client=f
  request=backend.AskRequest(question='Explain selected evidence',observation_context=context(KEYS),conversation_history=[{'role':'user','content':'Earlier question'},{'role':'assistant','content':'Earlier answer'}])
  r=self.assistant.ask(request,audit=False);self.assertEqual(r.status,'ok');self.assertEqual(len(f.calls),1)
  self.assertIsInstance(f.calls[0]['contents'],str);payload=json.loads(f.calls[0]['contents']);self.assertEqual(len(payload['conversation_history']),2)
  self.assertNotIn('image',payload['selected_observation']);self.assertEqual(r.context_receipt['included'],KEYS)
 def test_api_health_and_no_photo_errors(self):
  self.assistant.client=FakeGemini()
  with patch.object(backend,'assistant',self.assistant),TestClient(backend.api)as client:
   h=client.get('/assistant/health').json();self.assertFalse(h['image_context_supported']);self.assertTrue(h['data_context_supported'])
   raw=context(['coverage']);raw['image']={'data':'PRIVATE_SENTINEL','mime_type':'image/jpeg'}
   r=client.post('/assistant/ask',json={'question':'Explain','observation_context':raw});self.assertEqual(r.status_code,422);self.assertNotIn('PRIVATE_SENTINEL',r.text)
   self.assertEqual(client.post('/assistant/ask',json={'question':'Why are lichens useful?'}).status_code,200)
 def test_retry_waits_eight_seconds_and_uses_same_model(self):
  f=FakeGemini(errors=[ProviderError(503)]);self.assistant.client=f
  with patch.dict(backend.os.environ,{'ECOSIGNAL_GEMINI_BACKOFF_SECONDS':'8','ECOSIGNAL_CHAT_FALLBACK_MODELS':'','ECOSIGNAL_GEMINI_MAX_ATTEMPTS':'2'}),patch.object(backend.time,'sleep')as sleep,patch.object(backend.random,'uniform',return_value=0):
   r=self.assistant.ask(backend.AskRequest(question='Why are lichens useful?'),audit=False)
  self.assertEqual(r.status,'ok');sleep.assert_called_once_with(8);self.assertEqual([c['model']for c in f.calls],[backend.SETTINGS.chat_model]*2)
if __name__=='__main__':unittest.main()
