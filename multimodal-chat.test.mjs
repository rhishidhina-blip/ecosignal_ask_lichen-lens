import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import ts from 'typescript';
const moduleUrl=async(path,replacements={})=>{let s=await readFile(new URL(path,import.meta.url),'utf8');for(const[a,b]of Object.entries(replacements))s=s.replaceAll(a,b);return 'data:text/javascript;base64,'+Buffer.from(ts.transpileModule(s,{compilerOptions:{module:ts.ModuleKind.ESNext,target:ts.ScriptTarget.ES2022}}).outputText).toString('base64')};
const ctxUrl=await moduleUrl('../app/lib/observation-context.ts',{'./vendor/jpeg-encoder':await moduleUrl('../app/lib/vendor/jpeg-encoder.ts'),'jpeg-js':new URL('../node_modules/jpeg-js/index.js',import.meta.url).href});
const C=await import(ctxUrl);
const knowledgeUrl='data:text/javascript;base64,'+Buffer.from('export default '+await readFile(new URL('../app/lib/knowledge.json',import.meta.url),'utf8')).toString('base64');
const geminiUrl=await moduleUrl('../app/lib/gemini.ts',{'./observation-context':ctxUrl,'./knowledge.json':knowledgeUrl});
const routeUrl=await moduleUrl('../app/lib/assistant.ts',{'./observation-context':ctxUrl,'./knowledge.json':knowledgeUrl,'./gemini':geminiUrl,'./chat-request':await moduleUrl('../app/lib/chat-request.ts'),'./chat-response':await moduleUrl('../app/lib/chat-response.ts')});
const {handleAsk,boundedText}=await import(routeUrl);
const imageData='PRIVATE_IMAGE_SENTINEL';
const fields={image:{mime_type:'image/jpeg',data:imageData},coverage:{source:'Synthetic software fixture',percent:42},growth_form:{source:'Synthetic software fixture',label:'foliose'},satellite:{source:'Synthetic software fixture',ndvi:.5,ndmi:.1,bsi:-.2,mndwi:-.4,observed_at:'2026-09-28',resolution_m:20,patch_pixels:3},geomorphology:{source:'Synthetic software fixture',susceptibility:.2,percentile:35,reference_year:2024,sample_distance_m:5,sample_spacing_m:20},elevation:{source:'Synthetic software fixture',metres:112,resolution_m:90},location:{lat:35.8,lon:-78.7}};
function context(keys){return {observation_id:'SYNTHETIC-SOFTWARE-ONLY',selection:{...C.emptySelection(),...Object.fromEntries(keys.map(k=>[k,true]))},...Object.fromEntries(keys.map(k=>[k,fields[k]]))}}
const req=body=>new Request('https://fixture.example/api/ask',{method:'POST',body:JSON.stringify({question:'Explain this selected evidence',...body})});
const providerReply={answer:'Review the selected evidence cautiously [ES-LIMIT-001].',source_ids:['ES-LIMIT-001'],uncertainty:'Software fixture only.',recommended_action:'none'};

test('all 63 data selection combinations retain exact receipts and reject deselected fields',()=>{
 for(let bits=1;bits<64;bits++){
  const keys=C.DATA_CONTEXT_KEYS.filter((k,i)=>bits&(1<<i));const raw=context(keys);const parsed=C.normalizeObservationContext(raw);
  assert.deepEqual(C.contextReceipt(parsed).included,keys);assert.equal(C.contextReceipt(parsed).image_received,false);
  assert.equal(parsed.image,undefined);
 }
});
test('images rejected before provider call on both connection paths',async()=>{
 const saved=globalThis.fetch;let calls=0;globalThis.fetch=async()=>{calls++;throw Error('Must not call provider')};
 try{for(const env of [{GEMINI_API_KEY:'SYNTHETIC'},{ECOSIGNAL_CHATBOT_URL:'https://fixture.example/assistant/ask'}]){
  const raw=context(['coverage']);raw.selection.image=true;raw.image=fields.image;
  assert.equal((await handleAsk(req({observation_context:raw}),env)).status,400);
 }assert.equal(calls,0)}finally{globalThis.fetch=saved}
});
test('available fields and select-all cannot include photographs',()=>{
 const r={id:'real-1',source:'Collected photo',coverage:42,image:'data:image/jpeg;base64,PRIVATE'};
 assert.equal(C.availableContext(r).image,undefined);assert.equal(C.sanitizeSelection({...C.emptySelection(),image:true},{image:'private'}).image,false);
 assert.throws(()=>C.buildObservationContext(r,{...C.emptySelection(),image:true}));
});
test('direct Gemini preserves selected data and conversation with no image parts',async()=>{
 const saved=globalThis.fetch;try{for(const keys of [...C.DATA_CONTEXT_KEYS.map(k=>[k]),C.DATA_CONTEXT_KEYS]){
 let sent;globalThis.fetch=async(url,options)=>{sent=JSON.parse(options.body);return Response.json({candidates:[{content:{parts:[{text:JSON.stringify(providerReply)}]}}]})};
 const r=await handleAsk(req({observation_context:context(keys),conversation_history:[{role:'user',content:'Earlier question'},{role:'assistant',content:'Earlier answer'}]}),{GEMINI_API_KEY:'SYNTHETIC'});
 assert.equal(r.status,200);assert.equal(sent.contents[0].parts[0].text,'Earlier question');
 const parts=sent.contents.at(-1).parts;assert.equal(parts.length,1);assert.equal(parts[0].inlineData,undefined);
 const payload=JSON.parse(parts[0].text);for(const k of C.DATA_CONTEXT_KEYS)assert.equal(k in payload.selected_observation,keys.includes(k));
 assert.doesNotMatch(JSON.stringify(sent),/PRIVATE_IMAGE_SENTINEL|inlineData|inline_data/);
 }}finally{globalThis.fetch=saved}
});
test('Render proxy preserves exact data receipt and both conversation roles',async()=>{
 const saved=globalThis.fetch;try{const raw=context(C.DATA_CONTEXT_KEYS);globalThis.fetch=async(url,options)=>{
 const sent=JSON.parse(options.body);assert.deepEqual(sent.observation_context,raw);assert.equal(sent.conversation_history.length,2);assert.equal(sent.observation_context.image,undefined);
 return Response.json({answer:'Selected evidence supplied.',mode:'grounded_llm',sources:[],scan_evidence:[],context_receipt:C.contextReceipt(raw)})};
 assert.equal((await handleAsk(req({observation_context:raw,conversation_history:[{role:'user',content:'Earlier question'},{role:'assistant',content:'Earlier answer'}]}),{ECOSIGNAL_CHATBOT_URL:'https://fixture.example/assistant/ask'})).status,200);
 }finally{globalThis.fetch=saved}
});
test('question-only and follow-up chat remain available',async()=>{
 const saved=globalThis.fetch;try{globalThis.fetch=async()=>Response.json({candidates:[{content:{parts:[{text:JSON.stringify(providerReply)}]}}]});
 assert.equal((await handleAsk(req({question:'Why are lichens useful?'}),{GEMINI_API_KEY:'SYNTHETIC'})).status,200);
 }finally{globalThis.fetch=saved}
});
test('unchecked values and invalid numbers are rejected',()=>{
 const raw=context(['coverage']);raw.location=fields.location;assert.throws(()=>C.normalizeObservationContext(raw));
 const bad=context(['coverage']);bad.coverage={...fields.coverage,percent:101};assert.throws(()=>C.normalizeObservationContext(bad));
});
