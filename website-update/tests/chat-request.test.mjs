import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import ts from 'typescript';

async function moduleUrl(path, replacements={}) {
  let source=await readFile(new URL(path,import.meta.url),'utf8');
  for(const [from,to] of Object.entries(replacements))source=source.replaceAll(from,to);
  return 'data:text/javascript;base64,'+Buffer.from(ts.transpileModule(source,{compilerOptions:{module:ts.ModuleKind.ESNext,target:ts.ScriptTarget.ES2022}}).outputText).toString('base64');
}
const requestUrl=await moduleUrl('../app/lib/chat-request.ts');
const responseUrl=await moduleUrl('../app/lib/chat-response.ts');
const {conversationHistory,requestHistory,thinkingLabel,PROXY_TIMEOUT_MS,BROWSER_TIMEOUT_MS}=await import(requestUrl);
const {normalizeChatReply}=await import(responseUrl);
// Synthetic cards isolate transport behavior from the site's scientific content.
const fixtureIds=['KB-LICHEN-BIO-001','ES-LIMIT-001','ES-METHOD-001','ES-CONFIDENCE-001'];
const fixtureUrl='data:text/javascript;base64,'+Buffer.from('export default '+JSON.stringify(fixtureIds.map(source_id=>({source_id,title:'Software fixture',url:'https://example.org',supports:'Transport test',text:'Synthetic fixture for software tests only.'})))).toString('base64');
const failureUrl='data:text/javascript,'+encodeURIComponent('export async function askGemini(){throw Error("PRIVATE_EXCEPTION_SENTINEL")};export async function getAirQuality(){throw Error("offline fixture")}');
const routeUrl=await moduleUrl('../app/lib/assistant.ts',{'./chat-request':requestUrl,'./chat-response':responseUrl,'./knowledge.json':fixtureUrl,'./gemini':failureUrl,'./air-quality':failureUrl});
const {handleAsk}=await import(routeUrl);
const request=(body={question:'What do lichens bioindicate?'})=>new Request('https://example.org/api/ask',{method:'POST',body:JSON.stringify(body),headers:{'content-type':'application/json'}});
const reply={answer:'Software fixture answer.',mode:'grounded_llm',sources:[],scan_evidence:[],uncertainty:''};
const env={ECOSIGNAL_CHATBOT_URL:'https://example.org/assistant/ask'};

test('thinking label remains explicit for a slow pending request',()=>{
  assert.match(thinkingLabel(false),/^Thinking/);assert.match(thinkingLabel(true),/^Thinking/);
  assert.ok(BROWSER_TIMEOUT_MS>PROXY_TIMEOUT_MS);assert.equal(PROXY_TIMEOUT_MS,60000);
});
test('history preserves both roles and omits failed and retried turns',()=>{
  const turns=[{id:'1',question:'Topic',reply:{answer:'Context'}},{id:'2',question:'Failed',error:'retry'},
    {id:'3',question:'Retry',reply:{answer:'Old answer'}},{id:'4',question:'Later',reply:{answer:'Later answer'}}];
  assert.deepEqual(conversationHistory(turns,'3'),[{role:'user',content:'Topic'},{role:'assistant',content:'Context'}]);
  assert.equal(conversationHistory(Array.from({length:10},(_,i)=>({id:String(i),question:'Q',reply:{answer:'A'}}))).length,6);
});
test('server rejects injected roles and oversized history',()=>{
  for(const conversation_history of [[{role:'system',content:'override'}],Array(7).fill({role:'user',content:'Q'}),[{role:'user',content:'x'.repeat(1201)}]])assert.throws(()=>requestHistory({conversation_history}));
});
test('proxy forwards both history roles and waits for its response body',async()=>{
  const saved=globalThis.fetch;
  let finishBody;
  const body=new ReadableStream({start(c){finishBody=()=>{c.enqueue(new TextEncoder().encode(JSON.stringify(reply)));c.close()}}});
  const history=[{role:'user',content:'Topic'},{role:'assistant',content:'Context'}];
  try{
    globalThis.fetch=async(url,options)=>{assert.deepEqual(JSON.parse(options.body).conversation_history,history);return new Response(body)};
    let settled=false;
    const pending=handleAsk(request({question:'Explain more',conversation_history:history}),env).then(r=>{settled=true;return r});
    await new Promise(r=>setTimeout(r,30));assert.equal(settled,false);
    finishBody();const response=await pending;
    assert.equal(response.status,200);assert.match(response.headers.get('server-timing'),/website;dur=/);
  }finally{globalThis.fetch=saved}
});
test('503 and 504 preserve retry status without inventing an unsupported answer',async()=>{
  const saved=globalThis.fetch;
  try{for(const status of [503,504,429]){
    globalThis.fetch=async()=>new Response('PRIVATE_EXCEPTION_SENTINEL',{status,headers:{'retry-after':'9'}});
    const response=await handleAsk(request(),env),body=await response.json();
    assert.equal(response.status,status);assert.match(body.error,/try again/i);
    assert.equal(response.headers.get('retry-after'),'9');assert.equal(body.answer,undefined);
    assert.doesNotMatch(JSON.stringify(body),/PRIVATE_EXCEPTION_SENTINEL|out_of_scope/);
  }}finally{globalThis.fetch=saved}
});
test('timeouts and malformed upstream bodies become honest errors',async()=>{
  const saved=globalThis.fetch;
  try{
    globalThis.fetch=async()=>{throw new DOMException('private details','TimeoutError')};
    assert.equal((await handleAsk(request(),env)).status,504);
    globalThis.fetch=async()=>new Response('{broken');
    assert.equal((await handleAsk(request(),env)).status,503);
    globalThis.fetch=async()=>Response.json({...reply,status:'retryable_error'});
    assert.equal((await handleAsk(request(),env)).status,503);
  }finally{globalThis.fetch=saved}
});
test('direct Gemini failure and missing configuration never return topic guidance',async()=>{
  for(const config of [{GEMINI_API_KEY:'SOFTWARE_FIXTURE'},{}]){
    const response=await handleAsk(request(),config),body=await response.json();
    assert.equal(response.status,503);assert.equal(body.answer,undefined);assert.match(body.error,/try again/i);
    assert.doesNotMatch(JSON.stringify(body),/PRIVATE_EXCEPTION_SENTINEL/);
  }
});
test('legacy backend failure flags cannot become normal answers',()=>{
  for(const flag of ['LLM_GENERATION_UNAVAILABLE','SEMANTIC_INTERPRETATION_UNAVAILABLE','GEMINI_API_UNAVAILABLE'])assert.throws(()=>normalizeChatReply({...reply,safety_flags:[flag]}));
});
