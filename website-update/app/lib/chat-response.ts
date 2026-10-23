export type ChatSource={source_id:string;title:string;url:string;supports?:string};
export type ChatReply={answer:string;mode:'grounded_llm'|'deterministic_fallback';uncertainty:string;sources:ChatSource[];scan_evidence:{label:string;value:string;source:string}[];recommended_action?:string};
export function chatModeLabel(mode:ChatReply['mode']){return mode==='grounded_llm'?'AI-assisted · Cited response':'Knowledge-based guidance'}
export function citationUrl(value:unknown):string|null{if(typeof value!=='string')return null;try{const u=new URL(value);return ['https:','http:'].includes(u.protocol)&&!u.username&&!u.password?u.href:null}catch{return null}}
// Return only presentation fields. Provider flags, model names, request IDs,
// exception details and other diagnostics never enter the website response.
export function normalizeChatReply(raw:unknown):ChatReply{
 if(!raw||typeof raw!=='object')throw Error('Invalid response');const r=raw as Record<string,unknown>;
 if(typeof r.answer!=='string'||!r.answer.trim()||r.answer.length>12000||!['grounded_llm','deterministic_fallback'].includes(String(r.mode)))throw Error('Invalid response');
 if(r.status==='retryable_error'||(Array.isArray(r.safety_flags)&&r.safety_flags.some(f=>['LLM_GENERATION_UNAVAILABLE','SEMANTIC_INTERPRETATION_UNAVAILABLE','GEMINI_API_UNAVAILABLE'].includes(String(f)))))throw Error('Generation unavailable');
 const sources=Array.isArray(r.sources)?r.sources.filter(s=>s&&typeof s.source_id==='string'&&typeof s.title==='string'&&typeof s.url==='string').slice(0,30).map(s=>({source_id:s.source_id,title:s.title,url:citationUrl(s.url)||'',...(typeof s.supports==='string'?{supports:s.supports}:{})})):[];
 const scan_evidence=Array.isArray(r.scan_evidence)?r.scan_evidence.filter(e=>e&&typeof e.label==='string'&&typeof e.value==='string'&&typeof e.source==='string').slice(0,20).map(e=>({label:e.label,value:e.value,source:e.source})):[];
 return {answer:r.answer,mode:r.mode as ChatReply['mode'],uncertainty:typeof r.uncertainty==='string'?r.uncertainty:'',sources,scan_evidence,...(typeof r.recommended_action==='string'?{recommended_action:r.recommended_action}:{})};
}
