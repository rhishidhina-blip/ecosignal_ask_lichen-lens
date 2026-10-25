"use client";
import {useEffect,useRef,useState} from 'react';
import {chatModeLabel,citationUrl,normalizeChatReply,type ChatReply} from './lib/chat-response';
import {BROWSER_TIMEOUT_MS,conversationHistory,thinkingLabel,createChatId} from './lib/chat-request';

import ObservationAttachments,{selectedAttachmentLabels,type AttachmentChoice} from './observation-attachments';
import {buildObservationContext,attachmentSignature,type ObservationRecord} from './lib/observation-context';
import './assistant-context.css';
type Turn={id:string;question:string;reply?:ChatReply;error?:string;attachments?:string[]};
function CitedAnswer({reply}:{reply:ChatReply}){
 return <p className="chat-answer">{reply.answer.split(/(\[[A-Za-z0-9_-]+\])/g).map((part,i)=>{const source=reply.sources.find(s=>`[${s.source_id}]`===part),url=source&&citationUrl(source.url);return url?<a key={i} href={url} target="_blank" rel="noreferrer" title={source?.title}>{part}</a>:part})}</p>;
}
export default function AssistantPanel({sample,observations,onClose,onAction}:{sample:Record<string,unknown>|null;observations?:ObservationRecord[];onClose:()=>void;onAction:(action:string)=>void}){
 const [question,setQuestion]=useState(''),[turns,setTurns]=useState<Turn[]>([]),[busy,setBusy]=useState(false),[slow,setSlow]=useState(false),[connection,setConnection]=useState('Ask about lichens and your environment');
 const attachmentChoice=useRef<AttachmentChoice>(null),contextSignature=useRef('');
 function changeAttachments(choice:AttachmentChoice){
  const signature=choice?attachmentSignature(choice.record,choice.selection,choice.snapshot):'';
  attachmentChoice.current=choice;if(signature!==contextSignature.current){contextSignature.current=signature;requestRef.current?.abort();setTurns([]);conversationId.current=null;setConnection('Choose the evidence Gemini receives');}
 }
 const conversationId=useRef<string|null>(null),requestRef=useRef<AbortController|null>(null),mounted=useRef(true),end=useRef<HTMLDivElement>(null),input=useRef<HTMLInputElement>(null);
 useEffect(()=>{mounted.current=true;return()=>{mounted.current=false;requestRef.current?.abort()}},[]);
 useEffect(()=>{input.current?.focus();const close=(e:KeyboardEvent)=>{if(e.key==='Escape')onClose();if(e.key==='Tab'){const nodes=Array.from(document.querySelectorAll<HTMLElement>('.conversational button:not(:disabled), .conversational input:not(:disabled), .conversational select:not(:disabled), .conversational a, .conversational summary'));const first=nodes[0],last=nodes.at(-1);if(e.shiftKey&&document.activeElement===first){e.preventDefault();last?.focus()}else if(!e.shiftKey&&document.activeElement===last){e.preventDefault();first?.focus()}}};document.addEventListener('keydown',close);return()=>document.removeEventListener('keydown',close)},[onClose]);
 useEffect(()=>{end.current?.scrollIntoView({behavior:window.matchMedia('(prefers-reduced-motion: reduce)').matches?'auto':'smooth'})},[turns,busy,slow]);
 async function ask(q:string,retryId?:string){
  if(requestRef.current||!q.trim())return;
  const turnId=retryId||createChatId(),controller=new AbortController();requestRef.current=controller;
  setTurns(t=>retryId?t.map(turn=>turn.id===turnId?{id:turn.id,question:turn.question}:turn):[...t,{id:turnId,question:q}]);setQuestion('');setBusy(true);setSlow(false);
  const slowTimer=setTimeout(()=>{if(mounted.current)setSlow(true)},15000),timeout=setTimeout(()=>controller.abort(),BROWSER_TIMEOUT_MS);
  try{
   const choice=attachmentChoice.current;
   const selected=choice?selectedAttachmentLabels(choice.selection):[];
   const observation_context=choice?buildObservationContext(choice.record,choice.selection,choice.snapshot):null;
   setTurns(t=>t.map(turn=>turn.id===turnId?{...turn,attachments:selected}:turn));
   conversationId.current??=createChatId();
   const response=await fetch('/api/ask',{method:'POST',headers:{'content-type':'application/json'},signal:controller.signal,body:JSON.stringify({question:q,conversation_id:conversationId.current,conversation_history:conversationHistory(turns,retryId),observation_context})});
   if(!response.ok){const error:any=await response.json().catch(()=>null);throw Error(response.status===504?'timeout':error?.error||'Ask LichenLens could not respond. Please try again.');}
   const reply=normalizeChatReply(await response.json());
   if(mounted.current){setConnection(chatModeLabel(reply.mode));setTurns(t=>t.map(turn=>turn.id===turnId?{...turn,reply,error:undefined}:turn))}
  }catch(e){
   if(mounted.current){const timeoutError=controller.signal.aborted||(e instanceof Error&&e.message==='timeout');setTurns(t=>t.map(turn=>turn.id===turnId?{...turn,error:timeoutError?'The response is taking longer than expected. Please try again.':(e instanceof Error&&/^(Choose |Select |Selected |Prepare |Check |The chatbot backend |Image preparation |This photograph |Ask LichenLens )/.test(e.message)?e.message:'Ask LichenLens could not respond right now. Please try again.')}:turn))}
  }finally{clearTimeout(slowTimer);clearTimeout(timeout);requestRef.current=null;if(mounted.current){setBusy(false);setSlow(false)}}
 }
 return <div className="chat-backdrop" onClick={onClose}><section className="chat-panel conversational" role="dialog" aria-modal="true" aria-label="Ask LichenLens" onClick={e=>e.stopPropagation()}>
  <div className="chat-heading"><span>✳ Ask LichenLens</span><button onClick={onClose} aria-label="Close assistant">×</button></div>
  <div className="chat-context"><span className="badge neutral">Choose observation fields below</span><small>{connection}</small></div>
  <ObservationAttachments observations={observations??(sample?[sample]:[])} initialId={typeof sample?.id==='string'?sample.id:undefined} disabled={busy} onChange={changeAttachments}/>
  <div className="chat-thread" aria-live="polite">
   {!turns.length&&<><h2>Ask. Explore. Act.</h2><div className="suggestions">{['Explain my scan','How reliable is this result?','Where should I sample next?','What does satellite change show?'].map(q=><button key={q} onClick={()=>ask(q)} disabled={busy}>{q}<span>↗</span></button>)}</div></>}
   {turns.map(t=><div className="chat-turn" key={t.id}><p className="user-message">{t.question}</p>{!!t.attachments?.length&&<p className="chat-attachments">Selected for this question: {t.attachments.join(" · ")}</p>}
    {t.reply&&<div className="assistant-message"><small className="chat-response-label">{chatModeLabel(t.reply.mode)}</small><CitedAnswer reply={t.reply}/>
     {!!t.reply.scan_evidence.length&&<div className="chat-evidence">{t.reply.scan_evidence.map((e,i)=><span key={i} title={e.source}>{e.label}<b>{e.value}</b></span>)}</div>}
     {!!t.reply.sources.length&&<div className="chat-citations"><small>Sources</small><div className="chat-sources">{t.reply.sources.map((s,i)=>{const url=citationUrl(s.url);return url?<a key={i} href={url} target="_blank" rel="noreferrer" title={s.supports}>{s.title} ↗</a>:<span key={i}>{s.title}</span>})}</div></div>}
     {t.reply.uncertainty&&<div className="chat-limitations"><small>Scientific limitations</small><p>{t.reply.uncertainty}</p></div>}
     {t.reply.recommended_action&&t.reply.recommended_action!=='none'&&<button className="text-button" onClick={()=>{if(t.reply?.recommended_action)onAction(t.reply.recommended_action)}}>{({review_scan:'Review observation',repeat_scan:'Open repeat scans',claim_field_mission:'Open field planner',check_airnow_or_aqs:'Open AirNow',retake_photo:'Retake photo',seek_expert_review:'Seek expert review'})[t.reply.recommended_action]||'Explore next step'} →</button>}
    </div>}
    {t.error&&<div className="chat-request-error" role="alert"><p>{t.error}</p><button className="text-button" disabled={busy} onClick={()=>ask(t.question,t.id)}>Try again ↗</button></div>}
   </div>)}
   {busy&&<div className="chat-loading" role="status"><span className="chat-loading-dots" aria-hidden="true"><i/><i/><i/></span><span>{thinkingLabel(slow)}</span></div>}<div ref={end}/>
  </div>
  <form className="chat-compose" onSubmit={e=>{e.preventDefault();ask(question)}}><label htmlFor="chat-question" className="sr-only">Ask a question</label><input id="chat-question" ref={input} value={question} onChange={e=>setQuestion(e.target.value)} placeholder="Ask about your observation…" maxLength={1200}/><button type="submit" disabled={busy||!question.trim()} aria-label="Send question">↑</button></form>
 </section></div>
}
