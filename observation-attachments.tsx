'use client';
import {useEffect,useRef,useState} from 'react';
import {availableContext,sanitizeSelection,DATA_CONTEXT_KEYS,emptySelection,isCollected,observationKey,observationPoint,type ContextKey,type EnvironmentSnapshot,type ObservationRecord,type Selection} from './lib/observation-context';
import {loadElevation} from './lib/prepare-observation';
import {sampleNative} from './lib/unified-satellite';
import {lookupGeoRisk} from './lib/georisk-numeric';
export type AttachmentChoice={record:ObservationRecord;selection:Selection;snapshot?:EnvironmentSnapshot}|null;
const labels:Record<ContextKey,string>={image:'Photograph',coverage:'Lichen coverage',growth_form:'Growth form',satellite:'Satellite indices',geomorphology:'Terrain susceptibility',elevation:'Elevation',location:'Latitude / longitude'};
export const selectedAttachmentLabels=(selection:Selection)=>DATA_CONTEXT_KEYS.filter(k=>selection[k]).map(k=>labels[k]);

export default function ObservationAttachments({observations,initialId,disabled,onChange}:{observations:ObservationRecord[];initialId?:string;disabled:boolean;onChange:(choice:AttachmentChoice)=>void}){
 const collected=observations.filter(isCollected),[id,setId]=useState(()=>collected.some(r=>r.id===initialId)?initialId!:'');
 const record=collected.find(r=>r.id===id);
 return <div className="attachment-container"><label className="attachment-observation-label">Collected observation<select value={record?id:''} disabled={disabled} onChange={e=>{onChange(null);setId(e.target.value)}}><option value="">No observation · question only</option>{collected.map(r=><option key={String(r.id)} value={String(r.id)}>{String(r.site||r.id)} · {typeof r.date==='string'?r.date.slice(0,10):'date unavailable'} · {String(r.id)}</option>)}</select></label>
 {!collected.length&&<p className="attachment-help">Save an observation in LichenLens first. Examples are excluded.</p>}
 <AttachmentFields key={observationKey(record)} record={record} disabled={disabled} onChange={onChange}/></div>;
}
function AttachmentFields({record,disabled,onChange}:{record?:ObservationRecord;disabled:boolean;onChange:(choice:AttachmentChoice)=>void}){
 const [chosen,setSelection]=useState(emptySelection),[snapshot,setSnapshot]=useState<EnvironmentSnapshot>(),[loading,setLoading]=useState(false),[elevationLoading,setElevationLoading]=useState(false),[notice,setNotice]=useState('');
 const key=observationKey(record),point=record&&observationPoint(record),available=record?availableContext(record,snapshot):{};
 const selection=sanitizeSelection(chosen,available);
 const active=useRef(true),change=useRef(onChange),lookups=useRef<AbortController[]>([]),sequence=useRef(0);change.current=onChange;
 useEffect(()=>{active.current=true;return()=>{active.current=false;sequence.current++;for(const c of lookups.current)c.abort()}},[]);
 const selectedKey=JSON.stringify(selection);
 useEffect(()=>{if(DATA_CONTEXT_KEYS.some(k=>chosen[k]&&!selection[k])){setSelection(selection);setNotice('Unavailable selected fields were cleared. Choose fields again after loading valid data.')}},[selectedKey,chosen]);
 useEffect(()=>{change.current(record?{record,selection,snapshot}:null)},[record,key,selectedKey,snapshot]);
 async function loadEnvironment(){
  if(!point||disabled)return;const requestSequence=++sequence.current;setLoading(true);setNotice('Loading values at this observation’s coordinates…');
  const results=await Promise.allSettled([sampleNative(point),lookupGeoRisk(point)]);
  if(!active.current||requestSequence!==sequence.current)return;
  const satellite=results[0].status==='fulfilled'?results[0].value:null,georisk=results[1].status==='fulfilled'?results[1].value:null;
  setSnapshot(previous=>({...previous,observation_key:key,location:point,satellite:satellite??undefined,georisk:georisk??undefined}));
  setNotice([satellite?'Satellite values loaded.':'No usable satellite values here.',georisk?'Numerical terrain sample loaded.':'No numerical terrain sample here.','Choose which fields to send below.'].join(' '));setLoading(false);
 }
 async function getElevation(){
  if(!point||disabled)return;const controller=new AbortController();lookups.current.push(controller);setElevationLoading(true);
  try{const elevation=await loadElevation(point,controller.signal);if(active.current){setSnapshot(previous=>({...previous,observation_key:key,location:point,elevation}));setNotice('Elevation loaded from Open-Meteo / Copernicus GLO-90 (90 m DEM, 2021 release). Select Elevation to include it.')}}
  catch{if(active.current)setNotice('Elevation could not load. No replacement value will be sent.')}
  finally{if(active.current)setElevationLoading(false)}
 }
 function summary(k:ContextKey){const v=available[k];if(!v)return k==='satellite'||k==='geomorphology'||k==='elevation'?'Load a sample at this observation’s coordinates':'Not available in this observation';
  if(k==='image')return 'Resized photo · embedded metadata removed';if(k==='coverage')return `${v.percent}% · ${v.source}`;if(k==='growth_form')return `${v.label} · ${v.source}`;
  if(k==='location')return `${v.lat}, ${v.lon}`;if(k==='satellite')return ['ndvi','ndmi','bsi','mndwi'].filter(n=>v[n]!=null).map(n=>`${n.toUpperCase()} ${v[n]}`).join(' · ')+` · ${v.observed_at||'date unavailable'}`;
  if(k==='geomorphology')return `${v.susceptibility} susceptibility${v.percentile!=null?` · percentile ${v.percentile}`:''} · ${v.reference_year||'year unavailable'}`;
  return `${v.metres} m · 90 m DEM`;
 }
 return <details className="observation-attachments"><summary>Choose what to send <span>{selectedAttachmentLabels(selection).length} selected</span></summary>
  {record&&<>
   <p className="attachment-help">Only checked data fields go to Gemini with your next question. Photographs are not sent. Changing shared data starts a new chat so earlier values stay out.</p>
   <div className="attachment-fields">{DATA_CONTEXT_KEYS.map(k=><label key={k} className={!available[k]?'attachment-unavailable':''}><input type="checkbox" checked={selection[k]&&!!available[k]} disabled={disabled||!available[k]} onChange={e=>setSelection(s=>({...s,[k]:e.target.checked}))}/><span><b>{labels[k]}</b><small>{summary(k)}</small></span></label>)}</div>
   <div className="attachment-actions"><button type="button" disabled={disabled} onClick={()=>setSelection(Object.fromEntries(DATA_CONTEXT_KEYS.map(k=>[k,!!available[k]])) as Selection)}>Select available</button><button type="button" disabled={disabled} onClick={()=>setSelection(emptySelection())}>Clear</button></div>
   {point?<div className="attachment-lookups"><button type="button" disabled={disabled||loading} onClick={loadEnvironment}>{loading?'Loading environmental samples…':'Load satellite + terrain at these coordinates'}</button><button type="button" disabled={disabled||elevationLoading} onClick={getElevation}>{elevationLoading?'Loading elevation…':'Load elevation · shares coordinates with Open-Meteo'}</button></div>:<p className="attachment-help">Add this observation’s latitude and longitude to load environmental samples.</p>}
   <p className="attachment-disclosure">Satellite and terrain use the uploaded datasets and retain their dates and resolution. Coordinates reach Gemini only if checked. Coverage and growth form are supplied model results, not visual analysis by the chatbot.</p>
   {notice&&<p role="status" className="attachment-notice">{notice}</p>}
   <div className="attachment-preview" aria-live="polite"><b>With your next question</b><p>{selectedAttachmentLabels(selection).join(' · ')||'Question only — no observation fields selected'}</p></div>
  </>}
 </details>;
}
