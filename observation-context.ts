import jpeg from 'jpeg-js';
import encodeJpeg from './vendor/jpeg-encoder';
export const CONTEXT_KEYS=['image','coverage','growth_form','satellite','geomorphology','elevation','location'] as const;
export const DATA_CONTEXT_KEYS=CONTEXT_KEYS.filter(k=>k!=='image');
export type ContextKey=typeof CONTEXT_KEYS[number];
export type Selection=Record<ContextKey,boolean>;
export const emptySelection=():Selection=>Object.fromEntries(CONTEXT_KEYS.map(k=>[k,false])) as Selection;
export type Point={lat:number;lon:number};
export type Metric={source:string;observed_at?:string;version?:string;[key:string]:unknown};
export type Context={observation_id:string;selection:Selection;image?:{mime_type:'image/jpeg';data:string};coverage?:Metric;growth_form?:Metric;satellite?:Metric;geomorphology?:Metric;elevation?:Metric;location?:Point};
export type ObservationRecord={id?:unknown;site?:unknown;date?:unknown;image?:unknown;coverage?:unknown;form?:unknown;source?:unknown;coverage_confidence?:unknown;growth_form_confidence?:unknown;latitude?:unknown;longitude?:unknown;captured_at?:unknown;raw_model_output?:any;screening?:any};
export type EnvironmentSnapshot={observation_key:string;location:Point;satellite?:any;georisk?:any;elevation?:any};
const object=(v:unknown):v is Record<string,any>=>!!v&&typeof v==='object'&&!Array.isArray(v);
const number=(v:unknown,min:number,max:number):v is number=>typeof v==='number'&&Number.isFinite(v)&&v>=min&&v<=max;
const text=(v:unknown,max:number)=>typeof v==='string'&&v.trim().length>0&&v.length<=max;
const only=(v:Record<string,unknown>,keys:string[])=>{if(Object.keys(v).some(k=>!keys.includes(k)))throw Error('Unexpected observation field.');};
export function validPoint(v:unknown):v is Point{return object(v)&&number(v.lat,-90,90)&&number(v.lon,-180,180)}
export function samePoint(a:unknown,b:unknown){return validPoint(a)&&validPoint(b)&&Math.abs(a.lat-b.lat)<.000001&&Math.abs(a.lon-b.lon)<.000001}
export function observationPoint(record:ObservationRecord):Point|undefined{
 const raw=record.screening?.location??{lat:record.latitude,lon:record.longitude};
 return validPoint(raw)?{lat:raw.lat,lon:raw.lon}:undefined;
}
export function observationKey(record?:ObservationRecord){const p=record&&observationPoint(record);return record?JSON.stringify([record.id,p,record.captured_at,record.date,record.image,record.coverage,record.form,record.source,record.coverage_confidence,record.growth_form_confidence,record.raw_model_output?.coverage?.version,record.raw_model_output?.growth_form?.version,record.screening]):''}
export function isCollected(record:ObservationRecord){return text(record.id,128)&&text(record.source,500)&&!/(illustrative|example|demo|synthetic|software fixture)/i.test(String(record.source))&&!/^demo/i.test(String(record.id))}
function provenance(raw:any,fallback:string):Metric{return {source:String(raw?.source||fallback).slice(0,160),...(text(raw?.observed_at,80)?{observed_at:raw.observed_at}:{}),...(text(raw?.version,120)?{version:raw.version}:{})}}
export function availableContext(record:ObservationRecord,snapshot?:EnvironmentSnapshot):Partial<Record<ContextKey,any>>{
 const result:Partial<Record<ContextKey,any>>={},location=observationPoint(record),screen=record.screening;
 if(location)result.location=location;
 if(number(record.coverage,0,100))result.coverage={...provenance({source:record.source,observed_at:record.captured_at,version:record.raw_model_output?.coverage?.version},'User-supplied coverage'),percent:record.coverage,...(number(record.coverage_confidence,0,1)?{confidence:record.coverage_confidence}:{})};
 if(text(record.form,80))result.growth_form={...provenance({source:record.source,observed_at:record.captured_at,version:record.raw_model_output?.growth_form?.version},'User-supplied growth-form result'),label:record.form,...(number(record.growth_form_confidence,0,1)?{confidence:record.growth_form_confidence}:{})};
 const current=snapshot&&snapshot.observation_key===observationKey(record)&&samePoint(snapshot.location,location)?snapshot:undefined;
 const satellite=current?.satellite??screen?.satellite;
 if(location&&satellite&&samePoint(satellite.location,location)){
  const metrics=Object.fromEntries(['ndvi','ndmi','bsi','mndwi'].filter(k=>number(satellite[k],-1,1)).map(k=>[k,satellite[k]]));
  if(Object.keys(metrics).length)result.satellite={...provenance(satellite,'Uploaded satellite dataset'),...metrics,
   ...(number(satellite.resolution_m,1,10000)?{resolution_m:satellite.resolution_m}:{}),
   ...(Number.isInteger(satellite.patch_pixels)&&number(satellite.patch_pixels,1,10000)?{patch_pixels:satellite.patch_pixels}:{}),
   acquisition_dates:Array.isArray(satellite.acquisition_dates)?satellite.acquisition_dates.filter((v:unknown)=>text(v,80)).slice(0,20):[]};
 }
 const geo=current?.georisk??screen?.georisk;
 // The numerical lattice location may differ from the requested observation;
 // only attach when the recorded requested point matches, never use map colours.
 if(location&&geo&&samePoint(geo.requested_location??geo.location,location)&&number(geo.raw,0,1))result.geomorphology={...provenance(geo,'User-supplied terrain result'),susceptibility:geo.raw,
  ...(number(geo.percentile,0,100)?{percentile:geo.percentile}:{}),...(Number.isInteger(geo.year)&&number(geo.year,1900,2100)?{reference_year:geo.year}:{}),
  ...(number(geo.sample_distance_m,0,100000)?{sample_distance_m:geo.sample_distance_m}:{}),...(number(geo.sample_spacing_m,1,100000)?{sample_spacing_m:geo.sample_spacing_m}:{})};
 const elevation=current?.elevation;
 if(location&&elevation&&samePoint(elevation.location,location)&&number(elevation.metres,-500,9000))result.elevation={...provenance(elevation,'Open-Meteo / Copernicus DEM GLO-90'),metres:elevation.metres,resolution_m:90};
 return result;
}

// Re-encoding in the browser strips metadata. Repeat stripping for JPEG inputs
// at the server boundary so crafted requests cannot forward EXIF or comments.
export function cleanJpeg(data:string):string{
 if(!/^[A-Za-z0-9+/]*={0,2}$/.test(data)||data.length%4||data.length>940000)throw Error('Invalid image.');
 const raw=Uint8Array.from(atob(data),c=>c.charCodeAt(0));
 if(raw.length>700000||raw[0]!==255||raw[1]!==216)throw Error('Invalid JPEG image.');
 try{
  const decoded=jpeg.decode(raw,{useTArray:true,formatAsRGBA:true,tolerantDecoding:false,maxResolutionInMP:2,maxMemoryUsageInMB:48});
  if(!decoded.width||!decoded.height||decoded.width>1280||decoded.height>1280)throw Error('dimensions');
  // Construct a new pixel-only object: decoder EXIF/comments never reach encoder.
  const clean=encodeJpeg({width:decoded.width,height:decoded.height,data:decoded.data},75).data;
  if(clean.length>700000)throw Error('size');
  let encoded='';for(let p=0;p<clean.length;p+=8192)encoded+=String.fromCharCode(...clean.subarray(p,p+8192));return btoa(encoded);
 }catch{throw Error('Choose a valid prepared JPEG photograph under 700 KB.');}
}

export function normalizeObservationContext(raw:unknown):Context|null{
 if(raw==null)return null;
 if(object(raw)&&(raw.image!=null||raw.selection?.image===true))throw Error('Photographs are disabled for chat. Select structured observation data instead.');
 if(!object(raw)||!text(raw.observation_id,128)||!object(raw.selection))throw Error('Invalid observation context.');
 only(raw,['observation_id','selection',...CONTEXT_KEYS]);only(raw.selection,[...CONTEXT_KEYS]);
 if(CONTEXT_KEYS.some(k=>typeof raw.selection[k]!=='boolean')||!CONTEXT_KEYS.some(k=>raw.selection[k]))throw Error('Select at least one item.');
 const out:Context={observation_id:raw.observation_id.trim(),selection:{...raw.selection} as Selection};
 for(const key of CONTEXT_KEYS){
  const value=raw[key],enabled=raw.selection[key];if(enabled!==(value!=null))throw Error('Selection does not match attached data.');
  if(!enabled)continue;
  if(!object(value))throw Error('Invalid selected data.');
  if(key==='image'){only(value,['mime_type','data']);if(value.mime_type!=='image/jpeg'||typeof value.data!=='string')throw Error('Choose a prepared JPEG photograph.');out.image={mime_type:'image/jpeg',data:cleanJpeg(value.data)};continue;}
  if(key==='location'){only(value,['lat','lon']);if(!validPoint(value))throw Error('Invalid coordinates.');out.location={lat:value.lat,lon:value.lon};continue;}
  const bounds:Record<string,[number,number]>={percent:[0,100],confidence:[0,1],ndvi:[-1,1],ndmi:[-1,1],bsi:[-1,1],mndwi:[-1,1],resolution_m:[.000001,10000],patch_pixels:[1,10000],susceptibility:[0,1],percentile:[0,100],reference_year:[1900,2100],sample_distance_m:[0,100000],sample_spacing_m:[.000001,100000],metres:[-500,9000]};
  const keys:Record<string,string[]>={coverage:['percent','confidence'],growth_form:['label','confidence'],satellite:['ndvi','ndmi','bsi','mndwi','resolution_m','patch_pixels','acquisition_dates'],geomorphology:['susceptibility','percentile','reference_year','sample_distance_m','sample_spacing_m'],elevation:['metres','resolution_m']};
  only(value,['source','observed_at','version',...keys[key]]);if(!text(value.source,160))throw Error('A source is required.');
  for(const field of ['observed_at','version'])if(value[field]!=null&&!text(value[field],field==='version'?120:80))throw Error('Invalid provenance.');
  for(const [field,v]of Object.entries(value)){if(v==null)continue;if(bounds[field]&&!number(v,...bounds[field]))throw Error('Invalid measurement.');if(['patch_pixels','reference_year'].includes(field)&&!Number.isInteger(v))throw Error('Invalid measurement.');}
  if(key==='growth_form'&&!text(value.label,80))throw Error('Invalid growth form.');
  if(key==='satellite'){
   if(!['ndvi','ndmi','bsi','mndwi'].some(k=>number(value[k],-1,1)))throw Error('No satellite values.');
   if(value.acquisition_dates!=null&&(!Array.isArray(value.acquisition_dates)||value.acquisition_dates.length>20||value.acquisition_dates.some((v:unknown)=>!text(v,80))))throw Error('Invalid acquisition dates.');
  }
  const required:Record<string,string>={coverage:'percent',geomorphology:'susceptibility',elevation:'metres'};
  if(required[key]&&!number(value[required[key]],...bounds[required[key]]))throw Error('Missing selected measurement.');
  (out as any)[key]={...value};
 }
 return out;
}
export function buildObservationContext(record:ObservationRecord,selection:Selection,snapshot?:EnvironmentSnapshot,image?:Context['image']):Context|null{
 if(selection.image||image)throw Error('Photographs are disabled for chat.');
 if(!CONTEXT_KEYS.some(k=>selection[k]))return null;if(!isCollected(record))throw Error('Choose one of your collected observations.');
 const available=availableContext(record,snapshot),out:any={observation_id:record.id,selection:{...selection}};
 for(const key of CONTEXT_KEYS)if(selection[key]){if(key==='image'){if(!available.image||!image)throw Error('Prepare the selected photograph first.');out.image=image;}else{if(!available[key])throw Error('Selected data is unavailable for this observation.');out[key]=available[key];}}
 return normalizeObservationContext(out);
}
export function contextEvidence(context:Context|null){
 if(!context)return [];const labels:Record<ContextKey,string>={image:'Attached photograph',coverage:'Visible coverage',growth_form:'Growth form',satellite:'Satellite indices',geomorphology:'Geomorphology susceptibility',elevation:'Elevation',location:'Selected coordinates'};
 return CONTEXT_KEYS.filter(k=>context.selection[k]).map(k=>({label:labels[k],value:k==='image'?'Photograph supplied for visual description only':JSON.stringify(context[k]),source:`User-selected observation ${context.observation_id}; submitted, not independently verified`}));
}
export function contextText(context:Context|null){if(!context)return null;const {image,...text}=context;return {...text,...(image?{image_attached:true}:{}),provenance_status:'User-selected submitted observation; not independently verified'}}

export type ContextReceipt={version:'observation-context-v1';observation_id:string;included:ContextKey[];image_received:boolean};
export function contextReceipt(context:Context):ContextReceipt{return {version:'observation-context-v1',observation_id:context.observation_id,included:CONTEXT_KEYS.filter(k=>context.selection[k]),image_received:!!context.image}}
export function requireContextReceipt(reply:unknown,context:Context|null){
 if(!context)return true;
 const value=object(reply)?reply.context_receipt:null;if(!object(value))return false;
 const expected=contextReceipt(context);
 return Object.keys(value).length===4&&value.version===expected.version&&value.observation_id===expected.observation_id&&value.image_received===expected.image_received&&Array.isArray(value.included)&&value.included.length===expected.included.length&&value.included.every((v:unknown,i:number)=>v===expected.included[i]);
}
export function attachmentSignature(record:ObservationRecord,selection:Selection,snapshot?:EnvironmentSnapshot){
 const values=availableContext(record,snapshot);
 return JSON.stringify([observationKey(record),CONTEXT_KEYS.filter(k=>selection[k]).map(k=>[k,values[k]??null])]);
}

export function sanitizeSelection(chosen:Selection,available:Partial<Record<ContextKey,unknown>>):Selection{return Object.fromEntries(CONTEXT_KEYS.map(k=>[k,k!=='image'&&chosen[k]&&available[k]!=null])) as Selection}
