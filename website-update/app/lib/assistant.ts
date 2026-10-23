import {PROXY_TIMEOUT_MS,RETRY_MESSAGE,requestHistory} from './chat-request';
import {normalizeChatReply} from './chat-response';
import knowledge from './knowledge.json';
import {askGemini,type ChatEnvironment} from './gemini';
import {getAirQuality} from './air-quality';
type Sample = {site?:string;coverage?:number;form?:string;confidence?:number;source?:string;status?:string};
export function groundedReply(question:string,sample?:Sample,history?:{question:string}[]) {
  let q=question.toLowerCase();
  if(/^(why|how so|what about that|explain more|tell me more)[?.! ]*$/.test(q)&&history?.length) q+=' '+history.at(-1)!.question.toLowerCase();
  let ids=['KB-LICHEN-BIO-001','ES-LIMIT-001'],intent='lichen_education',action='none';
  if(/ignore.*instruction|system prompt|api.?key|password|secret|jailbreak|bypass/.test(q)) return {intent:'security_redirect',answer:'I can help interpret lichen observations and fieldwork. I cannot reveal credentials or change the evidence rules.',sources:[],scan_evidence:[],uncertainty:'',recommended_action:'none',mode:'deterministic_fallback'};
  if(/satellite|sentinel|ndvi|ndmi|bsi|mndwi|moisture index|cloud/.test(q)){ids=['ES-SAT-RALEIGH-001','ES-LIMIT-001'];intent='satellite';}
  else if(/repeat|changed|change over time|before.*after/.test(q)){ids=['ES-REPEAT-001','ES-METHOD-001'];intent='repeat_scan';action='repeat_scan';}
  else if(/mission|next|where.*sample|collect|planner/.test(q)){ids=['ES-MISSION-001','ES-REPEAT-001'];intent='field_mission';action='claim_field_mission';}
  else if(/confidence|reliable|uncertain|trust|risk|pollut/.test(q)){ids=['ES-CONFIDENCE-001','KB-LICHEN-BIO-001'];intent='reliability';action='review_scan';}
  else if(/growth|foliose|fruticose|crustose|species/.test(q)){ids=['KB-GROWTH-001','ES-METHOD-001'];intent='growth_form';}
  else if(/coverage|mask|segment/.test(q)){ids=['ES-METHOD-001','ES-REPEAT-001'];intent='coverage';}
  else if(/aqi|airnow|aqs|air quality|health|medical/.test(q)){ids=['KB-AQI-001','KB-AQS-001'];intent='air_quality_data';action='check_airnow_or_aqs';}
  else if(/limitation|cannot|can't|chlorosis|cause|diagnos/.test(q)){ids=['ES-LIMIT-001','KB-LICHEN-BIO-001'];intent='limitations';}
  else if(/scan|photo|image|result/.test(q)){ids=['ES-METHOD-001','ES-CONFIDENCE-001'];intent='scan_explanation';action='review_scan';}
  else if(!/lichen|bioindicator|ecosignal|hello|hi\b/.test(q)) return {intent:'out_of_scope',answer:'Ask me about lichen coverage, growth form, confidence, air data, satellite context, or your next field observation.',sources:[],scan_evidence:[],uncertainty:'',recommended_action:'none',mode:'deterministic_fallback'};
  const sources=ids.map(id=>knowledge.find(k=>k.source_id===id)!);
  const evidence:{label:string;value:string;source:string}[]=[];
  const source=String(sample?.source||'User-supplied context').slice(0,100);
  if(sample?.site)evidence.push({label:'Site',value:String(sample.site).slice(0,120),source});
  if(typeof sample?.coverage==='number'&&Number.isFinite(sample.coverage)&&sample.coverage>=0&&sample.coverage<=100)evidence.push({label:'Coverage',value:sample.coverage+'%',source});
  if(sample?.form)evidence.push({label:'Growth form',value:String(sample.form).slice(0,80),source});
  if(typeof sample?.confidence==='number'&&Number.isFinite(sample.confidence)&&sample.confidence>=0&&sample.confidence<=100)evidence.push({label:'Model confidence',value:sample.confidence+'%',source});
  const explanations:Record<string,string>={
    coverage:'Coverage is the visible fraction of the analyzed surface labeled as lichen. Compare the same surface and framing; the mask can include prediction errors. It does not measure pollutant concentration.',
    satellite:'The uploaded Raleigh Sentinel-2 dataset is connected: NDVI (vegetation), NDMI (vegetation moisture), BSI (bare soil), and MNDWI (water). Switch Sep 21–27, Sep 28–Oct 4, or Change, then click the map for a pixel value. Change uses common clear pixels; transparent areas have no usable data. These landscape indices do not measure lichen health or prove causes.',
    reliability:'Check each component separately: image quality, coverage, growth-form confidence, location completeness, and environmental inputs. Missing evidence stays missing. A model confidence value is not the probability that pollution caused the observation.',
    repeat_scan:'Confirm the same tree or surface, then match distance, framing, angle, lighting, and moisture. Check registration and comparability before interpreting a coverage difference. A visible change does not establish its cause.',
    field_mission:'Choose an accessible site that fills a geographic gap, resolves an uncertain observation, or supports a repeat scan. Record the site, time, substrate, and coordinates. Planner priority is a sampling heuristic, not a health-risk estimate.',
    air_quality_data:'Use AirNow for current AQI and EPA AQS for ambient monitoring records. No AQI reading is attached to this conversation. The Landscape panel can fetch AirNow reported observations for the selected map location. A lichen photo complements monitoring data; it cannot replace an air-quality measurement or health guidance.',
    limitations:'EcoSignal reports visible coverage and broad growth forms. It cannot identify pollutants from color, diagnose chlorosis, prove pollution causation, or provide regulatory-grade air monitoring.',
    scan_explanation:'Review the photo, coverage mask, growth form, and their source together. New uploaded photos remain pending until actual model results are imported. Missing outputs are not predictions.'
  };
  let answer=explanations[intent]||sources[0].text;
  if(evidence.length&&/coverage|growth_form|scan_explanation|reliability/.test(intent)) answer+=' Attached values are '+(/Illustrative/.test(source)?'illustrative example data.':'user-supplied records awaiting independent verification.');
  return {intent,answer,sources:sources.map(({text,...s})=>s),scan_evidence:evidence,uncertainty:evidence.length?'Framing, lighting, moisture, and verification affect interpretation.':'No analyzed observation is attached.',recommended_action:action,mode:'deterministic_fallback'};
}

async function handleAskRequest(request:Request,env:ChatEnvironment) {
  const headers={'content-type':'application/json','cache-control':'no-store'};
  if(request.method==='GET')return new Response(JSON.stringify({configured:!!(env.GEMINI_API_KEY||env.ECOSIGNAL_CHATBOT_URL),provider:env.ECOSIGNAL_CHATBOT_URL?'endpoint':env.GEMINI_API_KEY?'gemini':'guidance'}),{headers});
  if(request.method!=='POST')return new Response(JSON.stringify({error:'Use POST'}),{status:405,headers});
  if(Number(request.headers.get('content-length')||0)>16000)return new Response(JSON.stringify({error:'Request too large'}),{status:413,headers});
  try {
    const raw=await request.text();if(raw.length>16000)return new Response(JSON.stringify({error:'Request too large'}),{status:413,headers});
    const body=JSON.parse(raw);const question=body.question;
    if(typeof question!=='string'||!question.trim()||question.length>1200)return new Response(JSON.stringify({error:'Enter a question under 1,200 characters.'}),{status:400,headers});
    let conversation_history: ReturnType<typeof requestHistory>;
    try { conversation_history=requestHistory(body); }
    catch { return Response.json({error:'Invalid conversation history.'},{status:400,headers}); }
    const history=conversation_history.flatMap((m,i)=>m.role==='user'?[{question:m.content,...(conversation_history[i+1]?.role==='assistant'?{answer:conversation_history[i+1].content}:{})}]:[]);
    const sample=body.sample&&typeof body.sample==='object'?body.sample:undefined;
    const fallback=groundedReply(question,sample,history);
    if(env.ECOSIGNAL_CHATBOT_URL) {
      try {
        const conversationId=typeof body.conversation_id==='string'&&/^[a-zA-Z0-9_-]{1,80}$/.test(body.conversation_id)?body.conversation_id:crypto.randomUUID();
        const endpoint=new URL(env.ECOSIGNAL_CHATBOT_URL);if(endpoint.protocol!=='https:')throw Error('Invalid endpoint');
        const upstream=await fetch(endpoint,{method:'POST',headers:{'content-type':'application/json',...(env.ECOSIGNAL_CHATBOT_TOKEN?{authorization:'Bearer '+env.ECOSIGNAL_CHATBOT_TOKEN}:{})},body:JSON.stringify({question,user_id:`ecosignal-website:${conversationId}`,sample_id:null,mission_id:null,conversation_id:conversationId,conversation_history}),signal:AbortSignal.any([request.signal,AbortSignal.timeout(PROXY_TIMEOUT_MS)])});
        if(!upstream.ok)return Response.json({error:RETRY_MESSAGE},{status:upstream.status===504?504:upstream.status===429?429:503,headers:{...headers,'retry-after':upstream.headers.get('retry-after')||'5'}});
        const text=await upstream.text();if(text.length>64000)throw Error('Invalid response');
        return Response.json(normalizeChatReply(JSON.parse(text)),{headers});
      } catch(e) {const timeout=e instanceof Error&&['TimeoutError','AbortError'].includes(e.name);return Response.json({error:RETRY_MESSAGE},{status:timeout?504:503,headers});}
    }
    if(fallback.intent==='air_quality_data'&&/current|today|right now|local|near|aqi.*(here|cary|raleigh)|(cary|raleigh).*aqi/i.test(question)){
      try{const lat=typeof sample?.lat==='number'?sample.lat:35.7915,lon=typeof sample?.lon==='number'?sample.lon:-78.7811;const air=await getAirQuality(lat,lon);return new Response(JSON.stringify({...fallback,answer:`AirNow reports a U.S. AQI of ${Math.round(air.aqi!)} (${air.category}) for the selected location in ${air.reporting_area}, near ${lat.toFixed(4)}, ${lon.toFixed(4)}, at ${air.timestamp}.${air.stale?' One or more observations are older than three hours.':''} The highest reported pollutant is ${air.primary_pollutant}. Compare the same coordinates and hour on AirNow; forecasts and other monitor locations can differ.`,scan_evidence:[{label:'U.S. AQI · AirNow reported',value:String(Math.round(air.aqi!)),source:air.source},{label:'Observation timestamp',value:air.timestamp,source:air.source}],sources:[{source_id:'ES-AIR-LIVE-001',title:'AirNow location-specific observations',url:air.source_url,supports:'Reported AQI for the selected coordinates and observation time.'}],uncertainty:air.limitations,mode:'live_data_guidance'}),{headers});}catch{return new Response(JSON.stringify({...fallback,service_notice:'Current air-quality lookup failed. No AQI value is available.'}),{headers});}
    }
    if(env.GEMINI_API_KEY&&fallback.intent!=='security_redirect'){try{return new Response(JSON.stringify(await askGemini(question,history,fallback,env)),{headers})}catch(e){return new Response(JSON.stringify({error:RETRY_MESSAGE}),{status:503,headers:{...headers,'retry-after':'5'}})}}
    if(fallback.intent==='security_redirect')return Response.json(fallback,{headers});
    return Response.json({error:RETRY_MESSAGE},{status:503,headers:{...headers,'retry-after':'5'}});
  } catch {return new Response(JSON.stringify({error:'Could not read this question.'}),{status:400,headers});}
}

// Include proxy/body processing in the measurement, for success and failure.
export async function handleAsk(request:Request,env:ChatEnvironment) {
  const started=performance.now();
  const response=await handleAskRequest(request,env);
  response.headers.set('server-timing',`website;dur=${(performance.now()-started).toFixed(3)}`);
  return response;
}
