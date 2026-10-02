// Portable local incident reports. No network, model calls, or stream subscriptions.
const finite = n => Number.isFinite(n) ? n : null;
export function chooseResult(results, timestamp) {
  if (!results.length) return {result: null, selection: "no_results", timestamp: finite(timestamp)};
  if (timestamp === undefined || timestamp === null || !Number.isFinite(timestamp))
    return {result: results[results.length - 1], selection: "no_renderer_timestamp", timestamp: null};
  let best;
  for (const r of results) if (r.t <= timestamp * 1000 + 1000 && (!best || r.t > best.t)) best = r;
  return {result: best ?? results[results.length - 1], selection: best ? "matched" : "no_result_at_or_before_frame", timestamp};
}
function record(r) {
  if (!r) return null;
  return {t: finite(r.t), w: finite(r.w), h: finite(r.h), full: !!r.full,
    dets: (r.dets ?? []).slice(0, 1000).map(d => d.slice(0, 7)),
    regions: (r.regions ?? []).slice(0, 1000).map(d => d.slice(0, 4)),
    motion: (r.motion ?? []).slice(0, 2000).map(d => d.slice(0, 4)), mb: finite(r.mb),
    truncated: (r.dets?.length ?? 0) > 1000 || (r.regions?.length ?? 0) > 1000 || (r.motion?.length ?? 0) > 2000,
    stats: Object.fromEntries(["frames", "inferred", "held", "ms", "full_ms"].map(k => [k, finite(r.stats?.[k])]))};
}
export function snapshotMetadata({choice, results, show, wanted, camera, events = []}) {
  const selected = record(choice.result);
  return {schema_version: 1, kind: "viewer_report", captured_at: new Date().toISOString(),
    camera: String(camera).slice(0, 128), video_clip: {available: false, reason: "viewer_has_no_video_history"},
    timing: {renderer_timestamp_raw: choice.timestamp, renderer_unit_assumed: "milliseconds",
      renderer_source: "moq-watch renderer.out.timestamp", presentation_semantics_verified: false,
      detection_timestamp_us: selected?.t ?? null,
      result_minus_video_ms: selected?.t != null && choice.timestamp != null ? selected.t / 1000 - choice.timestamp : null,
      note: "Alignment using the viewer's existing unit assumption; not end-to-end latency."},
    detection_size: selected ? {width: selected.w, height: selected.h} : null,
    box_coordinate_space: "detection_size", coordinate_mapping: "Not calibrated; preserve independent image dimensions.",
    ai_on: !!wanted, layers: {boxes: !!show.boxes, motion: !!show.motion, regions: !!show.regions},
    detections: {available: !!selected && wanted, reason: !wanted ? "ai_off" : choice.selection,
      selected, history: results.slice(-30).map(record), history_truncated: results.length > 30},
    events: events.slice(-10).map(e => ({camera: String(camera).slice(0,128), id: e.id, start: finite(e.start), end: finite(e.end),
      state: e.state, labels: (e.labels ?? []).slice(0,100), summary: String(e.summary ?? "").slice(0,4000),
      trace: {state: e.state === "analyzing" ? "pending" : "unavailable", reason: "not_instrumented", access: "unknown"}})),
    human: {issue: null, note: "", region: null}, source_revision: null};
}
// This copy is synchronous: later rendering, camera switches and slow PNG encoding cannot alter it.
function copyCanvas(source) {
  if (!source?.width || !source?.height) return {available: false, reason: "canvas_unavailable"};
  if (source.width * source.height > 8388608) return {available: false, reason: "pixel_limit"};
  try {
    const canvas = document.createElement("canvas"); canvas.width = source.width; canvas.height = source.height;
    const ctx = canvas.getContext("2d"); ctx.drawImage(source, 0, 0);
    ctx.getImageData(0, 0, 1, 1); // Detect tainted readback before async encoding.
    return {available: true, width: canvas.width, height: canvas.height, canvas};
  } catch (e) { return {available: false, reason: e.name || "readback_failed"}; }
}
async function encode(copy) {
  if (!copy.available) return {info: copy, blob: null};
  try {
    const blob = await new Promise((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error("encoding_timeout")), 5000);
      copy.canvas.toBlob(b => {clearTimeout(timer); b ? resolve(b) : reject(new Error("encoding_failed"));}, "image/png");
    });
    if (blob.size > 16 * 1024 * 1024) throw new Error("image_byte_limit");
    return {info: {available: true, width: copy.width, height: copy.height}, blob};
  } catch (e) { return {info: {available: false, reason: e.message}, blob: null}; }
}
export function captureReport(input, frame, overlay) {
  const metadata = snapshotMetadata(input);
  const clean = copyCanvas(frame), painted = copyCanvas(overlay);
  return Promise.all([encode(clean), encode(painted)]).then(([a,b]) => {
    metadata.frame = {...a.info, readback_content_verified: false}; metadata.overlay = b.info;
    metadata.video_size = a.info.available ? {width: a.info.width, height: a.info.height} : null;
    return {metadata, frame: a.blob, overlay: b.blob};
  });
}
const encoder = new TextEncoder();
export function crc32(bytes) {
  let c = 0xffffffff;
  for (const b of bytes) { c ^= b; for (let i=0;i<8;i++) c = (c >>> 1) ^ (0xedb88320 & -(c & 1)); }
  return (c ^ 0xffffffff) >>> 0;
}
export function zipStore(files) {
  const local=[], central=[]; let offset=0, centralSize=0;
  for (const [name, bytes] of files) {
    if (!/^[a-zA-Z0-9_.-]+$/.test(name)) throw new Error("Unsafe archive filename");
    const n=encoder.encode(name), crc=crc32(bytes), h=new Uint8Array(30+n.length), v=new DataView(h.buffer);
    v.setUint32(0,0x04034b50,true); v.setUint16(4,20,true); v.setUint32(14,crc,true);
    v.setUint32(18,bytes.length,true); v.setUint32(22,bytes.length,true); v.setUint16(26,n.length,true); h.set(n,30);
    const c=new Uint8Array(46+n.length), d=new DataView(c.buffer);
    d.setUint32(0,0x02014b50,true); d.setUint16(4,20,true); d.setUint16(6,20,true); d.setUint32(16,crc,true);
    d.setUint32(20,bytes.length,true); d.setUint32(24,bytes.length,true); d.setUint16(28,n.length,true); d.setUint32(42,offset,true); c.set(n,46);
    local.push(h,bytes); central.push(c); offset+=h.length+bytes.length; centralSize+=c.length;
  }
  if(offset+centralSize>40*1024*1024) throw new Error("Report exceeds 40 MiB");
  const end=new Uint8Array(22), v=new DataView(end.buffer); v.setUint32(0,0x06054b50,true);
  v.setUint16(8,files.length,true); v.setUint16(10,files.length,true); v.setUint32(12,centralSize,true); v.setUint32(16,offset,true);
  return new Blob([...local,...central,end],{type:"application/zip"});
}
export async function reportZip(report, human) {
  const m=structuredClone(report.metadata); m.human={issue: human.issue || null, note: String(human.note || "").slice(0,2000), region:null};
  const files=[];
  for(const [name, blob] of [["frame.png",report.frame],["overlay.png",report.overlay]]) if(blob) files.push([name,new Uint8Array(await blob.arrayBuffer())]);
  const json=(name,data)=>files.push([name,encoder.encode(JSON.stringify(data,null,2))]);
  const {detections,events,...incident}=m;
  json("incident.json",incident); json("detections.json",detections); json("events.json",events);
  const readme=`Codec Vision incident report\nCamera: ${m.camera}\nCaptured: ${m.captured_at}\nVideo clip: absent. This viewer keeps detection metadata, not rewindable video.\nClean frame: ${m.frame.available ? 'included; player readback content not independently verified' : m.frame.reason}\nOverlay: ${m.overlay.available ? 'included' : m.overlay.reason}\nBox coordinates: detection_size. Frame and overlay may have different dimensions.\nTiming: renderer unit follows existing viewer assumption; alignment delta is not live latency.\nWeave: no trace references recorded by this viewer version.\nPredictions are not reviewed labels. Human notes are separate in incident.json.\nNo Breakpoint or Weave account required to inspect this archive.\n`;
  files.push(["README.txt",encoder.encode(readme)]);
  if(globalThis.crypto?.subtle) { const hashes={}; for(const [name,bytes] of files) hashes[name]=Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256",bytes))).map(b=>b.toString(16).padStart(2,"0")).join(""); json("sha256.json",hashes); }
  return zipStore(files);
}
export function mountReporter(parent, getCamera, getContext) {
  const button=document.createElement("button"); button.textContent="Report incident"; button.type="button";
  const panel=document.createElement("section"); panel.className="incident-panel"; panel.hidden=true;
  panel.innerHTML=`<h2>Incident captured</h2><p class="incident-status"></p><div class="incident-preview"></div><label>Issue (optional) <select><option value="">Unspecified</option><option>Missing detection</option><option>Wrong box or class</option><option>Overlay timing</option><option>Other</option></select></label><label>Note (optional) <textarea maxlength="2000" rows="2"></textarea></label><button type="button" class="incident-download">Download report</button> <button type="button" class="incident-discard">Discard</button><p>No video clip is recorded. Predictions are not human labels. Nothing is uploaded.</p>`;
  parent.append(button,panel); let report=null, urls=[];
  const clear=()=>{for(const u of urls) URL.revokeObjectURL(u);urls=[];};
  button.onclick=async()=>{
    button.disabled=true; panel.hidden=false; panel.querySelector(".incident-status").textContent="Capturing on the next rendered frame…";
    report=null;clear();panel.querySelector(".incident-preview").replaceChildren();panel.querySelector("select").value="";panel.querySelector("textarea").value="";
    const download=panel.querySelector(".incident-download"); download.disabled=true;
    try {
      report=await getCamera().capture(getContext()); const m=report.metadata;
      panel.querySelector(".incident-status").textContent=`${m.camera} · ${m.captured_at} · ${m.detections.reason} · alignment delta ${m.timing.result_minus_video_ms ?? 'unavailable'} ms · AI ${m.ai_on?'on':'off'} · boxes ${m.layers.boxes?'on':'off'}, motion ${m.layers.motion?'on':'off'}`;
      for(const [label,blob] of [["Clean frame",report.frame],["Overlay (separate coordinates)",report.overlay]]) if(blob){const fig=document.createElement("figure"),img=document.createElement("img"),cap=document.createElement("figcaption");img.src=URL.createObjectURL(blob);urls.push(img.src);img.alt=label;cap.textContent=label;fig.append(img,cap);panel.querySelector(".incident-preview").append(fig);}
      download.disabled=false;
    }catch(e){panel.querySelector(".incident-status").textContent="Capture unavailable: "+e.message;}finally{button.disabled=false;}
  };
  panel.querySelector(".incident-download").onclick=async()=>{if(!report)return;const b=panel.querySelector(".incident-download");b.disabled=true;try{const zip=await reportZip(report,{issue:panel.querySelector("select").value,note:panel.querySelector("textarea").value});const a=document.createElement("a"),u=URL.createObjectURL(zip);a.href=u;a.download="codec-incident-"+Date.now()+".zip";a.click();setTimeout(()=>URL.revokeObjectURL(u),1000);}catch(e){panel.querySelector(".incident-status").textContent="Download failed: "+e.message;}finally{b.disabled=false;}};
  panel.querySelector(".incident-discard").onclick=()=>{report=null;clear();panel.hidden=true;};
  addEventListener("keydown",e=>{if(e.key.toLowerCase()==="r"&&!e.repeat&&!e.ctrlKey&&!e.metaKey&&!e.altKey&&!e.target.closest?.("input,textarea,select,[contenteditable]")){e.preventDefault();if(!button.disabled)button.click();}});
}
