import test from 'node:test';
import assert from 'node:assert/strict';
import {writeFile,readFile} from 'node:fs/promises';
import {chooseResult,snapshotMetadata,captureReport,reportZip,crc32} from '../web/incident.mjs';
const r={t:10000,w:100,h:50,dets:[[1,2,3,4,'car',.8,'cached']],motion:[],regions:[],stats:{frames:1},mb:16};
const input=()=>({choice:chooseResult([r],10),results:[r],show:{boxes:true,motion:false,regions:true},wanted:true,camera:'driveway',events:[]});
test('selection preserves existing matching slack and all fallback paths',()=>{
 assert.equal(chooseResult([r],10).selection,'matched');
 assert.equal(chooseResult([r],9).selection,'matched');
 assert.equal(chooseResult([r],8).selection,'no_result_at_or_before_frame');
 assert.equal(chooseResult([r],undefined).selection,'no_renderer_timestamp');
 assert.equal(chooseResult([],10).selection,'no_results');
 assert.equal(chooseResult([r],8).result,r);
});
test('AI off is explicit, capture keeps independent timing and dimensions',()=>{
 const i=input();i.wanted=false;i.results=[];i.choice=chooseResult([],10);
 const m=snapshotMetadata(i);assert.equal(m.detections.reason,'ai_off');assert.equal(m.timing.result_minus_video_ms,null);assert.equal(i.wanted,false);
 const x=snapshotMetadata(input());assert.equal(x.timing.result_minus_video_ms,0);assert.equal(x.detection_size.width,100);assert.equal(x.box_coordinate_space,'detection_size');
});
test('metadata allowlist excludes URLs and raw extra fields',()=>{
 const i=input();i.url='https://relay/?token=SECRET';i.choice={...i.choice,result:{...r,url:i.url}};i.events=[{id:1,start:1,url:i.url,token:'SECRET'}];
 const s=JSON.stringify(snapshotMetadata(i));assert.ok(!s.includes('SECRET'));assert.ok(!s.includes('https://'));assert.ok(!s.includes('token'));
});
test('history, events, toggles and selected record survive later mutation',()=>{
 const i=input();i.results=structuredClone(i.results);i.choice=chooseResult(i.results,10);i.events=[{id:7,start:1,labels:['car']}];
 const m=snapshotMetadata(i);i.results[0].dets[0][0]=99;i.results.length=0;i.show.boxes=false;i.events[0].labels.push('truck');
 assert.equal(m.detections.selected.dets[0][0],1);assert.equal(m.layers.boxes,true);assert.deepEqual(m.events[0].labels,['car']);
});
test('missing pixels still produce portable ZIP with explicit absence',async()=>{
 const report=await captureReport(input(),null,null);assert.equal(report.metadata.frame.available,false);
 const zip=await reportZip(report,{note:'Please inspect'});assert.equal(zip.type,'application/zip');
 await writeFile('/tmp/codec-incident-test.zip',new Uint8Array(await zip.arrayBuffer()));
});
test('slow toBlob cannot mix later source pixels, toggles or histories',async()=>{
 const old=globalThis.document;let copies=[];
 globalThis.document={createElement(){const c={width:0,height:0,pixel:null,getContext(){return {drawImage(src){c.pixel=src.pixel},getImageData(){return {}}}},toBlob(cb){copies.push(()=>cb(new Blob([c.pixel])))}};return c;}};
 try{const i=input(),frame={width:200,height:100,pixel:'old-frame'},overlay={width:100,height:50,pixel:'old-overlay'};
 const promise=captureReport(i,frame,overlay);frame.pixel='new';overlay.pixel='new';i.show.boxes=false;i.results=[];copies.forEach(f=>f());
 const report=await promise;assert.equal(await report.frame.text(),'old-frame');assert.equal(await report.overlay.text(),'old-overlay');assert.equal(report.metadata.layers.boxes,true);assert.equal(report.metadata.video_size.width,200);assert.equal(report.metadata.overlay.width,100);
 }finally{globalThis.document=old;}
});
test('tainted readback produces metadata instead of aborting capture',async()=>{
 const old=globalThis.document;globalThis.document={createElement(){return {getContext(){return {drawImage(){throw Object.assign(new Error(),{name:'SecurityError'})}}}}}};
 try{const report=await captureReport(input(),{width:100,height:50},null);assert.equal(report.metadata.frame.reason,'SecurityError');}finally{globalThis.document=old;}
});
test('CRC matches known ZIP polynomial',()=>assert.equal(crc32(new TextEncoder().encode('123456789')),0xcbf43926));
test('integration captures inside draw and does not add inference/subscription code',async()=>{
 const camera=await readFile(new URL('../web/camera.js',import.meta.url),'utf8');
 const reporter=await readFile(new URL('../web/incident.mjs',import.meta.url),'utf8');
 assert.ok(camera.includes('finishCapture(choice);'));
 assert.ok(camera.includes('Camera switched before capture'));
 assert.ok(!/fetch\(|subscribe\(|setAI\(|follow\(/.test(reporter));
 assert.ok(!reporter.includes('location.search'));
});
