"""Browser mapping regression: independent mesh-space containment and picking."""
import asyncio, json, urllib.request
from pathlib import Path
import websockets
ROOT = Path(__file__).resolve().parent
async def main():
    pages = json.load(urllib.request.urlopen('http://127.0.0.1:9227/json'))
    page = next(p for p in pages if p.get('type') == 'page')
    async with websockets.connect(page['webSocketDebuggerUrl'], max_size=20*1024**2) as ws:
        async def evaluate(expression):
            await ws.send(json.dumps({'id':1,'method':'Runtime.evaluate','params':{'expression':expression,'returnByValue':True}}))
            while True:
                r=json.loads(await ws.recv())
                if r.get('id')==1:
                    result=r['result']
                    if 'exceptionDetails' in result: raise AssertionError(result['exceptionDetails'])
                    return result['result'].get('value')
        result=await evaluate('''(()=>{
const t=viewerTest,T=THREE,checks=[];function ok(name,value){checks.push({name,pass:!!value});if(!value)throw Error(name)}
const meshes=t.layers.body.children.filter(o=>o.isMesh&&o.geometry.attributes.position.count>100);
const head=meshes.find(o=>(o.userData.meshName||'').includes('c_head'));
const abdomen=meshes.find(o=>(o.userData.meshName||'').includes('abdomen')&&o.userData.index===12);
ok('head and abdominal host meshes exist',head&&abdomen);
const bodyRoot=new T.Vector3(),initialLocal={};
for(const p of [0,.25,.7,1]){
 t.setProgress(p);
 for(const depth of [0,40,100]){
  const slider=document.getElementById('depth');slider.value=depth;slider.dispatchEvent(new Event('input'));t.scene.updateMatrixWorld(true);
  for(const [kind,host] of [['neural',head],['electrodes',head],['calcium',abdomen]]){
   host.geometry.computeBoundingBox();const box=host.geometry.boundingBox.clone().expandByScalar(1e-6),inverse=host.matrixWorld.clone().invert();let count=0,outside=0;
   t.layers[kind].traverse(o=>{const a=o.geometry?.attributes?.position;if(!a)return;for(let i=0;i<a.count;i++){const v=new T.Vector3().fromBufferAttribute(a,i).applyMatrix4(o.matrixWorld).applyMatrix4(inverse);count++;if(!box.containsPoint(v))outside++;}});
   ok('mesh-local bounding-box containment '+kind+' p='+p+' depth='+depth,count>0&&outside===0);
   if(kind!=='electrodes'){
    const o=t.layers[kind].children.find(o=>o.isPoints),v=new T.Vector3().fromBufferAttribute(o.geometry.attributes.position,0).applyMatrix4(o.matrixWorld).applyMatrix4(inverse);
    if(!initialLocal[kind])initialLocal[kind]=v;ok('rigid host attachment '+kind+' p='+p,v.distanceTo(initialLocal[kind])<1e-6);
   }
  }
 }
}
const filter=document.getElementById('pick-filter');
t.setProgress(0);t.focus('neural');t.scene.updateMatrixWorld(true);t.camera.updateMatrixWorld(true);
if(filter){filter.value='auto';filter.dispatchEvent(new Event('change'));}
const pixel=t.project('neural',0);t.clearSelection();const hit=t.pickAt(pixel.x,pixel.y);ok('internal picking not intercepted by transparent body',hit&&hit.kind!=='body');
t.clearSelection();t.focus('all');t.renderer.render(t.scene,t.camera);
return {checks,limitation:'Containment verified against mesh-local bounding boxes, not watertight anatomical volume.',mapping:t.mapping||t.mappings||null,errors:viewerErrors};})()''')
        (ROOT/'viewer/body_internal_test.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
        assert not result['errors'],result['errors']
        print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__': asyncio.run(main())
