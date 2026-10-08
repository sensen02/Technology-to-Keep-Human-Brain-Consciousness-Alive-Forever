"""Local CDP browser test without installing automation dependencies."""
import asyncio,json,urllib.request,base64
from pathlib import Path
import websockets
ROOT=Path(__file__).resolve().parent
async def main():
    pages=json.load(urllib.request.urlopen('http://127.0.0.1:9227/json'))
    page=next(p for p in pages if p.get('type')=='page')
    async with websockets.connect(page['webSocketDebuggerUrl'],max_size=30*1024**2) as ws:
        counter=0
        async def cmd(method,params={}):
            nonlocal counter
            counter+=1;i=counter
            await ws.send(json.dumps({'id':i,'method':method,'params':params}))
            while True:
                r=json.loads(await ws.recv())
                if r.get('id')==i:
                    if 'error' in r:raise RuntimeError(r['error'])
                    return r.get('result',{})
        await cmd('Page.enable');await cmd('Runtime.enable')
        await cmd('Emulation.setDeviceMetricsOverride',{'width':1440,'height':1000,'deviceScaleFactor':1,'mobile':False})
        await cmd('Page.navigate',{'url':'http://127.0.0.1:8766/'})
        ready=False
        for _ in range(60):
            r=await cmd('Runtime.evaluate',{'expression':'Boolean(window.viewerReady)','returnByValue':True})
            if r.get('result',{}).get('value'):ready=True;break
            await asyncio.sleep(.25)
        r=await cmd('Runtime.evaluate',{'expression':'JSON.stringify({ready:window.viewerReady,errors:window.viewerErrors,test:window.viewerTest?Object.keys(window.viewerTest):[],canvas:document.querySelectorAll("canvas").length,body:document.body.innerText.slice(0,4000)})','returnByValue':True})
        data=json.loads(r['result']['value'])
        expression='''JSON.stringify((()=>{const t=window.viewerTest, checks=[]; const ok=(n,v)=>{checks.push({name:n,pass:!!v});if(!v)throw Error(n)};
        ok('actual soma/electrode geometry',t.counts.neural===6000&&t.counts.electrodes===256);
        ok('select electrode',t.select('electrodes',120));ok('electrode inspector',document.getElementById('inspector').textContent.includes('20'));
        ok('select neuron exact root string',t.select('neural',0)&&document.getElementById('inspector').textContent.includes(window.VIEWER_DATA.neural.ids[0]));
        t.setProgress(.8);ok('timeline state',t.progress===.8);ok('membrane value shown',document.getElementById('inspector').textContent.includes('膜电位'));
        t.focus('body');ok('body focus',t.focusName==='body');t.focus('calcium');t.select('calcium',10);ok('calcium inspection',document.getElementById('inspector').textContent.includes('上皮'));
        const layer=document.getElementById('layer-neural');layer.checked=false;layer.dispatchEvent(new Event('change'));ok('hide neural',!t.layers.neural.visible);layer.checked=true;layer.dispatchEvent(new Event('change'));
        const clip=document.getElementById('clip');clip.value=50;clip.dispatchEvent(new Event('input'));ok('section plane',t.renderer.clippingPlanes.length===1);clip.value=100;clip.dispatchEvent(new Event('input'));
        const ca=document.getElementById('calcium-variable');ca.value='er_values';ca.dispatchEvent(new Event('change'));t.select('calcium',10);ok('ER calcium field',document.getElementById('inspector').textContent.includes('ER'));ca.value='ip3_values';ca.dispatchEvent(new Event('change'));ok('IP3 field',document.getElementById('inspector').textContent.includes('IP3'));ca.value='values';ca.dispatchEvent(new Event('change'));
        t.focus('body');t.setProgress(.9);ok('actual body meshes',t.layers.body.children.some(o=>o.geometry&&o.geometry.attributes.position.count>100));
        t.focus('neural');t.setProgress(.3);t.select('electrodes',120);t.renderer.render(t.scene,t.camera);return checks})())'''
        tests=await cmd('Runtime.evaluate',{'expression':expression,'returnByValue':True})
        if 'exceptionDetails' in tests:raise AssertionError(tests['exceptionDetails'])
        data['interactions']=json.loads(tests['result']['value'])
        extra=await cmd('Runtime.evaluate',{'expression':'''(async()=>{for(let k=0;k<30;k++){if(document.getElementById('service-health')?.textContent.includes('cell-workbench'))break;await new Promise(r=>setTimeout(r,100));} const t=viewerTest;t.select('electrodes',120);const z=t.selectionRing.position.z;const depth=document.getElementById('depth');depth.value=40;depth.dispatchEvent(new Event('input'));if(t.selectionRing.position.z===z)throw Error('selection did not follow insertion');depth.value=100;depth.dispatchEvent(new Event('input'));const chain=[...document.querySelectorAll('button')].find(b=>b.textContent.startsWith('界面阻抗'));chain.click();t.setProgress(.5);if(!document.getElementById('inspector').textContent.includes('记录链节点'))throw Error('timeline overwrites chain');if(!document.getElementById('service-health').textContent.includes('cell-workbench'))throw Error('service not connected');return {service:document.getElementById('service-health').textContent,insertionFollow:true,chainRetained:true};})()''','awaitPromise':True,'returnByValue':True})
        if 'exceptionDetails' in extra:raise AssertionError(extra['exceptionDetails'])
        data['service_interactions']=extra['result']['value']
        internal=await cmd('Runtime.evaluate',{'expression':'''JSON.stringify((()=>{const t=viewerTest,T=THREE,checks=[];const ok=(name,value)=>{checks.push({name,pass:!!value});if(!value)throw Error(name)};ok('fixed source electrode geometry',VIEWER_DATA.electrodes.diameter_um===7&&VIEWER_DATA.electrodes.pitch_um===20);const positions=[];for(const progress of [0,.25,.7,1]){t.setProgress(progress);t.select('neural',0);t.scene.updateMatrixWorld(true);const a=t.layers.neural.children.find(o=>o.isPoints).geometry.attributes.position;const p=new T.Vector3().fromBufferAttribute(a,0);t.layers.neural.localToWorld(p);ok('selection follows full transform '+progress,p.distanceTo(t.selectionRing.position)<1e-7);positions.push(p.toArray());t.select('electrodes',120);const capture=document.getElementById('capture-neighborhood');capture.checked=true;capture.dispatchEvent(new Event('change'));ok('capture follows contact '+progress,t.captureMesh.position.distanceTo(t.selectionRing.position)<1e-7&&t.captureMesh.visible);}ok('internal layers follow animated body',new T.Vector3(...positions[0]).distanceTo(new T.Vector3(...positions[2]))>1e-7);t.setProgress(0);t.focus('all');t.clearSelection();return checks})())''','returnByValue':True})
        if 'exceptionDetails' in internal:raise AssertionError(internal['exceptionDetails'])
        data['body_internal_interactions']=json.loads(internal['result']['value'])
        await cmd('Emulation.setDeviceMetricsOverride',{'width':390,'height':844,'deviceScaleFactor':1,'mobile':True})
        await asyncio.sleep(.3)
        mobile=await cmd('Runtime.evaluate',{'expression':'''(()=>{const r=document.getElementById('viewport').getBoundingClientRect();return {width:r.width,height:r.height,overflow:document.documentElement.scrollWidth>window.innerWidth,errors:viewerErrors}})()''','returnByValue':True})
        features=await cmd('Runtime.evaluate',{'expression':'''(()=>{const t=viewerTest,checks=[];const ok=(n,v)=>{checks.push({name:n,pass:!!v});if(!v)throw Error(n)};ok('manifest version loaded',!!VIEWER_MANIFEST.run_id);t.focus('neural');ok('local shell isolates head',t.bodyMeshes.filter(m=>m.o.visible).every(m=>m.o.userData.meshName.endsWith('/c_head')));document.getElementById('save-view').click();t.focus('all');document.getElementById('restore-view').click();ok('bookmark restores focus',t.focusName==='neural');const c=document.getElementById('accessible-colors');c.checked=true;c.dispatchEvent(new Event('change'));ok('numeric legend has units',document.getElementById('numeric-legend').textContent.includes('Hz'));const a=document.getElementById('clip-axis');a.value='X';a.dispatchEvent(new Event('change'));document.getElementById('clip').value=50;document.getElementById('clip').dispatchEvent(new Event('input'));ok('X clipping plane',t.renderer.clippingPlanes[0].normal.x===-1);document.getElementById('clip').value=100;document.getElementById('clip').dispatchEvent(new Event('input'));t.select('electrodes',0);ok('channel evidence explicit',document.getElementById('inspector').textContent.includes('该通道台架证据'));const p=[20,30,40],w=Workbench.transforms.sourceToWorld(t.layers.neural,p);ok('source transform reversible',Workbench.transforms.worldToSource(t.layers.neural,w).distanceTo(new THREE.Vector3(...p))<1e-6);document.getElementById('reset-camera').click();ok('reset camera',t.focusName==='all');c.checked=false;c.dispatchEvent(new Event('change'));t.clearSelection();return checks;})()''','returnByValue':True})
        if 'exceptionDetails' in features:raise AssertionError(features['exceptionDetails'])
        data['p123_features']=features['result']['value']
        data['mobile']=mobile['result']['value']
        assert data['mobile']['width']>100 and data['mobile']['height']>100 and not data['mobile']['errors']
        await cmd('Emulation.setDeviceMetricsOverride',{'width':1440,'height':1000,'deviceScaleFactor':1,'mobile':False})
        await asyncio.sleep(.3)
        await cmd('Runtime.evaluate',{'expression':"viewerTest.focus('all');viewerTest.renderer.render(viewerTest.scene,viewerTest.camera)"})
        await asyncio.sleep(.3)  # Let projected labels catch up after the final camera focus.
        print(json.dumps(data,ensure_ascii=False,indent=2))
        (ROOT/'viewer/browser_test.json').write_text(json.dumps(data,ensure_ascii=False,indent=2))
        img=await cmd('Page.captureScreenshot',{'format':'png','captureBeyondViewport':False})
        (ROOT/'viewer/browser_screenshot.png').write_bytes(base64.b64decode(img['data']))
        for name in ['neural','calcium']:
            await cmd('Runtime.evaluate',{'expression':f"viewerTest.focus('{name}');viewerTest.setProgress(.6);viewerTest.renderer.render(viewerTest.scene,viewerTest.camera)"})
            await asyncio.sleep(.3)
            detail=await cmd('Page.captureScreenshot',{'format':'png','captureBeyondViewport':False})
            (ROOT/f'viewer/{name}_screenshot.png').write_bytes(base64.b64decode(detail['data']))
        await cmd('Runtime.evaluate',{'expression':"viewerTest.focus('all')"})
        if not ready:raise AssertionError('viewer not ready')
        if data.get('errors'):raise AssertionError(data['errors'])
if __name__=='__main__':asyncio.run(main())
