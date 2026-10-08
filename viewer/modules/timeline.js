/* Independent clocks: normalized display playback never claims measurement synchronization. */
(function(){'use strict';const cache=new WeakMap();
function index(data,key,progress){const f=Array.isArray(data[key])?data[key]:[];if(!f.length)return -1;let c=cache.get(data);if(!c){c={};cache.set(data,c);}if(!(key in c)){const times=data.times_s;c[key]=Array.isArray(times)&&times.length===f.length&&times.every((v,i)=>Number.isFinite(v)&&(!i||v>=times[i-1]))&&times.at(-1)>times[0]?times:null;}const times=c[key],p=Math.max(0,Math.min(1,Number(progress)||0));if(!times)return Math.floor(p*(f.length-1));const target=times[0]+p*(times.at(-1)-times[0]);let lo=0,hi=times.length-1;while(lo<hi){const mid=Math.ceil((lo+hi)/2);if(times[mid]<=target)lo=mid;else hi=mid-1;}return lo;}
function value(data,key,i,p){const row=data[key]?.[index(data,key,p)],v=row?.[i];return typeof v==='number'&&Number.isFinite(v)?v:null;}
window.Workbench=window.Workbench||{};window.Workbench.timeline={index,value,mode:'independent-normalized-replay'};
})();
