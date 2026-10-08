(function(){'use strict';window.Workbench=window.Workbench||{};
function channelEvidence(report,id,index){const matching=(report?.checks||[]).filter(c=>[c.channel,c.channel_id,c.electrode_id].some(v=>v!==undefined&&(String(v)===String(id))));return {channel_id:String(id),report_status:report?.status||'NOT_RUN',checks:matching,notice:matching.length?'检查输入证据，不证明体内性能':'没有匹配此通道的台架证据；不把全局检查冒充该通道测量'};}
function color(value,range,color,accessible=false){if(value===null||value===undefined||!Number.isFinite(value)){color.setHex(0x808080);return;}const x=Math.max(0,Math.min(1,(value-range[0])/Math.max(1e-12,range[1]-range[0])));if(accessible)color.setRGB(.1+.85*x,.4+.3*x,.9-.75*x);else color.setHSL(.53*(1-x),.9,.5);}
window.Workbench.inspection={channelEvidence,color};})();
