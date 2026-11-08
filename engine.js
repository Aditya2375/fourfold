'use strict';
function winner(b){for(let r=0;r<6;r++)for(let c=0;c<7;c++){let p=b[r*7+c];if(!p)continue;for(let [dr,dc] of [[0,1],[1,0],[1,1],[1,-1]]){if(r+3*dr<6&&c+3*dc>=0&&c+3*dc<7&&[1,2,3].every(i=>b[(r+i*dr)*7+c+i*dc]===p))return p}}return 0}
function legal(b){return [0,1,2,3,4,5,6].filter(c=>b[c]===0)}
function move(b,t,c){if(!legal(b).includes(c))throw Error('Illegal column');let a=b.slice();for(let r=5;r>=0;r--)if(a[r*7+c]===0){a[r*7+c]=t;return a}}
function predict(b,t,m){let w=m.weights,h=w['fc.weight'].map((row,i)=>Math.max(0,row.reduce((s,v,j)=>s+v*b[j]*t,w['fc.bias'][i]))),log=w['policy.weight'].map((row,i)=>row.reduce((s,v,j)=>s+v*h[j],w['policy.bias'][i])),ls=legal(b),mx=Math.max(...ls.map(c=>log[c])),p=log.map((v,c)=>ls.includes(c)?Math.exp(v-mx):0),sum=p.reduce((a,b)=>a+b,0),v=Math.tanh(w['value.weight'][0].reduce((s,x,j)=>s+x*h[j],w['value.bias'][0]));return {policy:p.map(x=>x/sum),value:v}}
class Node{constructor(b,t,p=1){this.board=b;this.turn=t;this.prior=p;this.n=0;this.total=0;this.children=[];this.end=winner(b);this.draw=!legal(b).length}}
function expand(n,m){if(n.end)return -1;if(n.draw)return 0;let r=predict(n.board,n.turn,m);n.children=legal(n.board).map(c=>({c,node:new Node(move(n.board,n.turn,c),-n.turn,r.policy[c])}));return r.value}
function rootNode(b,t,m){let n=new Node(b,t);expand(n,m);return n}
function simulate(root,m){let n=root,path=[n];while(n.children.length){let best=null,bestScore=-Infinity;for(let {node:ch} of n.children){let score=-(ch.n?ch.total/ch.n:0)+1.5*ch.prior*Math.sqrt(Math.max(1,n.n))/(1+ch.n);if(score>bestScore){bestScore=score;best=ch}}n=best;path.push(n)}let v=expand(n,m);for(let i=path.length-1;i>=0;i--){path[i].n++;path[i].total+=v;v=-v}}
function report(root){let visits=Array(7).fill(0),values=Array(7).fill(null),priors=Array(7).fill(0);for(let {c,node:n} of root.children){visits[c]=n.n;values[c]=n.n?-n.total/n.n:null;priors[c]=n.prior}let sum=visits.reduce((a,b)=>a+b,0),policy=visits.map(x=>sum?x/sum:0);return{visits,values,priors,policy,value:root.n?root.total/root.n:0}}
if(typeof module!=='undefined')module.exports={winner,legal,move,predict,Node,rootNode,simulate,report};
