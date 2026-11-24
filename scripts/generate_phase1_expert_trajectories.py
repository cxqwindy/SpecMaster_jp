#!/usr/bin/env python3
"""Build deterministic five-tool SFT trajectories for the Phase1 expert negatives."""
from __future__ import annotations
import argparse,collections,json,sys
from pathlib import Path
sys.path.insert(0,'/fs_mol/cxq/Xmaster_AgentRL/Open-AgentRL')
from recipe.nmr_matching_agent.nmr_matching_runtime import NmrMatchingRuntime
from recipe.nmr_matching_agent.prompt import build_system_prompt
from recipe.nmr_matching_agent.cot_to_trajectory_v5 import _validate_think_number_provenance

CALLS=[('count_atoms',{'element':'C'}),('count_atoms',{'element':'H'}),('spectral_match',{}),('peak_assignment',{'nucleus':'C'}),('peak_assignment',{'nucleus':'H'}),('functional_group_scan',{})]
def summary(name,o):
 if name=='count_atoms':
  if o['element']=='C':return f"count_atoms(C) returned candidate_count={o['candidate_count']}, experimental_signals={o['experimental_distinct_signals']}, match={str(o['match']).lower()}"
  return f"count_atoms(H) returned candidate_count={o['candidate_count']}, experimental_total_nH={o['experimental_total_nH']}, match={str(o['match']).lower()}, exact_match={str(o.get('exact_match')).lower()}, difference={o.get('difference')}, tolerance={o.get('tolerance')}"
 if name=='spectral_match':return f"spectral_match returned similarity={o['similarity']}, c_sim={o['c_sim']}, h_sim={o['h_sim']}, n_pred_c={o['n_pred_c']}, n_exp_c={o['n_exp_c']}, n_pred_h={o['n_pred_h']}, n_exp_h={o['n_exp_h']}"
 if name=='peak_assignment':return f"peak_assignment({o['nucleus']}) returned n_matched={o['n_matched']}, n_unmatched_exp={o['n_unmatched_exp']}, n_unmatched_pred={o['n_unmatched_pred']}, n_large_residual={o['n_large_residual']}, max_abs={o['residual_stats']['max_abs']} ppm"
 return f"functional_group_scan returned n_groups={o['n_groups']}, groups_present={o['groups_present']}"
def contradiction(c,h):
 candidates=[]
 for o in (c,h):
  if o.get('n_large_residual',0)>0 or o.get('n_unmatched_exp',0)>0:
   threshold=float(o.get('large_residual_threshold_ppm') or (8.0 if o['nucleus']=='C' else .35));mx=float((o.get('residual_stats') or {}).get('max_abs') or 0);score=mx/threshold+o.get('n_large_residual',0)+o.get('n_unmatched_exp',0)
   text=f"peak_assignment({o['nucleus']}) found n_large_residual={o['n_large_residual']}, n_unmatched_exp={o['n_unmatched_exp']}, and max_abs={o['residual_stats']['max_abs']} ppm."
   candidates.append((score,text,o['nucleus']))
 if not candidates:raise ValueError('no assignment contradiction')
 return max(candidates,key=lambda x:(x[0],x[2]))[1]
def nmr_text(x):
 cs=', '.join(f"{float(p['delta (ppm)']):g}" for p in x['c_nmr_peaks']);hs=[]
 for p in x['h_nmr_peaks']:
  d=float(p.get('centroid',p.get('delta')));cat=p.get('category') or 'm';j=str(p.get('j_values') or '').strip('_ ');hs.append(f"{d:g} ({cat}{', J='+j.replace('_', ', ')+' Hz' if j else ''}, {p.get('nH')}H)")
 return f"Reported metadata: frequency={x.get('nmr_frequency')}, solvent={x.get('nmr_solvent')}\n13C NMR: {cs}\n1H NMR: "+', '.join(hs)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--input',default='data/phase1_expert_perturbations_100.json');ap.add_argument('--cache',default='data/phase1_prediction_cache.json');ap.add_argument('--output',default='distill_out/phase1_expert_trajectories_100.jsonl');ap.add_argument('--review',default='docs/phase1_expert_trajectories_review100.md');ap.add_argument('--limit',type=int,default=0);ap.add_argument('--diverse-preview',action='store_true');ap.add_argument('--case-ids',nargs='*');a=ap.parse_args();rows=json.load(open(a.input));
 if a.case_ids:
  index={x['case_id']:x for x in rows};rows=[index[x] for x in a.case_ids]
 elif a.limit:
  if a.diverse_preview:
   by={f:[x for x in rows if x['family']==f] for f in ('halogen_swap','ring_chalcogen_swap','positional_isomer','connectivity_isomer')};order=['halogen_swap','ring_chalcogen_swap','positional_isomer','connectivity_isomer','positional_isomer','connectivity_isomer'];seen=collections.Counter();picked=[]
   for fam in order:
    if len(picked)>=a.limit:break
    picked.append(by[fam][seen[fam]]);seen[fam]+=1
   rows=picked
  else:rows=rows[:a.limit]
 cache=json.load(open(a.cache));out=[];md=[f'# Phase1专家负样本：五工具SFT轨迹人工审阅（{len(rows)}条）','', '- 每条固定调用 count C/H、spectral、assignment C/H、FG；最终No只引用真实assignment矛盾。','']
 for idx,x in enumerate(rows,1):
  user={k:x.get(k) for k in ('candidate_smiles','c_nmr_peaks','h_nmr_peaks','nmr_solvent','nmr_frequency')};rt=NmrMatchingRuntime(user,prediction_cache=cache,allow_remote_predict=False,h_tolerance=2);messages=[{'role':'system','content':build_system_prompt()},{'role':'user','content':json.dumps(user,ensure_ascii=False)}];evidence=[];prev=''
  for turn,(name,args) in enumerate(CALLS,1):
   think=('Establish that gross counts are non-discriminative, then inspect one-to-one shift assignments. Start with count_atoms(C).' if turn==1 else f"{prev}. Next call {name}{'('+args['nucleus']+')' if 'nucleus' in args else ''}.")
   call={'name':name,'arguments':args};messages.append({'role':'assistant','content':f"<think>{think}</think>\n<tool_call>{json.dumps(call,ensure_ascii=False)}</tool_call>"});obs,_=rt.handlers[name](args);messages.append({'role':'tool','name':name,'content':json.dumps(obs,ensure_ascii=False)});evidence.append((name,obs));prev=summary(name,obs)
  pc=evidence[3][1];ph=evidence[4][1];key=contradiction(pc,ph);base=' ; '.join(summary(n,o) for n,o in evidence);final_think=f"{base}. C/H counts pass and are non-discriminative, but the one-to-one assignment contains a local shift contradiction. {key} Therefore final_answer = No.";ans={'final_answer':'No','confidence':0.95,'key_evidence':key};messages.append({'role':'assistant','content':f"<think>{final_think}</think>\n<answer>{json.dumps(ans,ensure_ascii=False)}</answer>"});v=_validate_think_number_provenance(messages)
  if v:raise RuntimeError(f"{x['case_id']} provenance: {v}")
  rec={'messages':messages,'meta':{'case_id':x['case_id'],'candidate_smiles':x['candidate_smiles'],'gt_smiles':x['gt_smiles'],'label':0,'negative_type':x['negative_type'],'family':x['family'],'perturbation_subtype':x['perturbation_subtype'],'final_answer':'No','n_tool_calls':6,'assignment_contract_version':2,'provenance_violations':[],'source':'phase1_expert_deterministic'}};out.append(rec)
  md += [f"## {idx:03d}. {x['case_id']} — {x['negative_type']}",'',f"- **正确结构**：`{x['gt_smiles']}`",f"- **负样本结构**：`{x['candidate_smiles']}`",f"- **核磁文本**：\n\n{nmr_text(x)}",f"- **工具摘要**：`{summary('spectral_match',evidence[2][1])}`",f"- **C assignment**：`{summary('peak_assignment',pc)}`",f"- **H assignment**：`{summary('peak_assignment',ph)}`",f"- **Final think**：{final_think}",f"- **Final answer**：`{json.dumps(ans,ensure_ascii=False)}`",'']
 Path(a.output).parent.mkdir(parents=True,exist_ok=True);Path(a.output).write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in out)+'\n');Path(a.review).write_text('\n'.join(md));print(f'wrote {len(out)} trajectories -> {a.output}; review -> {a.review}')
if __name__=='__main__':main()
