#!/usr/bin/env python3
"""Generate Phase-1 count/signal-preserving expert perturbation pilot."""
from __future__ import annotations
import argparse,collections,json,math,sys
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors,rdFMCS
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from scipy.optimize import linear_sum_assignment
sys.path.insert(0,'/fs_mol/cxq/SpecMaster_jp/perturbation_generator');sys.path.insert(0,'/fs_mol/cxq/SpecMaster_jp/scripts')
from nmr_predictor import predict_nmr_batch,_canonical
from spectral_scorer import spectral_similarity
from prepare_v6_4_sft_data import read_jsonl,split_grouped_stratified,DEFAULT_INPUT
FP=GetMorganGenerator(radius=2,fpSize=2048)
QUOTA={'halogen_swap':25,'ring_chalcogen_swap':25,'positional_isomer':35,'connectivity_isomer':15}

def nof(m):return all(a.GetAtomicNum()!=9 for a in m.GetAtoms())
def hcount(m):return sum(a.GetAtomicNum()==1 for a in Chem.AddHs(m).GetAtoms())
def symmetry_signature(m):
 ranks=list(Chem.CanonicalRankAtoms(m,breakTies=False));c=len({ranks[a.GetIdx()] for a in m.GetAtoms() if a.GetAtomicNum()==6});hg=collections.defaultdict(int)
 for a in m.GetAtoms():
  h=a.GetTotalNumHs(includeNeighbors=True)
  if h:hg[ranks[a.GetIdx()]]+=h
 return c,sorted(hg.values())
def mcs_coverage(a,b):
 r=rdFMCS.FindMCS([a,b],ringMatchesRingOnly=True,completeRingsOnly=True,timeout=2)
 return r.numAtoms/max(1,min(a.GetNumHeavyAtoms(),b.GetNumHeavyAtoms()))
def canon(m):return Chem.MolToSmiles(m,True)
def swap(m,i,z):
 rw=Chem.RWMol(m);a=rw.GetAtomWithIdx(i);a.SetAtomicNum(z);a.SetFormalCharge(0);a.SetNoImplicit(False)
 try:n=rw.GetMol();Chem.SanitizeMol(n);return canon(n)
 except:return None
def parents():
 d={}
 for line in open('/fs_mol/cxq/SpecMaster_jp/perturbation_output_1000/perturbed_1000_benchmark.jsonl'):
  r=json.loads(line);g=r['gt_smiles'];d.setdefault(g,{'gt_smiles':g,'c_nmr_peaks':r.get('c_nmr_peaks') or [],'h_nmr_peaks':r.get('h_nmr_peaks') or [],'nmr_solvent':r.get('nmr_solvent'),'nmr_frequency':r.get('nmr_frequency')})
 return list(d.values())
def positional(m):
 out=[];rings=[set(r) for r in m.GetRingInfo().AtomRings()]
 for b in m.GetBonds():
  if b.IsInRing() or b.GetBondType()!=Chem.BondType.SINGLE:continue
  for anchor,ext in [(b.GetBeginAtom(),b.GetEndAtom()),(b.GetEndAtom(),b.GetBeginAtom())]:
   if not anchor.GetIsAromatic() or ext.GetIdx() in {x for r in rings if anchor.GetIdx() in r for x in r}:continue
   ring=next((r for r in rings if anchor.GetIdx() in r),None)
   if not ring:continue
   for t in sorted(ring):
    at=m.GetAtomWithIdx(t)
    if t==anchor.GetIdx() or not at.GetIsAromatic() or at.GetAtomicNum()!=6 or at.GetTotalNumHs()<1:continue
    rw=Chem.RWMol(m)
    try:
     rw.RemoveBond(anchor.GetIdx(),ext.GetIdx());rw.AddBond(t,ext.GetIdx(),Chem.BondType.SINGLE);n=rw.GetMol();Chem.SanitizeMol(n);s=canon(n)
     if s!=canon(m):out.append(s)
    except:pass
 return sorted(set(out))
def enumerate_structural():
 pools=collections.defaultdict(list);seen=set()
 for p in parents():
  m=Chem.MolFromSmiles(p['gt_smiles'])
  if not m or not nof(m) or not p['c_nmr_peaks'] or not p['h_nmr_peaks']:continue
  g=canon(m)
  for a in m.GetAtoms():
   typ=None;s=None
   if a.GetAtomicNum() in (17,35):typ='halogen_swap';subtype='Cl_to_Br' if a.GetAtomicNum()==17 else 'Br_to_Cl';s=swap(m,a.GetIdx(),35 if a.GetAtomicNum()==17 else 17)
   elif a.IsInRing() and a.GetIsAromatic() and a.GetAtomicNum() in (8,16):typ='ring_chalcogen_swap';subtype='ring_O_to_S' if a.GetAtomicNum()==8 else 'ring_S_to_O';s=swap(m,a.GetIdx(),16 if a.GetAtomicNum()==8 else 8)
   if s and (typ,g,s) not in seen:seen.add((typ,g,s));pools[typ].append({**p,'candidate_smiles':s,'family':typ,'perturbation_subtype':subtype})
  for s in positional(m):
   if ('positional_isomer',g,s) not in seen:seen.add(('positional_isomer',g,s));pools['positional_isomer'].append({**p,'candidate_smiles':s,'family':'positional_isomer','perturbation_subtype':'aromatic_substituent_shift'})
 # Mine local constitutional connectivity isomers from the full same-formula NMRexp pool.
 for line in open('/fs_mol/cxq/SpecMaster_jp/negative_output_1000/matching_1000_matching.jsonl'):
  r=json.loads(line)
  if r.get('negative_type')!='neg/same_formula':continue
  gm=Chem.MolFromSmiles(r['gt_smiles']);cm=Chem.MolFromSmiles(r['candidate_smiles'])
  if not gm or not cm or not nof(gm) or not nof(cm) or not r.get('c_nmr_peaks') or not r.get('h_nmr_peaks'):continue
  if Chem.MolToSmiles(gm,isomericSmiles=False)==Chem.MolToSmiles(cm,isomericSmiles=False):continue
  t=Chem.DataStructs.TanimotoSimilarity(FP.GetFingerprint(gm),FP.GetFingerprint(cm))
  if t>=.30:pools['connectivity_isomer'].append({'gt_smiles':r['gt_smiles'],'candidate_smiles':r['candidate_smiles'],'family':'connectivity_isomer','perturbation_subtype':'mined_same_formula_attachment_isomer','c_nmr_peaks':r['c_nmr_peaks'],'h_nmr_peaks':r['h_nmr_peaks'],'nmr_solvent':r.get('nmr_solvent'),'nmr_frequency':r.get('nmr_frequency')})
 # structural prefilter/rank/cap
 for k in pools:
  good=[]
  for x in pools[k]:
   a=Chem.MolFromSmiles(x['gt_smiles']);b=Chem.MolFromSmiles(x['candidate_smiles'])
   if not a or not b or canon(a)==canon(b) or a.GetNumAtoms()==0:continue
   if sum(z.GetAtomicNum()==6 for z in a.GetAtoms())!=sum(z.GetAtomicNum()==6 for z in b.GetAtoms()) or hcount(a)!=hcount(b):continue
   if x['family'] in ('positional_isomer','connectivity_isomer') and symmetry_signature(a)!=symmetry_signature(b):continue
   coverage=mcs_coverage(a,b) if x['family']=='connectivity_isomer' else 1.0
   if x['family']=='connectivity_isomer' and coverage<.60:continue
   x['mcs_atom_coverage']=round(coverage,4)
   t=Chem.DataStructs.TanimotoSimilarity(FP.GetFingerprint(a),FP.GetFingerprint(b));x['tanimoto']=t
   min_t=.30 if x['family']=='connectivity_isomer' else .45
   if t>=min_t:good.append(x)
  pools[k]=sorted(good,key=lambda x:(-x['tanimoto'],x['gt_smiles'],x['candidate_smiles']))[:500]
 return pools
def pred_groups(p):
 if not p or not p.get('ok') or p.get('C_groups') is None or p.get('H_groups') is None:return None
 return len(p['C_groups']),len(p['H_groups']),sorted(len(g) for g in p['H_groups'])
def assign(pred,exp,nuc):
 ev=[float(p.get('delta (ppm)') if nuc=='C' else p.get('centroid',p.get('delta'))) for p in exp];pv=pred['c_peaks' if nuc=='C' else 'h_peaks'];cost=[[abs(a-b) for b in pv] for a in ev];r,c=linear_sum_assignment(cost);res=[cost[i][j] for i,j in zip(r,c)];return {'mean':sum(res)/len(res),'max':max(res),'large':sum(x>(8 if nuc=='C' else .35) for x in res)}
def fmt_nmr(x):
 f=x.get('nmr_frequency') or 'reported';s=x.get('nmr_solvent') or 'unknown';cs=', '.join(f"{p.get('delta (ppm)'):g}" for p in x['c_nmr_peaks']);hs=[]
 for p in x['h_nmr_peaks']:
  d=p.get('centroid',p.get('delta'));cat=p.get('category') or 'm';j=str(p.get('j_values') or '').strip('_ ');hs.append(f"{d:g} ({cat}{', J='+j.replace('_', ', ')+' Hz' if j else ''}, {p.get('nH')}H)")
 return f"Reported NMR metadata: frequency={f}, solvent={s}\n\n13C NMR: {cs}\n\n1H NMR: "+', '.join(hs)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',default='data/phase1_expert_perturbations_100.json');ap.add_argument('--review-json',default='docs/phase1_expert_perturbations_review30.json');ap.add_argument('--review-md',default='docs/phase1_expert_perturbations_review30.md');a=ap.parse_args();pools=enumerate_structural();print('structural', {k:len(v) for k,v in pools.items()},flush=True)
 cache=json.load(open('/fs_mol/cxq/Xmaster_AgentRL/Open-AgentRL/recipe/nmr_matching_agent/prediction_cache.json'));cm={_canonical(k):v for k,v in cache.items()};phase_cache=Path('data/phase1_prediction_cache.json')
 if phase_cache.exists():cm.update(json.loads(phase_cache.read_text()))
 smiles=sorted({_canonical(x[y]) for v in pools.values() for x in v for y in ('gt_smiles','candidate_smiles')});missing=[s for s in smiles if s not in cm];print('predict_missing',len(missing),flush=True)
 for s,p in zip(missing,predict_nmr_batch(missing,batch_size=16)):cm[s]=p
 phase_cache.write_text(json.dumps({s:cm[s] for s in smiles if s in cm},ensure_ascii=False)+'\n')
 eligible=collections.defaultdict(list)
 for fam,xs in pools.items():
  for x in xs:
   gp=cm.get(_canonical(x['gt_smiles']));cp=cm.get(_canonical(x['candidate_smiles']));gg=pred_groups(gp);cg=pred_groups(cp)
   if not gg or gg!=cg:continue
   ex=x['h_nmr_peaks'];exp_ivec=sorted(p['nH'] for p in ex if p.get('nH') is not None)
   gs=spectral_similarity(gp['c_peaks'],gp['h_peaks'],x['c_nmr_peaks'],x['h_nmr_peaks']);cs=spectral_similarity(cp['c_peaks'],cp['h_peaks'],x['c_nmr_peaks'],x['h_nmr_peaks'])
   ca=assign(cp,x['c_nmr_peaks'],'C');ha=assign(cp,x['h_nmr_peaks'],'H')
   if gs<.55 or not (.25<=cs<=.95) or gs-cs<.015 or not (ca['large']>=1 or ha['large']>=1):continue
   rec={k:v for k,v in x.items() if k!='tanimoto'};rec.update({'negative_type':f"neg/expert_perturbation/{fam}/{x['perturbation_subtype']}",'label':0,'tanimoto':round(x['tanimoto'],4),'gt_similarity':round(gs,4),'candidate_similarity':round(cs,4),'similarity_margin':round(gs-cs,4),'predicted_c_signal_count':gg[0],'predicted_h_signal_count':gg[1],'h_integration_vector':gg[2],'experimental_h_integration_vector':exp_ivec,'candidate_c_assignment':ca,'candidate_h_assignment':ha,'tool_results':{'candidate_gt_C_atoms_match':True,'candidate_gt_H_atoms_match':True,'predicted_signal_counts_match':True,'predicted_integration_vector_match':True}});eligible[fam].append(rec)
  eligible[fam].sort(key=lambda z:(-(z['similarity_margin']+.03*z['tanimoto']),z['gt_smiles'],z['candidate_smiles']))
 print('eligible',{k:len(v) for k,v in eligible.items()},flush=True)
 selected=[];usedgt=collections.Counter()
 for fam,q in QUOTA.items():
  cand=[x for x in eligible[fam] if usedgt[x['gt_smiles']]<2]
  take=cand[:q];selected+=take
  for x in take:usedgt[x['gt_smiles']]+=1
 if len(selected)<100:
  rest=sorted([x for fam in QUOTA for x in eligible[fam] if x not in selected],key=lambda z:-(z['similarity_margin']+.03*z['tanimoto']))
  selected+=rest[:100-len(selected)]
 if len(selected)<100:raise RuntimeError(f'only {len(selected)} accepted')
 selected=selected[:100]
 for i,x in enumerate(selected,1):x['case_id']=f'phase1_{i:03d}'
 Path(a.out).write_text(json.dumps(selected,ensure_ascii=False,indent=2)+'\n')
 # diverse review: quotas 8/8/9/5, round-robin GT
 rq={'halogen_swap':8,'ring_chalcogen_swap':8,'positional_isomer':9,'connectivity_isomer':5};review=[]
 for fam,q in rq.items():
  ordered=sorted([x for x in selected if x['family']==fam],key=(lambda z:(-z['tanimoto'],-z['candidate_similarity'],z['gt_smiles']) if fam=='connectivity_isomer' else (-z['similarity_margin'],-z['tanimoto'],z['gt_smiles'])))
  pick=[];seen_gt=set()
  for x in ordered:
   if x['gt_smiles'] not in seen_gt:pick.append(x);seen_gt.add(x['gt_smiles'])
   if len(pick)==q:break
  if len(pick)<q:pick += [x for x in ordered if x not in pick][:q-len(pick)]
  review+=pick
 clean=[];md=['# Phase 1 等信号数高价值微扰：专家核验30条','', '- 禁止含F；GT与负样本的C/H原子数、预测C/H信号数和预测H组大小向量均保持一致','- 注：预测H组大小向量表示预测等价氢分组；实验multiplet可能聚合多个预测组，另列核磁原文供专家判断。','']
 for i,x in enumerate(review,1):
  y=dict(x);y['nmr_text']=fmt_nmr(x);clean.append(y);md += [f"## {i:02d}. {x['negative_type']}",'',f"- **样本类型**：`{x['negative_type']}`",f"- **正确结构（SMILES）**：`{x['gt_smiles']}`",f"- **负样本结构（SMILES）**：`{x['candidate_smiles']}`",f"- **筛选证据**：C/H signals `{x['predicted_c_signal_count']}/{x['predicted_h_signal_count']}`；预测H组大小向量 `{x['h_integration_vector']}`；实验积分向量 `{x['experimental_h_integration_vector']}`；GT sim `{x['gt_similarity']}`；negative sim `{x['candidate_similarity']}`；margin `{x['similarity_margin']}`",'- **核磁文本**：','',y['nmr_text'],'']
 Path(a.review_json).write_text(json.dumps(clean,ensure_ascii=False,indent=2)+'\n');Path(a.review_md).write_text('\n'.join(md));print('selected',collections.Counter(x['family'] for x in selected),'review',collections.Counter(x['family'] for x in review))
if __name__=='__main__':main()
