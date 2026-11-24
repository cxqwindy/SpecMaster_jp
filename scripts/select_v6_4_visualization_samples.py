#!/usr/bin/env python3
"""Select 30 high-quality/diverse v6.4 train negatives per requested class."""
from __future__ import annotations
import argparse, collections, json, math, sys
from pathlib import Path
from typing import Any
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

sys.path.insert(0, "/fs_mol/cxq/SpecMaster_jp/scripts")
from prepare_v6_4_sft_data import DEFAULT_INPUT, read_jsonl, sha256, split_grouped_stratified

FPGEN=GetMorganGenerator(radius=2,fpSize=2048)
GROUPS=("random","same_formula","similar_structure","perturbation")

def group_of(nt):
 if nt=="neg/random": return "random"
 if nt=="neg/same_formula": return "same_formula"
 if nt=="neg/similar_structure": return "similar_structure"
 if nt and nt.startswith("neg/perturbation/"): return "perturbation"
 return None

def atoms(m):
 c=collections.Counter(a.GetSymbol() for a in m.GetAtoms());c['H']=sum(1 for a in Chem.AddHs(m).GetAtoms() if a.GetSymbol()=='H');return c

def obs_map(row):
 out={}
 for m in row['messages']:
  if m.get('role')!='tool':continue
  try:o=json.loads(m['content'])
  except:continue
  if o.get('element'):out['count_'+o['element']]=o
  elif 'similarity' in o:out['spectral']=o
 return out

def finite(x):
 try:return math.isfinite(float(x))
 except:return False

def features(row,idx):
 meta=row['meta'];g=group_of(meta.get('negative_type'))
 if not g or meta.get('label')!=0:return None
 try:u=json.loads(row['messages'][1]['content'])
 except:return None
 cand=Chem.MolFromSmiles(meta.get('candidate_smiles',''));gt=Chem.MolFromSmiles(meta.get('gt_smiles',''))
 if not cand or not gt:return None
 cc=Chem.MolToSmiles(cand,True);gc=Chem.MolToSmiles(gt,True)
 if cc==gc or '.' in cc or '.' in gc:return None
 cp=u.get('c_nmr_peaks') or [];hp=u.get('h_nmr_peaks') or []
 if not cp or not hp or not u.get('nmr_solvent') or not u.get('nmr_frequency'):return None
 solvent=str(u.get('nmr_solvent')).strip().lower()
 if solvent in {'not_known','unknown','mixed','n/a','na','none','unspecified'}:return None
 cvals=[p.get('delta (ppm)',p.get('delta')) for p in cp];hvals=[p.get('centroid',p.get('delta')) for p in hp]
 if not all(finite(x) and -10<=float(x)<=250 for x in cvals):return None
 if not all(finite(x) and -2<=float(x)<=20 and isinstance(p.get('nH'),int) and p['nH']>0 for x,p in zip(hvals,hp)):return None
 ga,ca=atoms(gt),atoms(cand);cov=len(cp)/max(ga['C'],1)
 if not .60<=cov<=1.0 or sum(p['nH'] for p in hp)!=ga['H']:return None
 om=obs_map(row);s=om.get('spectral',{})
 if not s.get('ok') or not all(finite(s.get(k)) and 0<=float(s[k])<=1 for k in ('similarity','c_sim','h_sim')):return None
 if s.get('n_exp_c')!=len(cp) or s.get('n_exp_h')!=len(hp):return None
 tan=Chem.DataStructs.TanimotoSimilarity(FPGEN.GetFingerprint(cand),FPGEN.GetFingerprint(gt));sim=float(s['similarity']);sub=min(float(s['c_sim']),float(s['h_sim']))
 formula_c=rdMolDescriptors.CalcMolFormula(cand);formula_g=rdMolDescriptors.CalcMolFormula(gt)
 delta={e:ca[e]-ga[e] for e in sorted(set(ca)|set(ga)) if ca[e]!=ga[e]}
 heavy_ratio=cand.GetNumHeavyAtoms()/max(gt.GetNumHeavyAtoms(),1)
 hobs=om.get('count_H',{});eligible=False
 if g=='random':eligible=formula_c!=formula_g and .05<=tan<=.20 and .05<=sim<=.60 and .5<=heavy_ratio<=2.0
 elif g=='same_formula':eligible=formula_c==formula_g and .15<=tan<=.40 and sim<=.65 and sub<=.60
 elif g=='similar_structure':eligible=formula_c!=formula_g and .60<=tan<=.85 and sim<=.85
 elif g=='perturbation':eligible=meta.get('_class')=='insert_ch2' and delta=={'C':1,'H':2} and .60<=tan<=.92 and sim<=.90 and hobs.get('match') is False
 if not eligible:return None
 scaffold=Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(gt),isomericSmiles=False,canonical=True)
 # Quality balances complete C coverage, class hardness, and decisive spectral evidence.
 hardness={'random':1-abs(tan-.12)/.08,'same_formula':(tan-.15)/.25,'similar_structure':1-abs(tan-.75)/.15,'perturbation':(tan-.60)/.32}[g]
 score=35*min(1,cov)+25*max(0,min(1,hardness))+30*(1-sim)+10*(sub<.60)
 return {'source_index':idx,'group':g,'negative_type':meta['negative_type'],'class':meta.get('_class'),'correct_smiles':meta['gt_smiles'],'negative_smiles':meta['candidate_smiles'],'correct_canonical':gc,'negative_canonical':cc,'correct_formula':formula_g,'negative_formula':formula_c,'formula_delta':delta,'heavy_atom_ratio':round(heavy_ratio,4),'tanimoto':round(tan,4),'similarity':round(sim,4),'c_sim':round(float(s['c_sim']),4),'h_sim':round(float(s['h_sim']),4),'c_coverage':round(cov,4),'scaffold':scaffold,'quality_score':round(score,4),'nmr':u,'gt_fp':FPGEN.GetFingerprint(gt)}

def select(pool,n=30):
 chosen=[];used_gt=set();used_c=set();sc=collections.Counter()
 while len(chosen)<n:
  best=None;bestv=None
  for x in pool:
   if x in chosen or x['negative_canonical'] in used_c:continue
   if x['correct_canonical'] in used_gt and len(pool)>n:continue
   novelty=1.0 if not chosen else 1-max(Chem.DataStructs.TanimotoSimilarity(x['gt_fp'],y['gt_fp']) for y in chosen)
   scaffold_bonus=1 if sc[x['scaffold']]==0 else 0
   val=.70*x['quality_score']+20*novelty+10*scaffold_bonus
   tie=(val,x['quality_score'],x['correct_canonical'],x['negative_canonical'])
   if bestv is None or tie>bestv:best,bestv=x,tie
  if best is None:raise RuntimeError(f"cannot select {n}")
  chosen.append(best);used_gt.add(best['correct_canonical']);used_c.add(best['negative_canonical']);sc[best['scaffold']]+=1
 return chosen

def fmt_num(x):return f"{float(x):g}"
def nmr_text(u):
 solvent=u.get('nmr_solvent') or 'unspecified';freq=u.get('nmr_frequency') or 'unspecified'
 cs=', '.join(fmt_num(p.get('delta (ppm)',p.get('delta'))) for p in u['c_nmr_peaks'])
 hs=[]
 for p in u['h_nmr_peaks']:
  d=p.get('centroid',p.get('delta'));cat=p.get('category') or 'm';j=str(p.get('j_values') or '').strip('_ ');inside=cat+(f", J={j.replace('_', ', ')} Hz" if j else '')+f", {p['nH']}H";hs.append(f"{fmt_num(d)} ({inside})")
 return f"Reported NMR metadata: frequency={freq}, solvent={solvent}\n\n13C NMR: {cs}\n\n1H NMR: "+', '.join(hs)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out-md',default='/fs_mol/cxq/SpecMaster_jp/docs/v6_4_negative_samples_visualization.md');ap.add_argument('--out-json',default='/fs_mol/cxq/SpecMaster_jp/docs/v6_4_negative_samples_visualization.json');a=ap.parse_args()
 src=read_jsonl(DEFAULT_INPUT);train,val,split=split_grouped_stratified(src,.03,42);pools={g:[] for g in GROUPS}
 for i,r in enumerate(train):
  x=features(r,i)
  if x:pools[x['group']].append(x)
 selected={g:select(pools[g]) for g in GROUPS}
 manifest={'source':str(DEFAULT_INPUT),'source_sha256':sha256(DEFAULT_INPUT),'split_seed':42,'val_frac':.03,'actual_train_rows':len(train),'eligible_counts':{g:len(pools[g]) for g in GROUPS},'selection_method':'hard quality gates + greedy quality/MaxMin/scaffold diversity','groups':{}}
 md=['# v6.4 实际训练集：四类高质量负样本可视化','',f'- 实际训练源：`{DEFAULT_INPUT}`',f'- SHA-256：`{manifest["source_sha256"]}`','- split：与 v6.4 一致（seed=42, val_frac=0.03），仅从 4009 条 train_source 选择','- 每类 30 条；微扰仅含 `neg/perturbation/alkyl_chain_shift/insert_ch2`，不含 hard_nmr','']
 names={'random':'随机负样本','same_formula':'同分子式负样本','similar_structure':'相似结构负样本','perturbation':'微扰负样本'}
 for g in GROUPS:
  md += [f'## {names[g]}（30条）','']
  clean=[]
  for k,x in enumerate(selected[g],1):
   y={kk:vv for kk,vv in x.items() if kk not in ('gt_fp','nmr')};y['nmr_text']=nmr_text(x['nmr']);clean.append(y)
   md += [f'### {g}-{k:02d}',f'- **样本类型**：`{x["negative_type"]}`',f'- **正确结构（SMILES）**：`{x["correct_smiles"]}`',f'- **负样本结构（SMILES）**：`{x["negative_smiles"]}`',f'- **质量指标**：formula `{x["correct_formula"]}` → `{x["negative_formula"]}`；Tanimoto `{x["tanimoto"]}`；spectral `{x["similarity"]}`（C `{x["c_sim"]}` / H `{x["h_sim"]}`）', '- **核磁文本**：','',nmr_text(x['nmr']),'']
  manifest['groups'][g]=clean
 Path(a.out_md).write_text('\n'.join(md),encoding='utf-8');Path(a.out_json).write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print(json.dumps({'eligible':manifest['eligible_counts'],'selected':{g:len(v) for g,v in selected.items()},'md':a.out_md,'json':a.out_json},indent=2))
if __name__=='__main__':main()
