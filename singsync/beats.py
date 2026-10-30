"""Local audio analysis and beat-aware cut snapping for the WVRPS MV engine."""
from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks, stft


def _decode(path: str | Path, sr: int) -> np.ndarray:
    p = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1",
                        "-ar", str(sr), "-f", "f32le", "-"],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    return np.frombuffer(p.stdout, dtype="<f4").astype(np.float32, copy=False)


def _frames(x: np.ndarray, size: int, hop: int) -> np.ndarray:
    if len(x) < size:
        x = np.pad(x, (0, size-len(x)))
    n = 1 + (len(x)-size)//hop
    return np.lib.stride_tricks.sliding_window_view(x, size)[::hop][:n]


def _norm01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if not x.size:
        return x
    lo, hi = np.percentile(x, 5), np.percentile(x, 95)
    if hi <= lo + 1e-12:
        return np.zeros_like(x)
    return np.clip((x-lo)/(hi-lo), 0, 1)


def _onset_and_energy(x: np.ndarray, sr: int, duration: float):
    hop = max(1, round(sr*0.01)); size = max(64, round(sr*0.046))
    fr = _frames(x, size, hop)
    rms = np.sqrt(np.mean(fr.astype(float)**2, axis=1))
    win = np.hanning(size)
    spec = np.abs(np.fft.rfft(fr*win, axis=1))
    flux = np.maximum(np.diff(np.log1p(spec), axis=0), 0).sum(axis=1)
    flux = np.r_[0., flux]
    onset = _norm01(flux)
    # Blend positive spectral change with RMS attack to retain percussive click trains.
    onset = _norm01(0.85*onset + 0.15*_norm01(np.maximum(np.diff(rms, prepend=rms[0]), 0)))
    env_times = np.arange(len(onset))*hop/sr
    ehop = 0.5
    n = max(1, int(np.ceil(duration/ehop)))
    rms_curve=[]; flux_curve=[]; centers=[]
    for i in range(n):
        a,b=i*ehop,(i+1)*ehop
        sel=(env_times>=a)&(env_times<min(b,duration))
        rms_curve.append(float(np.sqrt(np.mean(rms[sel]**2))) if sel.any() else 0.)
        flux_curve.append(float(np.mean(flux[sel])) if sel.any() else 0.)
        centers.append(min(duration, (a+b)/2))
    energy=_norm01(.65*_norm01(np.array(rms_curve))+.35*_norm01(np.array(flux_curve)))
    return onset, env_times, np.asarray(rms_curve), np.asarray(flux_curve), energy, np.asarray(centers)


def _track_beats(onset: np.ndarray, times: np.ndarray, duration: float):
    # Search a fine tempo grid against a phase-invariant pulse train. A Gaussian
    # tolerance makes timing robust to frame quantization and slightly soft attacks.
    best=(-1., 120., 0.)
    for bpm in np.arange(60., 200.01, .1):
        period=60./bpm
        phases=np.arange(0, min(period, .20), .01)
        for phase in phases:
            ts=np.arange(phase, duration, period)
            if not len(ts): continue
            dt=times[1]-times[0]
            idx=np.rint(ts/dt).astype(int)
            # Local maximum tolerates attack envelope delay and frame quantization.
            scores=[]
            for ix in idx:
                lo=max(0,ix-4); hi=min(len(onset),ix+5)
                if lo<hi: scores.append(float(np.max(onset[lo:hi])))
            score=float(np.mean(scores)) if scores else 0.
            # Mild preference against implausibly fast grids explaining every other hit.
            score *= 1.-.0005*(bpm-120.)**2/100.
            if score>best[0]: best=(score,bpm,phase)
    confidence=float(np.clip(best[0],0,1))
    _,bpm,phase=best
    period=60./bpm
    beats=np.arange(phase, duration, period)
    if len(beats)<2:
        return float(bpm), confidence, beats, beats, 0.
    # Phase is selected by mean onset accent on every fourth beat.
    vals=[]
    dt=times[1]-times[0]
    bi=np.clip(np.rint(beats/dt).astype(int),0,len(onset)-1)
    accents=[]
    for ix in bi:
        lo=max(0,ix-4); hi=min(len(onset),ix+5)
        accents.append(float(np.max(onset[lo:hi])) if lo<hi else 0.)
    accents=np.asarray(accents)
    for q in range(4):
        vals.append(float(np.mean(accents[q::4])) if len(bi[q::4]) else 0.)
    # The tracker phase is anchored at the first pulse; downbeat phase is then
    # the strongest four-beat accent class. Include the initial attack, whose
    # STFT flux can be clipped at file start, as the phase origin when present.
    q=int(np.argmax(vals))
    if beats[0] < .08 and vals[0] >= max(vals)*.72:
        q=0
    down=beats[q::4]
    sortedv=sorted(vals, reverse=True)
    dbconf=float((sortedv[0]-sortedv[1])/(sortedv[0]+1e-9))
    return float(bpm), confidence, beats, down, float(np.clip(dbconf,0,1))


def _sections(x: np.ndarray, sr: int, energy: np.ndarray, centers: np.ndarray,
              downbeats: np.ndarray):
    # Short-time spectral descriptors aggregated to 0.5s bins. Log-frequency
    # band energy is a compact chroma/MFCC-like timbral representation.
    _, tt, z=stft(x, fs=sr, nperseg=2048, noverlap=1536, boundary=None)
    mag=np.abs(z)
    edges=np.geomspace(max(30,sr/2048), sr/2, 25)
    bands=[]
    for a,b in zip(edges[:-1],edges[1:]):
        lo=max(0,int(a*2048/sr)); hi=max(lo+1,int(b*2048/sr))
        bands.append(np.mean(mag[lo:hi]**2,axis=0))
    feat=np.log1p(np.asarray(bands).T)
    # Aggregate to same half-second grid.
    f=[]
    for c in centers:
        m=(tt>=c-.25)&(tt<c+.25)
        f.append(np.mean(feat[m],axis=0) if m.any() else np.zeros(feat.shape[1]))
    f=np.asarray(f)
    f=(f-f.mean(axis=1,keepdims=True))/(f.std(axis=1,keepdims=True)+1e-8)
    f=f/(np.linalg.norm(f,axis=1,keepdims=True)+1e-8)
    sim=f@f.T
    # Checkerboard novelty kernel; transitions create a positive score where
    # cross-boundary similarity is lower than within-region similarity.
    win=4; novelty=np.zeros(len(f))
    for i in range(win,len(f)-win):
        left=sim[i-win:i,i-win:i]; right=sim[i:i+win,i:i+win]
        cross=sim[i-win:i,i:i+win]
        novelty[i]=(left.mean()+right.mean())/2-cross.mean()
    peaks,_=find_peaks(novelty, distance=12, prominence=max(.025,float(np.std(novelty)*.8)) )
    # Keep the strongest boundaries, limiting fragmentation and dropping edges.
    peaks=sorted(peaks,key=lambda i:novelty[i],reverse=True)[:min(8,max(1,len(energy)//18))]
    bounds=[0]+sorted(int(i) for i in peaks)+[len(energy)]
    # Add a strong broad energy transition when novelty misses a clear drop.
    jumps=np.diff(energy,prepend=energy[0])
    j=int(np.argmax(jumps)) if len(jumps) else 0
    if j>1 and jumps[j]>.20 and all(abs(j-b)>1 for b in bounds):
        bounds=sorted(set(bounds+[j]))
    n=len(bounds)-1; sections=[]
    for k,(a,b) in enumerate(zip(bounds,bounds[1:])):
        if b<=a: continue
        em=float(np.mean(energy[a:b])); pos=float((centers[a]+centers[b-1])/(2*max(centers[-1]+.25,1e-9)))
        prev=float(np.mean(energy[max(0,a-2):a])) if a else em
        jump=em-prev
        if a and jump>.18 and em>=.55 and np.any(np.abs(downbeats-centers[a])<.3): label='drop'
        elif pos<.10: label='intro'
        elif pos>.90: label='outro'
        elif jump>.12: label='build'
        elif em<.28: label='breakdown'
        else: label='verse'
        sections.append({'start':round(float(centers[a]-.25 if a else 0),3),
                         'end':round(float(min(centers[b-1]+.25,centers[-1]+.25)),3),
                         'energy_mean':round(em,4),'label':label})
    return sections


def analyse(path: str | Path, sr: int = 22050) -> dict:
    x=_decode(path,sr)
    if not len(x): raise ValueError('decoded audio is empty')
    duration=len(x)/sr
    onset,times,rms,flux,energy,centers=_onset_and_energy(x,sr,duration)
    bpm,conf,beats,down,dbconf=_track_beats(onset,times,duration)
    sections=_sections(x,sr,energy,centers,down)
    # Drops are section starts already classified as drops, with prior level.
    drops=[]
    for s in sections:
        if s['label']=='drop': drops.append({'time':s['start'],'energy_jump':round(float(s['energy_mean']-next((p['energy_mean'] for p in sections if p['end']<=s['start']),0)),4)})
    return {'duration':round(duration,4),'sample_rate':sr,'tempo_bpm':round(bpm,3),
            'tempo_confidence':round(conf,4),'beats':[round(float(v),4) for v in beats],
            'downbeats':[round(float(v),4) for v in down],
            'downbeat_confidence':round(dbconf,4),
            'onset_strength':{'hop_s':round(float(times[1]-times[0]),5),
                'mean':round(float(np.mean(onset)),5),'peak':round(float(np.max(onset)),5),
                'values':[round(float(v),4) for v in onset]},
            'energy_curve':[{'time':round(float(t),3),'rms':round(float(r),5),
                'spectral_flux':round(float(f),5),'energy':round(float(e),4)}
                for t,r,f,e in zip(centers,rms,flux,energy)],
            'sections':sections,'drops':drops}


def snap_cuts_to_beats(cutplan: dict, analysis: dict, words: list,
                       tol_s: float=.25, prefer: str='downbeat'):
    out=copy.deepcopy(cutplan); log=[]
    candidates=analysis.get('downbeats' if prefer=='downbeat' else 'beats',[])
    duration=float(out.get('duration',analysis.get('duration',0)))
    cuts=out.get('cuts',[])
    for i in range(len(cuts)-1):
        left,right=cuts[i],cuts[i+1]
        old=float(left['owns'][1]); gap_lo=max(float(left['owns'][0]),float(right['owns'][0]))
        if any(float(w['start']) < old < float(w['end']) for w in words):
            continue
        # Adjacent tile boundaries have a point position; actual safe interval is
        # bounded by nearest word end/start straddling the boundary.
        prev_ends=[float(w['end']) for w in words if float(w['end'])<=old+1e-9]
        next_starts=[float(w['start']) for w in words if float(w['start'])>=old-1e-9]
        lo=max(gap_lo,max(prev_ends) if prev_ends else 0.)
        hi=min(float(right['owns'][1]),min(next_starts) if next_starts else duration)
        near=min(candidates,key=lambda t:abs(float(t)-old),default=None)
        if near is None or abs(float(near)-old)>tol_s or float(near)<lo-1e-9 or float(near)>hi+1e-9:
            continue
        new=float(near)
        if abs(new-old)<1e-9: continue
        left['owns'][1]=new; right['owns'][0]=new
        log.append({'left_cut':left.get('index',i),'right_cut':right.get('index',i+1),
                    'from':round(old,4),'to':round(new,4),'delta':round(new-old,4)})
    # Owned spans still tile exactly; audio spans remain untouched, preserving shared-word wings.
    return out,log


def vibe_brief(analysis: dict, cut: dict) -> str:
    span=cut.get('owns',[0,0]); mid=(float(span[0])+float(span[1]))/2
    section=next((s for s in analysis.get('sections',[]) if s['start']<=mid<s['end']),None)
    energy=next((e['energy'] for e in analysis.get('energy_curve',[]) if e['time']>=mid),0.)
    curve=analysis.get('energy_curve',[]); idx=next((i for i,e in enumerate(curve) if e['time']>=mid),0)
    rising=idx>0 and curve[idx]['energy']>curve[idx-1]['energy']+.02
    label=section['label'] if section else 'unknown'
    t=float(span[0]); db=analysis.get('downbeats',[]); nearest=min(db,key=lambda v:abs(v-t),default=None)
    if nearest is not None and abs(nearest-t)<.08:
        beat_idx=min(range(len(analysis.get('beats',[]))),key=lambda i:abs(analysis['beats'][i]-nearest),default=0)
        bar=beat_idx//4+1; landing=f'cut lands on downbeat 1 of bar {bar}'
    else: landing='cut does not land on a downbeat'
    return f"section={label}, energy={float(energy):.2f}, {'rising' if rising else 'steady/falling'}; {landing}"


def main(argv=None):
    ap=argparse.ArgumentParser(); sub=ap.add_subparsers(dest='cmd',required=True)
    a=sub.add_parser('analyse'); a.add_argument('audio'); a.add_argument('-o','--out',required=True)
    s=sub.add_parser('snap'); s.add_argument('cutplan'); s.add_argument('analysis'); s.add_argument('alignment'); s.add_argument('-o','--out',required=True)
    ns=ap.parse_args(argv)
    if ns.cmd=='analyse': result=analyse(ns.audio)
    else:
        cp=json.loads(Path(ns.cutplan).read_text(encoding='utf-8')); an=json.loads(Path(ns.analysis).read_text(encoding='utf-8')); al=json.loads(Path(ns.alignment).read_text(encoding='utf-8'))
        result,log=snap_cuts_to_beats(cp,an,al.get('words',[])); result['beat_snap_log']=log
    Path(ns.out).write_text(json.dumps(result,indent=2),encoding='utf-8')
    return 0


if __name__=='__main__': sys.exit(main())
