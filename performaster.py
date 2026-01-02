from collections import defaultdict
import os
import threading
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import mido
import math
import logging
import tempfile
logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)
DEFAULTS ={
    "PITCH_LOW":21,
    "PITCH_HIGH":108,
    "TIME_BINS":500,
    "ACCENT_THRESHOLD":20,
    "LOW_REGISTER_FRACTION":0.35,
    "HIGH_REGISTER_FRACTION":0.35,
    "CACHE_DIR":os.path.expanduser("~/.perfinterp/cache")}
try:
    os.makedirs(DEFAULTS["CACHE_DIR"], exist_ok=True)
except Exception:
    DEFAULTS["CACHE_DIR"]=tempfile.gettempdir()
PITCH_LOW = DEFAULTS["PITCH_LOW"]
PITCH_HIGH = DEFAULTS["PITCH_HIGH"]
PITCH_RANGE = PITCH_HIGH-PITCH_LOW+1
TIME_BINS = DEFAULTS["TIME_BINS"]
PITCH_BIN = PITCH_RANGE
ACCENT_THRESHOLD = DEFAULTS["ACCENT_THRESHOLD"]
LOW_REGISTER_FRACTION = DEFAULTS["LOW_REGISTER_FRACTION"]
HIGH_REGISTER_FRACTION = DEFAULTS["HIGH_REGISTER_FRACTION"]
CACHE_DIR = DEFAULTS["CACHE_DIR"]
NOTE_NAMES= ["C", "C#/Db", "D", "D#/Eb", "E", "F", "F#/Gb", "G", "G#/Ab", "A", "A#/Bb", "B"]
def midi_to_note_name(m):
    try:
        m= int(m)
    except Exception:
        return str(m)
    octave=(m//12)-1
    name=NOTE_NAMES[m%12]
    return f"{name}{octave}"
def midi_to_events(mid:mido.MidiFile):
    ticks_per_beat = getattr(mid,"ticks_per_beat", 480)
    flat = []
    for ti, tr in enumerate(mid.tracks):
        abs_ticks = 0
        for msg in tr:
            abs_ticks += msg.time
            flat.append((abs_ticks,msg,ti))
    flat.sort(key=lambda x: x[0])        
    tempo = 500000
    last_tick = 0
    current_seconds = 0.0
    events = []
    for abs_ticks, msg, track_idx in flat:
        delta_ticks = abs_ticks - last_tick
        if delta_ticks < 0:
            delta_ticks = 0
        sec = mido.tick2second(delta_ticks,ticks_per_beat,tempo)
        current_seconds += sec
        last_tick = abs_ticks
        mtype = getattr(msg,"type",None)
        if mtype == "set_tempo":
            tempo = getattr(msg,"tempo",tempo)
            events.append({"time":current_seconds, "type": "set_tempo", "tempo": tempo, "track": track_idx})
        elif mtype in ("note_on", "note_off"):
            events.append({
                "time":current_seconds,
                "type":mtype,
                "note":getattr(msg,"note",None),
                "velocity":getattr(msg,"velocity",0),
                "channel":getattr(msg,"channel",None),
                "track":track_idx})
        elif mtype == "control_change":
            events.append({
                "time":current_seconds,
                "type": "control_change",
                "control":getattr(msg,"control",None),
                "value":getattr(msg,"value",None),
                "channel":getattr(msg,"channel",0),
                "track":track_idx,})
    return events, current_seconds
def extract_beat_times(events, ticks_per_beat):
    tempo= 500000
    last_tick= 0
    current_time=0.0
    beat_times=[]
    beat_count= 0
    ticks_since_last_beat=0
    for e in events:
        if e["type"]=="set_tempo":
            tempo = e["tempo"]
            continue
        abs_time=e["time"]
        delta_time=abs_time - current_time
        ticks_advanced=mido.second2tick(delta_time,ticks_per_beat,tempo)
        ticks_since_last_beat += ticks_advanced
        while ticks_since_last_beat>=ticks_per_beat:
            ticks_since_last_beat -= ticks_per_beat
            beat_count += 1
            beat_times.append((current_time,beat_count))
        current_time=abs_time
    return beat_times
def normalize_note_times_by_beat(notes,beat_times):
    if not notes or not beat_times:
        for n in notes:
            n["t0b"] = n.get("t0n",0.0)
        return notes
    beat_seconds=np.array([t for t, _ in beat_times])
    beat_indices = np.array([b for _, b in beat_times])
    max_beat=beat_indices[-1] if len(beat_indices) else 1.0
    for n in notes:
        t=n["start"]
        idx=np.searchsorted(beat_seconds,t,side="right") - 1
        idx=max(0,min(idx,len(beat_indices) - 1))
        n["t0b"]=beat_indices[idx] / max_beat
    return notes
def events_to_notes_and_pedal(events):
    note_stack = defaultdict(list)
    notes = []
    pedal_events = []
    for e in events:
        etype = e.get("type")
        if etype == "note_on" and e.get("velocity", 0) > 0:
            key = (e.get("note"), e.get("channel"))
            note_stack[key].append({"start": e["time"], "velocity": e.get ("velocity",0), "track": e.get("track")})
        elif etype == "note_off" or (etype == "note_on" and e.get("velocity",0)==0):
            key = (e.get("note"), e.get("channel"))
            if key in note_stack and note_stack[key]:
                startinfo = note_stack[key].pop()
                notes.append({
                    "note": e.get("note"),
                    "start": startinfo["start"],
                    "end": e.get("time"),
                    "velocity": startinfo.get("velocity",0),
                    "channel": e.get("channel"),
                    "track": startinfo["track"]})
            else:
                logger.debug("Unmatched note_off for note %s at time %.3f", e.get("note"), e.get("time"))
        elif etype == "control_change" and e.get("control") == 64:
            pedal_events.append((e.get("time"), e.get("value",0) >= 64))
    leftovers = sum(len(v) for v in note_stack.values())
    if leftovers:
        logger.debug("Leftover unmatched note_on events: %d", leftovers)
    return notes, pedal_events
def normalize_note_times(notes,total_duration):
    if total_duration<= 0:
        total_duration =1.0
    for n in notes:
        n["t0n"] = max(0.0, min(1.0,n["start"] /total_duration))
        n["t1n"]=max(0.0, min(1.0, n["end"]/total_duration))
        n["dur"] =max(0.0, n['end'] - n["start"])
    return notes
def note_density_over_time(notes,bins=TIME_BINS):
    if not notes:
        return np.zeros(bins), np.linspace(0,1,bins)
    starts= [n["t0n"] for n in notes]
    hist, edges =np.histogram(starts,bins=bins,range=(0.0,1.0))
    centers =(edges[:-1] +edges[1:])/2
    return hist, centers
def velocity_profile(notes, bins=TIME_BINS):
    if not notes:
        return np.zeros(bins), np.linspace(0,1,bins)
    times =[(n["t0n"] + n["t1n"])/2 for n in notes]
    vel= [n["velocity"] for n in notes]
    bin_vals= np.zeros(bins)
    bin_counts =np.zeros(bins)
    for t, v in zip(times,vel):
        idx =min(bins - 1,int(t* bins))
        bin_vals[idx]+= v
        bin_counts[idx] +=1
    avg=np.divide(bin_vals,np.maximum(bin_counts,1))
    centers =np.linspace(0,1,bins)
    return avg, centers
def compute_metrics(notes,pedal_events,total_duration, time_bins=TIME_BINS,accent_threshold=ACCENT_THRESHOLD):
    metrics = {}
    if total_duration <=0:
        total_duration= 1.0
    notes_norm =[n.copy() for n in notes]
    normalize_note_times(notes_norm, total_duration)
    segs= 20
    seg_centers =np.linspace(0,1,segs+1)
    seg_ranges= [(seg_centers[i],seg_centers[i+1]) for i in range(segs)]
    dyn_range_per_seg=[]
    for a, b in seg_ranges:
        vs =[n["velocity"] for n in notes if (n["t0n"]>= a and n["t0n"] <b) and n.get("velocity",0)>0]
        if vs:
            dyn_range_per_seg.append(float(max(vs) - min(vs)))
        else:
            dyn_range_per_seg.append(0.0)
    metrics["dyn_range_per_segment"]=dyn_range_per_seg
    all_vels= np.array([n["velocity"] for n in notes if n.get("velocity",0)>0], dtype=float)
    if all_vels.size >= 2 and np.ptp(all_vels)>0:
        all_vels_scaled=(all_vels-all_vels.min()) / np.ptp(all_vels)
    else:
        all_vels_scaled=all_vels
    if all_vels.size>1:
        mu=np.mean(all_vels)
        sigma= np.std(all_vels) + 1e-6
        z_vels= (all_vels-mu) / sigma
        metrics["dyn_range_overall_z"]=float(np.max(z_vels) - np.min(z_vels))
        metrics["mean_velocity_z"]= float(np.mean(z_vels))
    else:
        metrics["dyn_range_overall_z"]=0.0
        metrics["mean_velocity_z"]=0.0

    vel_profile, centers= velocity_profile([{**n, "t0n": n["t0b"]} for n in notes], bins=time_bins)
    vel_diff =np.diff(vel_profile)
    metrics["velocity_profile"] =vel_profile
    metrics["velocity_smoothness_std"] = float(np.std(vel_diff)) if vel_diff.size >0 else 0.0
    if vel_profile.size>0:
        accent_thresh= np.percentile(vel_profile, 85)
        accents_idx= np.where(vel_profile > accent_thresh)[0]
    else:
        accents_idx=[]
    metrics["accents_count"]= int(len(accents_idx))
    metrics["accents_positions"] =(accents_idx/float(time_bins)).tolist()
    dens, centers = note_density_over_time(notes, bins=time_bins)
    metrics["note_density"]=dens
    notes_nonzero= [n for n in notes if n.get("velocity",0)>0]
    onsets= [n.get("t0b", n.get("t0n",0.0)) for n in notes_nonzero]
    durations=[n.get("dur",n.get("end",0.0)-n.get("start",0.0)) for n in notes_nonzero]

    if len(onsets) >= 2:
        ioi = np.diff(onsets)
        ioi_mean= float(np.mean(ioi))
        ioi_std =float(np.std(ioi))
        metrics["rubato_cv"] = ioi_std / (ioi_mean + 1e-9)
        metrics["rubato_index"] = ioi_std
        min_ioi = 0.02
        ratios=np.array(durations[:-1]) / np.maximum(ioi, min_ioi)
        metrics["rubato_logstd"] = float(np.std(np.log(np.maximum(ioi, min_ioi))))
        metrics["articulation_ratio_mean"]=float(np.mean(ratios))
        metrics["articulation_ratio_std"]=float(np.std(ratios))

    else: 
        metrics["rubato_cv"] =0.0
        metrics["rubato_logstd"] =0.0
        metrics["rubato_index"] =0.0
        metrics["articulation_ratio_mean"] =0.0
        metrics["articulation_ratio_std"] =0.0
    
    if pedal_events:
        total_on_time =0.0
        last_on_time = None
        for t, state in sorted(pedal_events, key=lambda x: x[0]):
            if state and last_on_time is None:
                last_on_time = t
            elif (not state) and last_on_time is not None:
                total_on_time+= t - last_on_time
                last_on_time=None
        if last_on_time is not None:
            total_on_time +=1.0 - last_on_time
        metrics["pedal_fraction"] =float(total_on_time)
        onsets=sorted([n.get("t0b",n.get("t0n",0.0)) for n in notes])
        pedal_on_counts=sum(1 for t,s in pedal_events if s)
        metrics["pedal_events_per_beat"]= pedal_on_counts / max(1.0, np.max(onsets)) if onsets else 0.0
    else:
        metrics["pedal_fraction"]= 0.0
        metrics["pedal_events_per_beat"]=0.0
    per_channel = defaultdict(list)
    for n in notes:
        per_channel[n.get("channel", 0)].append(n)
    chan_metrics ={}
    for ch, lst in per_channel.items():
        vels =[x["velocity"] for x in lst]
        chan_metrics[ch]= {
            "count": len(lst),
            "mean_velocity": float(np.mean(vels)) if vels else 0.0,
            "std_velocity": float(np.std(vels)) if vels else 0.0}
    metrics["per_channel"]= chan_metrics
    pitch_groups = defaultdict(list)
    for n in notes:
        pitch_groups[n["note"]].append(n["velocity"])
    pitch_std ={p: float(np.std(vs)) for p, vs in pitch_groups.items() if len(vs) > 2}
    metrics["pitch_velocity_std"] =pitch_std
    dens2=dens
    metrics["overall_polyphony_mean"] = float(np.mean(dens2)) if dens2.size > 0 else 0.0
    return metrics
def generate_advice_from_metrics(user_metrics,ref_metrics,user_heat,ref_heat,low_frac=LOW_REGISTER_FRACTION,high_frac=HIGH_REGISTER_FRACTION):
    advice= []
    if user_metrics is None:
        return ["No analysis available. Run Analyze first."]
    u_dr =user_metrics.get("dyn_range_overall",0.0)
    r_dr= ref_metrics.get("dyn_range_overall", 0.0) if ref_metrics else None
    if r_dr is not None:
        if u_dr <r_dr * 0.6:
            advice.append("Your overall dynamic range is significantly narrower than the references. Consider increasing contrast between soft and loud passages.")
        elif u_dr > r_dr* 1.4:
            advice.append("Your dynamic range is much wider than references - you may be over-accenting or playiing too loud on certain passages. Watch for loss of clarity in dense textures.")
        else:
            advice.append("Overall dynamic range is comparable to references.")
    else:
        advice.append(f"Overall dynamic range: {u_dr:.1f} velocity units (no reference).")
    u_acc = user_metrics.get("accents_count", 0)
    r_acc= ref_metrics.get("accents_count",0) if ref_metrics else None
    if r_acc is not None:
        if u_acc> r_acc* 1.5:
            advice.append("You have many more strong accents than references - check for unintentional emphasis.")
        elif u_acc< r_acc *0.6:
            advice.append("You have fewer accents than references - consider emphasizing phrase peaks. Check on your score for unnoticed accents.")
    u_rub= user_metrics.get("rubato_index", 0.0)
    r_rub = ref_metrics.get("rubato_index",0.0) if ref_metrics else None
    advice.append("Rubato index (user)= {:.3f}{}".format(u_rub,(f" (ref)= {r_rub:.3f}" if r_rub is not None else "")))
    if r_rub is not None and u_rub >r_rub+ 0.05:
        advice.append("Your rubato (timing variability) is higher than references - check phrase-level tempo shaping.")
    u_ped=user_metrics.get("pedal_fraction",0.0)
    r_ped = ref_metrics.get("pedal_fraction", 0.0) if ref_metrics else None
    advice.append("Pedal usage fraction (user)={:.2f}{}".format(u_ped, (f" (ref)= {r_ped:.2f}" if r_ped is not None else "")))
    if r_ped is not None:
        if u_ped>r_ped+0.15:
            advice.append("You use pedal signicantly more than references - may cause blurring.")
        elif u_ped <r_ped-0.15:
            advice.append("You use pedal less than references - consider tasteful sustain.")
    pitch_std= user_metrics.get("pitch_velocity_std", {})
    if pitch_std:
        high_std=[p for p, s in pitch_std.items() if s >12.0]
        if high_std:
            named=[midi_to_note_name(p) for p in high_std[:8]]
            advice.append(f"Inconsistent attack velocities for pitches: {named}. Consider consistent touch for repeated tones.")
    poly = user_metrics.get("overall_polyphony_mean",0.0)
    advice.append(f"Mean polyphony proxy = {poly:.2f} (higher-> denser).")
    advice.append("Timing analysis is beat-aligned, making rubato and accent patterns musically interpretable. Future work could extend this score-based DTW alignment.")
    critique=[]
    if u_dr < 10:
        critique.append("Dynamics: narrow - practice cresc./dim. across phrases.")
    elif u_dr>60:
        critique.append("Dynamics: wide - ensure control in louder passages.")
    elif 10<=u_dr <=20:
        critique.append("Dynamics: modest - could add more contrast for expressiveness.")
    rubato_cv=user_metrics.get("rubato_cv",0.0)
    rubato_logstd=user_metrics.get("rubato_logstd",0.0)
    if rubato_cv>0.12:
        critique.append("Timing stability: uneven - check technical control")
    elif rubato_logstd<0.015:
        critique.append("Timing expression: very rigid - consider subtle rubato to add musicality.")
    if poly >5.0:
        critique.append("Texture: dense - make primary voice stand out with voicing.")
    elif 2.0<=poly<=3.0:
        critique.append("Texture: light - can enrich with inner voices or accompaniment.")
    if u_ped>0.4:
        critique.append("Pedal: frequent use - check for blurring in fast passages.")
    elif u_ped <0.05:
        critique.append("Pedal: rare use - review pedal markings or consider adding subtle pedal for legato and enriching harmonies. If no pedal is a style-based or interpretational choice, disregard this comment.")
    if critique:
        advice.append("Critique summary:")
        advice.extend(crt for crt in critique)
    return advice 
def cache_key_for(path):
    try:
        st= os.stat(path)
        return f"{os.path.abspath(path)}{int(st.st_mtime)}"
    except:
        return os.path.abspath(path)
def cache_load(path):
    try:
        key= cache_key_for(path)
        fn =os.path.join(CACHE_DIR, f"{abs(hash(key))}.pkl")
        if os.path.exists(fn):
            import pickle
            with open(fn, "rb") as f:
                return pickle.load(f)
    except Exception:
        pass
    return None
def cache_save(path,data):
    try:
        key= cache_key_for(path)
        fn =os.path.join(CACHE_DIR, f"{abs(hash(key))}.pkl")
        import pickle
        with open(fn, "wb") as f:
            pickle.dump(data,f)
    except Exception:
        pass
def _times_and_vels(notes):
    if not notes:
        return np.array([]), np.array([])
    notes=[n for n in notes if n.get("velocity",0)>0]
    if not notes:
        return np.array([]), np.array([])
    notes=sorted(notes, key=lambda x: x["t0n"])
    t0= np.array([n.get("t0n",0.0) for n in notes], dtype=float)
    t1= np.array([n.get("t1n",t0[i]) for i, n in enumerate(notes)], dtype=float)
    mid =(t0+t1)/2.0
    vel=np.array([n.get("velocity",0) for n in notes],dtype=float)
    if mid.size> 1:
        mn,mx=mid.min(),mid.max()
        if mx>mn:
            mid= (mid - mn) / (mx - mn)
        else:
            mid =np.zeros_like(mid)
    order=np.argsort(mid)
    return mid[order],vel[order]
def loudness_consistency(notes, window=50):
    times,vels= _times_and_vels(notes)
    if vels.size==0:
        return 0.0
    w=max(1,min(window,max(1,int(len(vels)/5))))
    kernel=np.ones(w)/w
    mean_roll =np.convolve(vels,kernel, mode='same')
    std_roll=np.sqrt(np.convolve((vels - mean_roll)**2,kernel,mode='same'))
    rel= np.divide(std_roll,np.maximum(mean_roll,1e-6))
    return float(np.nanmean(rel))
def articulation_change_score(notes,short_thresh=0.12):
    if not notes:
        return{"staccato_fraction":0.0, "articulation_change_index":0.0}
    normalize_note_times(notes,1.0)
    dur_ratios=[]
    onsets=sorted([(n["t0n"],n["dur"]) for n in notes])
    times=np.array([t for t,_ in onsets]) 
    durs= np.array([d for _, d in onsets])
    if len(times)<2:
        return{"staccato_fraction":0.0, "articulation_change_index":0.0}
    ioi =np.diff(times)
    ratios = durs[:-1] / np.maximum(ioi,1e-6)
    staccato=np.sum(ratios< short_thresh)
    frac= float(staccato) / float(len(ratios))
    change_index =float(np.std(ratios))
    return{"staccato_fraction":frac, "articulation_change_index":change_index}    
def extract_expressive_features(notes, pedal_events,total_duration):
    feats ={}
    feats["loudness_consistency_relstd"]=loudness_consistency(notes)
    art=articulation_change_score(notes)
    feats.update(art)
    times,vels=_times_and_vels(notes)
    if vels.size>4:
        env=vels - np.mean(vels)
        spec=np.fft.rfft(env*np.hanning(len(env)))
        dt=np.median(np.diff(times))*total_duration if len(times)>1 else 1.0
        freqs=np.fft.rfftfreq(len(env), d=dt)
        if spec.size>1:
            am_rate=float(np.sum(freqs* np.abs(spec))/(np.sum(np.abs(spec))+ 1e-9))
            feats["amplitude_modulation_rate_hz"]= am_rate
        else:
            feats["amplitude_modulation_rate_hz"]=0.0
    else:
        feats["amplitude_modulation_rate_hz"]=0.0
    return feats
def generate_brief_commentary(user_metrics,expressive_feats, ref_metrics=None):
    parts=[]
    if user_metrics is None:
        return "No metrics available."
    dr= user_metrics.get("dyn_range_overall",0.0)
    parts.append(f"Dynamics: overall range ≈ {dr:.0f} velocity units.")
    lc= expressive_feats.get("loudness_consistency_relstd",None) if expressive_feats else None
    if lc is not None:
        parts.append(f"Loudness consistency (rel std) = {lc:.2f} (lower = steadier).")
    stacc= expressive_feats.get("staccato_fraction",None) if expressive_feats else None
    if stacc is not None:
        parts.append(f"Detached notes fraction ≈ {stacc*100:.0f}%.")
    vib =expressive_feats.get("vibrato_rate_hz",None) if expressive_feats else None
    if vib:
        parts.append(f"Detected vibrato ≈ {vib:.2f} Hz (depth {expressive_feats.get('vibrato_depth_cents',0):.1f} cents).")
    rub =user_metrics.get("rubato_index", 0.0)
    parts.append(f"Rubato index = {rub:.3f}.")
    score = 70.0
    if dr < 8:
        score-=8
    if lc and lc>0.25:
        score-=6
    if stacc and stacc>0.4:
        score -= 5
    if rub > 0.12:
        score-=6
    score = max(20, min(95,score))
    parts.append(f"Overall Performance Score (heuristic): {score:.0f}/100")
    return " ".join(parts)
class PerformanceInterpreterApp:
    def _extract_arrays_from_notes(self,notes):
        if not notes:
            return np.array([]), np.array([]), np.array([])
        pitches=np.array([n['note'] for n in notes],dtype= np.int32)
        times= np.array([n['t0n'] for n in notes],dtype= np.float32)
        velocities = np.array([n['vel'] for n in notes],dtype= np.float32)
        return pitches,times,velocities
    def __init__(self,master):
        self.master=master 
        if master is not None:
            master.title("Performaster - Performance Interpretation Analytics")
        self.user_path=None
        self.ref_paths= []
        self.user_notes=[]
        self.ref_notes_list=[]
        self.user_pedal=[]
        self.ref_pedals=[]
        self.user_duration= 1.0
        self.ref_durations=[]
        self.user_heat=None
        self.ref_heats=[]
        self.ref_heat_avg=None
        self.diff_heat=None
        self.user_metrics=None
        self.ref_metrics=None
        self.expressive_feats=None
        self.advice=None
        self.midi_velocity=None
        self.time_bins=TIME_BINS
        self.top_buttons=[]
        if master is not None:
            self._build_ui()
    def _build_ui(self):
        top= tk.Frame(self.master)
        top.pack(side= "top",fill="x",padx=6,pady=6)
        btn_load_user=tk.Button(top,text="Load User MIDI",command=self.load_user)
        btn_load_user.pack(side="left", padx=3)
        self.top_buttons.append(btn_load_user)
        btn_add_ref=tk.Button(top,text="Add Reference MIDI",command=self.add_reference)
        btn_add_ref.pack(side="left", padx=3)
        self.top_buttons.append(btn_add_ref)
        btn_remove_ref=tk.Button(top,text="Remove Ref", command=self.remove_selected_ref)
        btn_remove_ref.pack(side="left", padx=3)
        self.top_buttons.append(btn_remove_ref)
        btn_analyze=tk.Button(top,text="Analyze", command=self.analyze_threaded)
        btn_analyze.pack(side="left", padx=12)
        self.top_buttons.append(btn_analyze)
        params=tk.Frame(self.master)
        params.pack(fill="x",padx=6,pady =4)
        tk.Label(params,text="Time bins:").pack(side="left")
        self.time_bins_var= tk.IntVar(value=self.time_bins)
        sb_time=tk.Spinbox(params, from_=100,to=2000,increment=50,textvariable=self.time_bins_var,width= 6)
        sb_time.pack(side="left",padx=2)
        info=tk.Frame(self.master)
        info.pack(fill="x",padx=6)
        self.user_label= tk.Label(info, text="User: (none)")
        self.user_label.pack(anchor="w")
        self.refs_label= tk.Label(info, text="References: 0")
        self.refs_label.pack(anchor="w")
        ref_frame= tk.Frame(self.master)
        ref_frame.pack(fill="x",padx=6)
        tk.Label(ref_frame,text ="References (select to remove)").pack(anchor="w")
        self.ref_listbox= tk.Listbox(ref_frame,height =4, selectmode= tk.SINGLE)
        self.ref_listbox.pack(fill="x",expand=True)
        nb=ttk.Notebook(self.master)
        nb.pack(fill="both",expand=True,padx=6,pady=6)
        self.frame_advice= tk.Frame(nb)
        nb.add(self.frame_advice,text="Advice & Summary")
        self.advice_text= tk.Text(self.frame_advice,wrap ="word")
        self.advice_text.pack(fill="both",expand=True)
        bottom= tk.Frame(self.master)
        bottom.pack(side="bottom",fill="x",padx =6,pady =4)
        self.status= tk.Label(bottom,text ="Ready")
        self.status.pack(side="left",padx=6)
    def _normalize_notes_dicts(self,notes):
        for n in notes:
            if "pitch" not in n and "note" in n:
                n["pitch"]= n.get("note")
            if "note" not in n and "pitch" in n:
                n["note"]= n.get("pitch")
            if "vel" not in n and "velocity" in n:
                n["vel"]=n.get("velocity")
            if "velocity" not in n and "vel" in n:
                n["velocity"]= n.get("vel")
            if "t0n" not in n:
                if "t0n" not in n:
                    n["t0n"]=n["start"]
                elif "time" in n:
                    n["t0n"]=n["time"]
            if "channel" not in n:
                n["channel"]= n.get("ch",0)
            if"t0b" not in n:
                n["t0b"]=n.get("t0n",0.0)
        return notes                    
    def load_user(self):
        path = filedialog.askopenfilename(title="Select user MIDI", filetypes=[("MIDI", "*.mid *.midi")])
        if not path:
            return
        self.user_path=path
        self.user_label.config(text=f"User:{os.path.basename(path)}")
        self._parse_user()
        _, _, vels =self._extract_arrays_from_notes(self.user_notes)
        self.midi_velocity= vels.tolist() if len(vels) else None
        self.status.config(text="User file loaded")
    def add_reference(self):
        path= filedialog.askopenfilename(title="Select reference MIDI", filetypes=[("MIDI", "*.mid *.midi")])
        if not path:
            return
        notes,pedal,dur=self._parse_midi_file(path)
        if not notes:
            messagebox.showwarning("Reference MIDI","No notes found in this MIDI file.")
            return
        self.ref_paths.append(path)
        self.ref_notes_list.append(notes)
        self.ref_pedals.append(pedal)
        self.ref_durations.append(dur)
        self.ref_listbox.insert("end",os.path.basename(path))
        self.refs_label.config(text=f"References: {len(self.ref_paths)}")
        self.status.config(text=f"Reference added: {os.path.basename(path)}")
    def remove_selected_ref(self):
        sel= self.ref_listbox.curselection()
        if not sel:
            messagebox.showinfo("Remove reference", "Select a reference first.")
            return
        idx= sel[0]
        try:
            if idx< len(self.ref_paths):
                p= self.ref_paths[idx]
                if p and os.path.exists(p) and (p.startswith(tempfile.gettempdir())):
                    try:
                        os.remove(p)
                    except Exception:
                        pass
        except Exception:
            pass
        self.ref_listbox.delete(idx)
        del self.ref_paths[idx]
        del self.ref_notes_list[idx]
        del self.ref_pedals[idx]
        del self.ref_durations[idx]
        self.refs_label.config(text=f"References: {len(self.ref_paths)}")
        self.status.config(text="Reference removed")
    def _parse_midi_file(self,path):
        cached= cache_load(path)
        if cached:
            raw_notes= cached["notes"]
            pedal= cached["pedal"]
            duration=cached["duration"]
            print("cached duration =", duration)
            if duration <= 0:
                duration=max((n["end"] for n in raw_notes), default=1.0)
            normalize_note_times(raw_notes,duration)
            notes= self._normalize_notes_dicts(raw_notes)
            for n in notes:
                if "t0b" not in n:
                    n["t0b"]=n.get("t0n",0.0)
            return notes, pedal,duration
        try:
            mid =mido.MidiFile(path)
        except Exception as e:
            messagebox.showerror("MIDI Error", f"Could not open MIDI file: {e}")
            return [], [], 1.0
        events, duration_from_parser= midi_to_events(mid)
        notes,pedal= events_to_notes_and_pedal(events)
        beat_times=extract_beat_times(events,mid.ticks_per_beat)
        notes=normalize_note_times_by_beat(notes, beat_times)
        raw_max_end= max((n["end"] for n in notes), default=0)
        total= raw_max_end
        normalize_note_times(notes, total)
        notes_final= self._normalize_notes_dicts(notes)
        for n in notes:
            if "t0b" not in n:
                n["t0b"]=n.get("t0n",0.0)        
        pedal_norm=[]
        if pedal:
            pedal_norm=[(t/total if total> 0 else 0.0, v) for t, v in pedal]
        raw_notes_to_save= [
            {
                "note": n["note"],
                "start": n["start"],
                "end": n["end"],
                "velocity": n["velocity"],
                "channel": n.get("channel",0),
                "track": n.get("track", 0)}
            for n in notes]
        data= {
            "notes":raw_notes_to_save,
            "pedal":pedal_norm,
            "duration":total,
        }
        try:
            cache_save(path,data)
        except Exception:
            pass
        return notes_final,pedal_norm,total       
    def _parse_user(self):
        if not self.user_path:
            return
        notes,pedal,dur= self._parse_midi_file(self.user_path)
        self.user_notes= notes
        self.user_pedal =pedal
        self.user_duration =dur
    def analyze_threaded(self):
        if not self.user_path:
            messagebox.showwarning("No user MIDI", "Please load a user MIDI file first.")
            return
        self.time_bins= int(self.time_bins_var.get())
        self._set_busy(True,"Analyzing...")
        t= threading.Thread(target=self._analyze_worker, daemon=True)
        t.start()        
    def _analyze_worker(self):
        try:
            self._parse_user()
            pedal_u=self.user_pedal
            self.user_metrics= compute_metrics(self.user_notes, pedal_u, self.user_duration,time_bins=self.time_bins)
            if self.ref_notes_list:
                all_ref_notes= [n for lst in self.ref_notes_list for n in lst]
                all_ref_pedal = [p for lst in self.ref_pedals for p in lst]
                ref_dur=max(self.ref_durations) if self.ref_durations else 1.0
                self.ref_metrics =compute_metrics(all_ref_notes,all_ref_pedal,ref_dur, time_bins=self.time_bins)
            else:
                self.ref_metrics=None
            self.expressive_feats= extract_expressive_features(self.user_notes,pedal_u,self.user_duration)
            self.advice =generate_advice_from_metrics(self.user_metrics,self.ref_metrics,self.user_heat,self.ref_heat_avg)
            self.commentary= generate_brief_commentary(self.user_metrics, self.expressive_feats, self.ref_metrics)
            _,_, vels=self._extract_arrays_from_notes(self.user_notes)
            self.midi_velocity=vels.tolist() if len(vels) else None
            if self.master:
                self.master.after(0,self._on_analyze_done)
        except Exception as e:
            logger.exception("Analysis failed")
            if self.master:
                self.master.after(0,lambda:messagebox.showerror("Error", f"Analysis failed: {e}"))
                self.master.after(0,lambda:self._set_busy(False))
    def _on_analyze_done(self):
        try:
            self.show_advice()
            self._set_busy(False)
            self.status.config(text="Analysis complete")
        except Exception:
            logger.exception("On analyze done failed")
            self._set_busy(False)
    def show_advice(self):
        if getattr(self, "advice", None) is None:
            self.advice_text.delete("1.0", "end")
            self.advice_text.insert("1.0", "No advice yet. Run Analyze")
            return
        self.advice_text.delete("1.0", "end")
        for line in self.advice:
            self.advice_text.insert("end", "• " + line + "\n\n")
        try:
            if getattr(self, "commentary", None):
                self.advice_text.insert("end", "\n---\n" + self.commentary)
        except Exception:
            pass
    def _set_busy(self,busy:bool, message:str=None):
        state ="disabled" if busy else "normal"
        for b in getattr(self,"top_buttons",[]):
            try:
                b.config(state=state)
            except Exception:
                pass
            self.status.config(text=message or "Working...")
def main():
    import tkinter as tk
    root=tk.Tk()
    app = PerformanceInterpreterApp(root)
    root.geometry("1280x900")
    root.mainloop()
    return 0
if __name__ == "__main__":
    import sys
    sys.exit(main())