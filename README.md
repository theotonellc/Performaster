# Performaster
Performaster is a performance analysis framework that compares MIDI recordings. This offers researchers a means of quantifying interpretative decisions and provides students with feedback to aim to help them reach a better understanding of what a sophisticated performance entails.
---
## Project Overview and Goals

**Objective**
Examine MIDI piano performances to measure expressive elements including:
- Dynamics
- Expressive timing (rubato)
- Articulation
- Pedal usage

**Goals**

- Obtain comprehensive performance statistics from MIDI files
- Evaluate user performances against benchmark recordings
- Offer practical suggestions and concise insights to facilitate enhancement
- Visualize dynamics, note density, and velocity patterns

---

## Architecture & Workflow

### MIDI Parsing:
- Converts MIDI files into time-stamped events
- Tracks note on/off events, velocities, channels and pedal control changes

### Feature Extraction
Computed metrics include:
- Overall dynamic range and per-segment dynamics
- Velocity profiles
- Note density
- Rubato index
- Pedal usage fraction
- Expressive features: loudness consistency, staccato fraction, and amplitude modulation rate

### Comparison & Advice Generation
- Compares user metrics to reference performances
- Generates actionable critique on:
  - Dynamics
  - Timing
  - Articulation
  - Texture
  - Pedal usage

### Caching & Reproducibility
- Uses local cache to store parsed MIDI data for faster re-analysis

## GUI Interface

Workflow:
1.Load a user MIDI performance
2.Load a reference MIDI performance
3.Run the analysis
4.View metrics, plots, and interpretative feedback

---

## Example Output

- **Velocity Profile:** average note velocities over time
- **Note Density:** number of notes per time bin.
- **Advice & Commentary:** heuristic feedback based on user vs reference comparisons

### Sample Advice

- Overall dynamic range is comparable to references.
- Rubato index (user)= 0.001 (ref)= 0.001
- Pedal usage fraction (user)=0.55 (ref)= 0.49
- Inconsistent attack velocities for pitches: ['A#/Bb4', 'A#/Bb3', 'A#/Bb2', 'D#/Eb5', 'F5', 'D#/Eb3', 'G4', 'G5']. Consider consistent touch for repeated tones

- Mean polyphony proxy = 7.77 (higher-> denser)
- Timing analysis is beat-aligned, making rubato and accent patterns musically interpretable. Future work could extend this score-based DTW alignment.

### Critique Summary

- **Dynamics:** narrow - practice cresc./dim. across phrases
- **Timing stability:** uneven - address technical control
- **Texture:** dense - make primary voice stand out with voicing
- **Pedal:** frequent use - check for blurring in fast passages
**Additional metrics**
- Overall Dynamic Range ≈ 0 velocity units
- Loudness consistency (rel std) = 0.14 (lower = steadier)
- Detached notes fraction ≈ 0%
- Rubato index = 0.001
- Overall Performance Score (heuristic): **62 / 100**

---

## Dependencies

- Python ≥ 3.8
- mido
- numpy
- tkinter (bundled with standard Python distributions)

---

## Potential Applications
- **Music Education**
  Helps students understand interpretative decision making and develop expressive control.

-**Research**
  Enables quantitative analysis of performance practice and stylistic differences.

- **Practice Tools**
  Provides structured feedback on dynamics, timing, articulation, and pedal use.

---

## Demo

Video demonstration
https://www.youtube.com/watch?v=a7eKG5UW5t0
