# Performaster
Performaster is a performance analysis framework that compares MIDI recordings. This offers researchers a means of quantifying interpretative decisions and provides students with feedback to aim to help them reach a better understanding of what a sophisticated performance entails.
Project Overview and Goals

Objective: Examine MIDI piano performances to measure expressive elements including dynamics, expressive timing (rubato), articulation, and pedal usage.

Goals:

Obtain comprehensive performance statistics from MIDI files.
Evaluate user performances against benchmark recordings.
Offer practical suggestions and concise insights to facilitate enhancement.
Facilitate the depiction of dynamics, note density, and velocity patterns.
Architecture & Workflow

MIDI Parsing:
Converts MIDI files into time-stamped events.
Takes note of note on/off events, velocities, channels and pedal control changes.

Feature Extraction
Computes metrics including:
Overall dynamic range and per-segment dynamics.
Velocity profiles, note density, rubato index and pedal usage fraction.
Expressive features: loudness consistency, staccato fraction, amplitude modulation rate.

Comparison & Advice Generation
Compares user metrics to reference performances.
Generates actionable advice and critiques on dynamics, timing, articulation, texture, and pedal usage.

Caching & Reproducibility
Uses local cache to store parsed MIDI data for faster re-analysis.

GUI Interface

Built with Tkinter for cross-platform visualization and user interaction.
People can load their MIDI files, then add a Reference MIDI file they’d like to compare the original to. After hitting the Analysis button, they can view metrics and advice.

Dependencies:

Python ≥ 3.8

mido, numpy, tkinter

Example Outputs / Demo

The Velocity Profile shows you the average note velocities over time.
Note Density displays the number of notes per time bin.
Advice & Commentary provides heuristic suggestions based on user vs reference performances.

Sample Advice:

• Overall dynamic range is comparable to references.
• Rubato index (user)= 0.001 (ref)= 0.001
• Pedal usage fraction (user)=0.55 (ref)= 0.49
• Inconsistent attack velocities for pitches: ['A#/Bb4', 'A#/Bb3', 'A#/Bb2', 'D#/Eb5', 'F5', 'D#/Eb3', 'G4', 'G5']. Consider consistent touch for repeated tones.
• Mean polyphony proxy = 7.77 (higher-> denser).
• Timing analysis is beat-aligned, making rubato and accent patterns musically interpretable. Future work could extend this score-based DTW alignment.

Critique summary:

• Dynamics: narrow - practice cresc./dim. across phrases.
• Timing stability: uneven - check technical control
• Texture: dense - make primary voice stand out with voicing.
• Pedal: frequent use - check for blurring in fast passages.

---
Dynamics: overall range ≈ 0 velocity units. Loudness consistency (rel std) = 0.14 (lower = steadier). Detached notes fraction ≈ 0%. Rubato index = 0.001. Overall Performance Score (heuristic): 62/100

Potential Applications
Music Education: Helps students understand the decision making of professional musicians and provides them advice to develop a similar decision making process if they so desire
Research: Quantitative analysis of performance practice and stylistic differences.
Practice Tools: Real-time feedback for pianists on dynamics, timing, articulation, and pedal use.
Demo Video
https://www.youtube.com/watch?v=a7eKG5UW5t0
