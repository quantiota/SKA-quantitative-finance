"""
binary_flow_to_sound.py

Converts the binary information flow to audio (WAV).

Mapping:
  Each 4-bit word (0-15) maps to a frequency on a chromatic scale.
  The 5 structural transition codes map to distinct pitches:
    0000 (neutral-neutral) → silence
    0001 (neutral-bull)    → E4  329.63 Hz  (rising)
    0010 (neutral-bear)    → A3  220.00 Hz  (falling)
    0100 (bull-neutral)    → G4  392.00 Hz  (returning up)
    1000 (bear-neutral)    → C4  261.63 Hz  (returning down)
    other words            → proportional frequency

Duration per word: 30ms
Sample rate: 44100 Hz
Output: binary_flow.wav
"""

import numpy as np
import wave
import struct
import os

INPUT  = "/SKA-quantitative-finance/ska_engine_c/binary_transition_space/binary_to_sound/binary_flow_sample.txt"
OUTPUT = "/SKA-quantitative-finance/ska_engine_c/binary_transition_space/binary_to_sound/binary_flow.wav"

SAMPLE_RATE  = 44100
WORD_DURATION = 0.030   # 30ms per word
MAX_WORDS    = 3497      # one loop

# frequency map: 4-bit word integer → Hz
# LONG  pair: 0001 (open) and 0100 (close) → A3 / A4  (octave)
# SHORT pair: 0010 (open) and 1000 (close) → E4 / E5  (octave)
FREQ_MAP = {
    0:  0,       # 0000 neutral-neutral → silence
    1:  220.00,  # 0001 neutral-bull    → A3 (LONG open)
    4:  440.00,  # 0100 bull-neutral    → A4 (LONG close)
    2:  329.63,  # 0010 neutral-bear    → E4 (SHORT open)
    8:  659.25,  # 1000 bear-neutral    → E5 (SHORT close)
    5:  523.25,  # 0101 bull-bull       → C5
    6:  293.66,  # 0110 bull-bear       → D4
    9:  392.00,  # 1001 bear-bull       → G4
    10: 196.00,  # 1010 bear-bear       → G3
}
# fill remaining with exponential scale
for v in range(16):
    if v not in FREQ_MAP:
        FREQ_MAP[v] = 110.0 * (2 ** (v / 12.0))


# open transitions  (0001, 0010) → sine wave  (smooth)
# close transitions (0100, 1000) → square wave (sharp)
OPEN_WORDS  = {1, 2}
CLOSE_WORDS = {4, 8}

def word_to_samples(word, freq, duration, sample_rate, amplitude=0.4):
    n = int(sample_rate * duration)
    if freq == 0:
        return np.zeros(n)
    t = np.linspace(0, duration, n, endpoint=False)
    envelope = np.ones(n)
    attack = int(n * 0.05)
    decay  = int(n * 0.10)
    envelope[:attack] = np.linspace(0, 1, attack)
    envelope[-decay:] = np.linspace(1, 0, decay)
    if word in CLOSE_WORDS:
        # descending sweep: freq → freq * 0.75
        freq_sweep = np.linspace(freq, freq * 0.75, n)
    else:
        # ascending sweep: freq * 0.75 → freq
        freq_sweep = np.linspace(freq * 0.75, freq, n)
    phase = np.cumsum(2 * np.pi * freq_sweep / sample_rate)
    return amplitude * np.sin(phase) * envelope


def main():
    # read words
    with open(INPUT) as f:
        lines = f.readlines()

    # skip header lines, find word data
    words = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith('Binary') or line[0].isdigit() and 'words' in line:
            continue
        tokens = line.split()
        for tok in tokens:
            if len(tok) == 4 and all(c in '01' for c in tok):
                words.append(int(tok, 2))

    words = words[:MAX_WORDS]
    print(f"Words loaded   : {len(words):,}")

    # count unique words
    from collections import Counter
    counts = Counter(words)
    print(f"Unique words   : {len(counts)}")
    for w, c in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {w:04b} ({w:2d}) : {c:,}")

    # generate audio
    samples = []
    for w in words:
        freq = FREQ_MAP.get(w, 440.0)
        s = word_to_samples(w, freq, WORD_DURATION, SAMPLE_RATE)
        samples.append(s)

    audio = np.concatenate(samples)
    audio_int16 = (audio * 32767).astype(np.int16)

    # write WAV
    with wave.open(OUTPUT, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_int16.tobytes())

    duration_s = len(audio) / SAMPLE_RATE
    print(f"\nDuration       : {duration_s:.1f}s ({duration_s/60:.1f} min)")
    print(f"Saved          : {OUTPUT}")


if __name__ == "__main__":
    main()
