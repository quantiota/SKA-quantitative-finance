# Binary Information Flow — Sound Encoding

## Concept

Each 4-bit word in the binary information flow is encoded as a 30ms sinusoidal segment.
The WAV file is the market speaking in sound.

## Encoding

| word | transition | sound | sweep |
|---|---|---|---|
| `0000` | neutral-neutral | silence | — |
| `0001` | neutral-bull | A3 220 Hz | ascending |
| `0100` | bull-neutral | A4 440 Hz | descending |
| `0010` | neutral-bear | E4 330 Hz | ascending |
| `1000` | bear-neutral | E5 659 Hz | descending |

**Open transitions** (neutral-bull, neutral-bear) → ascending pitch sweep  
**Close transitions** (bull-neutral, bear-neutral) → descending pitch sweep

## Sequences as melodic phrases

```
LONG  (320): silence → A3↑ → A4↓ → silence
SHORT (640): silence → E4↑ → E5↓ → silence
```

Each sequence is one melodic phrase. The market has a 4-note vocabulary.

## Properties

- **Visually readable**: zoom in Audacity to see each 4-bit word as one sinusoidal segment.
- **AI-processable**: the audio can be fed directly to multimodal audio models for pattern recognition.

## Generation

```bash
python3 binary_flow_to_sound.py
```

Output: `binary_flow.wav` — one loop (3,497 words, ~1.7 minutes)
