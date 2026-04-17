# False Start Panel — Observed Cases

Forensic archive of false starts captured on the live Grafana panel.
Each entry records the transition sequence and P values observed in real time.

---

## Format

```
Date       : UTC timestamp of the loop
Trade ID   : Binance trade_id at the false start transition
Sequence   : transition path observed
P values   : P at each transition
```

---

## Cases

<!-- Add new entries below as observed -->

---

### Case 1 — Bull False Start 

**Observed sequence** (trade_id window 1607297434–1607297456):

- `neutral→neutral` P = 1.00 — extended neutral gap
- `neutral→bull`    P ≈ 0.66 — at ~1607297440
- `bull→neutral`    P ≈ 0.51 — bull pair 1 complete ✓
- `neutral→bear`    P ≈ 0.15 — at ~1607297442
- `bear→bull`       P ≈ 0.45 — at ~1607297443
- `bull→bear`       P ≈ 0.02 — at ~1607297444
- `bear→neutral`    P ≈ 0.51 — at ~1607297445
- `neutral→bull`    P ≈ 0.66 — at ~1607297446
- `bull→neutral`    P ≈ 0.51 — bull pair 2 complete ✓
- `neutral→neutral` P = 1.00 — neutral gap resumes

All 7 transition types observed within ~22 trade IDs.

![False Start Case 1](screenshot_case1.png)

**Episode sequence** (neutral→neutral → ... → neutral→neutral):

```python
{
    "date": "2026-04-14T11:37:30.746Z",
    "trade_id_window": [1607297434, 1607297456],
    "sequence": [
        {"transition": "neutral→neutral", "P": 1.00},
        {"transition": "neutral→bull",    "P": 0.66},
        {"transition": "bull→neutral",    "P": 0.51},
        {"transition": "neutral→bear",    "P": 0.15},
        {"transition": "bear→bull",       "P": 0.45},
        {"transition": "bull→bear",       "P": 0.02},
        {"transition": "bear→neutral",    "P": 0.51},
        {"transition": "neutral→bull",    "P": 0.66},
        {"transition": "bull→neutral",    "P": 0.51},
        {"transition": "neutral→neutral", "P": 1.00}
    ]
}
```

### Case 2 — 2026-04-14T12:21:45.115Z

**Observed sequence** (trade_id window 1607313366–1607313382):

- `neutral→neutral` P = 1.00 — extended neutral gap
- `neutral→bear`    P ≈ 0.15 — at ~1607313371
- `bear→neutral`    P ≈ 0.51 — bear pair complete ✓
- `neutral→bull`    P ≈ 0.66 — at ~1607313373
- `bull→bear`       P ≈ 0.02 — at ~1607313374
- `bear→bull`       P ≈ 0.45 — at ~1607313375
- `bull→neutral`    P ≈ 0.51 — at ~1607313376
- `neutral→neutral` P = 1.00 — neutral gap resumes

![False Start Case 2](screenshot_case2.png)

**Episode sequence** (neutral→neutral → ... → neutral→neutral):

```python
{
    "date": "2026-04-14T12:21:45.115Z",
    "trade_id_window": [1607313366, 1607313382],
    "sequence": [
        {"transition": "neutral→neutral", "P": 1.00},
        {"transition": "neutral→bear",    "P": 0.15},
        {"transition": "bear→neutral",    "P": 0.51},
        {"transition": "neutral→bull",    "P": 0.66},
        {"transition": "bull→bear",       "P": 0.02},
        {"transition": "bear→bull",       "P": 0.45},
        {"transition": "bull→neutral",    "P": 0.51},
        {"transition": "neutral→neutral", "P": 1.00}
    ]
}
```
---


### Case 3 — 2026-04-14T12:44:51.600Z

**Observed sequence** (trade_id window 1607321228–1607321268):

- `neutral→neutral` P = 1.00 — extended neutral gap
- `neutral→bull`    P ≈ 0.66 — at ~1607321242
- `bull→neutral`    P ≈ 0.51 — at ~1607321244
- `neutral→bear`    P ≈ 0.15 — at ~1607321246
- `bear→bull`       P ≈ 0.45 — at ~1607321247
- `bull→bear`       P ≈ 0.02 — at ~1607321250
- `bear→neutral`    P ≈ 0.51 — at ~1607321266
- `neutral→neutral` P = 1.00 — neutral gap resumes

![False Start Case 3](screenshot_case3.png)

**Episode sequence** (neutral→neutral → ... → neutral→neutral):

```python
{
    "date": "2026-04-14T12:44:51.600Z",
    "trade_id_window": [1607321228, 1607321268],
    "sequence": [
        {"transition": "neutral→neutral", "P": 1.00},
        {"transition": "neutral→bull",    "P": 0.66},
        {"transition": "bull→neutral",    "P": 0.51},
        {"transition": "neutral→bear",    "P": 0.15},
        {"transition": "bear→bull",       "P": 0.45},
        {"transition": "bull→bear",       "P": 0.02},
        {"transition": "bear→neutral",    "P": 0.51},
        {"transition": "neutral→neutral", "P": 1.00}
    ]
}
```


### Case 4 — 2026-04-14T14:10:22.829Z

**Observed sequence** (trade_id window 1607389098–1607389108):

- `neutral→neutral` P = 1.00 — extended neutral gap
- `neutral→bear`    P ≈ 0.15 — at ~1607389103
- `bear→bull`       P ≈ 0.45 — at ~1607389104
- `bull→neutral`    P ≈ 0.51 — at ~1607389105
- `neutral→neutral` P = 1.00 — neutral gap resumes

![False Start Case 4](screenshot_case4.png)

**Episode sequence** (neutral→neutral → ... → neutral→neutral):

```python
{
    "date": "2026-04-14T14:10:22.829Z",
    "trade_id_window": [1607389098, 1607389108],
    "sequence": [
        {"transition": "neutral→neutral", "P": 1.00},
        {"transition": "neutral→bear",    "P": 0.15},
        {"transition": "bear→bull",       "P": 0.45},
        {"transition": "bull→neutral",    "P": 0.51},
        {"transition": "neutral→neutral", "P": 1.00}
    ]
}
```


### Case 5 — 2026-04-14T14:23:35.600Z

**Observed sequence** (trade_id window 1607399476–1607399494):

- `neutral→neutral` P = 1.00 — extended neutral gap
- `neutral→bull`    P ≈ 0.66 — at ~1607399482
- `bull→bear`       P ≈ 0.02 — at ~1607399483
- `bear→neutral`    P ≈ 0.51 — at ~1607399484
- `neutral→neutral` P = 1.00 — neutral gap resumes

![False Start Case 5](screenshot_case5.png)

**Episode sequence** (neutral→neutral → ... → neutral→neutral):

```python
{
    "date": "2026-04-14T14:23:35.600Z",
    "trade_id_window": [1607399476, 1607399494],
    "sequence": [
        {"transition": "neutral→neutral", "P": 1.00},
        {"transition": "neutral→bull",    "P": 0.66},
        {"transition": "bull→bear",       "P": 0.02},
        {"transition": "bear→neutral",    "P": 0.51},
        {"transition": "neutral→neutral", "P": 1.00}
    ]
}
```

### Case 6 — 2026-04-14T14:50:52.094Z

**Observed sequence** (trade_id window 1607434219–1607434236):

- `neutral→neutral` P = 1.00 — extended neutral gap
- `neutral→bear`    P ≈ 0.13 — at ~1607434228
- `bear→neutral`    P ≈ 0.51 — bear pair complete ✓
- `neutral→bull`    P ≈ 0.66 — at ~1607434230
- `bull→bear`       P ≈ 0.02 — at ~1607434231
- `bear→neutral`    P ≈ 0.51 — at ~1607434232
- `neutral→neutral` P = 1.00 — neutral gap resumes

![False Start Case 6](screenshot_case6.png)

**Episode sequence** (neutral→neutral → ... → neutral→neutral):

```python
{
    "date": "2026-04-14T14:50:52.094Z",
    "trade_id_window": [1607434219, 1607434236],
    "sequence": [
        {"transition": "neutral→neutral", "P": 1.00},
        {"transition": "neutral→bear",    "P": 0.13},
        {"transition": "bear→neutral",    "P": 0.51},
        {"transition": "neutral→bull",    "P": 0.66},
        {"transition": "bull→bear",       "P": 0.02},
        {"transition": "bear→neutral",    "P": 0.51},
        {"transition": "neutral→neutral", "P": 1.00}
    ]
}
```


### Case 7 — 2026-04-14T16:01:02.800Z

**Observed sequence** (trade_id window 1607465174–1607465185):

- `neutral→neutral` P = 1.00 — extended neutral gap
- `neutral→bear`    P ≈ 0.13 — at ~1607465178
- `bear→bull`       P ≈ 0.45 — at ~1607465179
- `bull→bear`       P ≈ 0.02 — at ~1607465180
- `bear→neutral`    P ≈ 0.51 — at ~1607465181
- `neutral→neutral` P = 1.00 — neutral gap resumes

![False Start Case 7](screenshot_case7.png)

**Episode sequence** (neutral→neutral → ... → neutral→neutral):

```python
{
    "date": "2026-04-14T16:01:02.800Z",
    "trade_id_window": [1607465174, 1607465185],
    "sequence": [
        {"transition": "neutral→neutral", "P": 1.00},
        {"transition": "neutral→bear",    "P": 0.13},
        {"transition": "bear→bull",       "P": 0.45},
        {"transition": "bull→bear",       "P": 0.02},
        {"transition": "bear→neutral",    "P": 0.51},
        {"transition": "neutral→neutral", "P": 1.00}
    ]
}
```


### Case 8 — 2026-04-14T16:27:15.437Z

**Observed sequence** (trade_id window 1607464542–1607464551):

- `neutral→neutral` P = 1.00 — extended neutral gap
- `neutral→bear`    P ≈ 0.15 — at ~1607464543
- `bear→bull`       P ≈ 0.45 — at ~1607464544
- `bull→neutral`    P ≈ 0.51 — at ~1607464545
- `neutral→bear`    P ≈ 0.15 — at ~1607464546
- `bear→bull`       P ≈ 0.45 — at ~1607464547
- `bull→bear`       P ≈ 0.02 — at ~1607464548
- `bear→neutral`    P ≈ 0.51 — at ~1607464549
- `neutral→neutral` P = 1.00 — neutral gap resumes

Inner length 7 — beyond library (inner ≤ 4). Δ pips = −1.

![False Start Case 8](screenshot_case8.png)

**Episode sequence** (neutral→neutral → ... → neutral→neutral):

```python
{
    "date": "2026-04-14T16:27:15.437Z",
    "trade_id_window": [1607464542, 1607464551],
    "sequence": [
        {"transition": "neutral→neutral", "P": 1.00},
        {"transition": "neutral→bear",    "P": 0.15},
        {"transition": "bear→bull",       "P": 0.45},
        {"transition": "bull→neutral",    "P": 0.51},
        {"transition": "neutral→bear",    "P": 0.15},
        {"transition": "bear→bull",       "P": 0.45},
        {"transition": "bull→bear",       "P": 0.02},
        {"transition": "bear→neutral",    "P": 0.51},
        {"transition": "neutral→neutral", "P": 1.00}
    ]
}
```


### Case 9 — 2026-04-15T13:45:18.679Z

**Observed sequence** (trade_id window 1607853467–1607853476):

- `neutral→neutral` P = 1.00 — extended neutral gap
- `neutral→bear`    P ≈ 0.14 — at ~1607853471
- `bear→neutral`    P ≈ 0.51 — at ~1607853472
- `neutral→bull`    P ≈ 0.66 — at ~1607853473
- `bull→neutral`    P ≈ 0.51 — at ~1607853474
- `neutral→neutral` P = 1.00 — neutral gap resumes

Inner length 4 — SHORT pair + LONG pair. Δ pips = 0.

![False Start Case 9](screenshot_case9.png)

**Episode sequence** (neutral→neutral → ... → neutral→neutral):

```python
{
    "date": "2026-04-15T13:45:18.679Z",
    "trade_id_window": [1607853467, 1607853476],
    "sequence": [
        {"transition": "neutral→neutral", "P": 1.00},
        {"transition": "neutral→bear",    "P": 0.14},
        {"transition": "bear→neutral",    "P": 0.51},
        {"transition": "neutral→bull",    "P": 0.66},
        {"transition": "bull→neutral",    "P": 0.51},
        {"transition": "neutral→neutral", "P": 1.00}
    ]
}
```


### Case 10 — 2026-04-15T14:23:02.255Z

**Observed sequence** (trade_id window 1607874138–1607874147):

- `neutral→neutral` P = 1.00 — extended neutral gap
- `neutral→bull`    P ≈ 0.66 — at ~1607874142
- `bull→bear`       P ≈ 0.02 — at ~1607874143
- `bear→bull`       P ≈ 0.45 — at ~1607874144
- `bull→neutral`    P ≈ 0.51 — at ~1607874145
- `neutral→neutral` P = 1.00 — neutral gap resumes

Inner length 4 — LONG detour. Δ pips = +1.

![False Start Case 10](screenshot_case10.png)

**Episode sequence** (neutral→neutral → ... → neutral→neutral):

```python
{
    "date": "2026-04-15T14:23:02.255Z",
    "trade_id_window": [1607874138, 1607874147],
    "sequence": [
        {"transition": "neutral→neutral", "P": 1.00},
        {"transition": "neutral→bull",    "P": 0.66},
        {"transition": "bull→bear",       "P": 0.02},
        {"transition": "bear→bull",       "P": 0.45},
        {"transition": "bull→neutral",    "P": 0.51},
        {"transition": "neutral→neutral", "P": 1.00}
    ]
}
```


### Case 11 — 2026-04-15T15:29:38.410Z

**Observed sequence** (trade_id window 1607916604–1607916621):

- `neutral→neutral` P = 1.00 — extended neutral gap
- `neutral→bull`    P ≈ 0.66 — at ~1607916609
- `bull→bear`       P ≈ 0.02 — at ~1607916610
- `bear→neutral`    P ≈ 0.51 — at ~1607916611
- `neutral→bull`    P ≈ 0.66 — at ~1607916612
- `bull→bear`       P ≈ 0.02 — at ~1607916613
- `bear→bull`       P ≈ 0.45 — at ~1607916614
- `bull→neutral`    P ≈ 0.51 — at ~1607916615
- `neutral→neutral` P = 1.00 — neutral gap resumes

Inner length 7 — LONG false start + LONG detour. Δ pips = +1.

![False Start Case 11](screenshot_case11.png)

**Episode sequence** (neutral→neutral → ... → neutral→neutral):

```python
{
    "date": "2026-04-15T15:29:38.410Z",
    "trade_id_window": [1607916604, 1607916621],
    "sequence": [
        {"transition": "neutral→neutral", "P": 1.00},
        {"transition": "neutral→bull",    "P": 0.66},
        {"transition": "bull→bear",       "P": 0.02},
        {"transition": "bear→neutral",    "P": 0.51},
        {"transition": "neutral→bull",    "P": 0.66},
        {"transition": "bull→bear",       "P": 0.02},
        {"transition": "bear→bull",       "P": 0.45},
        {"transition": "bull→neutral",    "P": 0.51},
        {"transition": "neutral→neutral", "P": 1.00}
    ]
}
```




### Case 12 — 2026-04-15T15:54:37.855Z

**Observed sequence** (trade_id window 1607932430–1607932474):

- `neutral→neutral` P = 1.00 — extended neutral gap
- `neutral→bear`    P ≈ 0.14 — at ~1607932445
- `bear→neutral`    P ≈ 0.51 — at ~1607932446
- `neutral→bull`    P ≈ 0.66 — at ~1607932447
- `bull→neutral`    P ≈ 0.51 — at ~1607932448
- `neutral→bear`    P ≈ 0.14 — at ~1607932449
- `bear→bull`       P ≈ 0.45 — at ~1607932450
- `bull→neutral`    P ≈ 0.51 — at ~1607932451
- `neutral→neutral` P = 1.00 — neutral gap resumes

Inner path appears composite — alternating SHORT and LONG detours inside the same neutral envelope.

![False Start Case 12](screenshot_case12.png)

**Episode sequence** (neutral→neutral → ... → neutral→neutral):

```python
{
    "date": "2026-04-15T15:54:37.855Z",
    "trade_id_window": [1607932430, 1607932474],
    "sequence": [
        {"transition": "neutral→neutral", "P": 1.00},
        {"transition": "neutral→bear",    "P": 0.14},
        {"transition": "bear→neutral",    "P": 0.51},
        {"transition": "neutral→bull",    "P": 0.66},
        {"transition": "bull→neutral",    "P": 0.51},
        {"transition": "neutral→bear",    "P": 0.14},
        {"transition": "bear→bull",       "P": 0.45},
        {"transition": "bull→neutral",    "P": 0.51},
        {"transition": "neutral→neutral", "P": 1.00}
    ]
}
```
