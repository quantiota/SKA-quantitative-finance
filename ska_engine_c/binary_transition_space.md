## Binary Transition Space

We believe—like John Archibald Wheeler—that the ultimate foundation of reality is information:

> "It from bit symbolizes the idea that every item of the physical world has at bottom—a very deep bottom, in most instances—an immaterial source and explanation; that what we call reality arises, in the last analysis, from the posing of yes-no questions and the registering of equipment-evoked responses; in short, that all things physical are information-theoretic in origin and that this is a participatory universe."

*John Archibald Wheeler, "Information, Physics, Quantum: The Search for Links" (1989/1990).*



## State Encoding

| State   | Code |
|---------|------|
| neutral | `00` |
| bull    | `01` |
| bear    | `10` |

Code `11` is undefined and never occurs.

---

## Transition Encoding

A transition A→B is a **4-bit word** `[a₁a₀b₁b₀]` (from-state | to-state):

The index is `prev_regime × 3 + regime` where `neutral=0, bull=1, bear=2`:

| Index | Transition       | 4-bit word |
|-------|-----------------|------------|
| 0     | neutral→neutral | `0000`     |
| 1     | neutral→bull    | `0001`     |
| 2     | neutral→bear    | `0010`     |
| 3     | bull→neutral    | `0100`     |
| 4     | bull→bull       | `0101`     | — never observed |
| 5     | bull→bear       | `0110`     |
| 6     | bear→neutral    | `1000`     |
| 7     | bear→bull       | `1001`     |
| 8     | bear→bear       | `1010`     | — never observed |



## Sequence

A sequence `S` is the ordered list of 4-bit words including its `0000` (neutral→neutral) boundaries:

```
S = 0000 a₁ a₂ ... aₖ 0000
```

where each `aᵢ` is a 4-bit transition word and every consecutive pair composes.

The binary code of `S` is the concatenation of all its 4-bit words:

```
code(S) = 0000 a₁ a₂ ... aₖ 0000  =  4(k+2) bits
```

Two sequences are identical if and only if their binary codes are equal. The code is the complete, unambiguous identity of the episode — independent of time, price, and asset.



## Binary Information Flow

The entire market is a continuous binary stream of 4-bit words:

```
... 0000 0000 0000 0010 1000 0001 0100 0010 1001 0100 0000 0000 0001 0100 0000 0000 0010 1001 0110 1001 0100 0000 0000 0000 ...
```

- `0000` — neutral→neutral (silence between episodes)
- any other word — regime transition (episode content)


