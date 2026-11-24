# ToneTwist AFx Dataset

## Universal Audio 6176 Vintage Channel Strip - 610B Mic Preamp + 1176LN Limiter

### Dry

Dry inputs are from:

- [IDMT-SMT-Audio-Effects](https://www.idmt.fraunhofer.de/en/business_units/m2d/smt/audio_effects.html)

### Settings - 610B Mic Preamp

| Gain | Level | Input | High | Low |
|---|---|---|---|---|
| 10dB | 6 | Line | 0dB | 0dB |

From source publication:

- 610B Mic Preamp: This microphone tube preamplifier (preamp) is sampled from a 6176 Vintage Channel Strip unit. In order to obtain an output signal with high harmonic distortion, the preamp is overdriven with the following settings: gain +10 dB, level 6, line impedance and high and low boost/cut 0 dB.

### Settings - 1176LN Limiter

| Attack | Release | Input | Output | Ratio |
|---|---|---|---|---|
| 10 | 10 | 4 | 7 | All |

From source publication:

- 1176LN Limiter: Similarly, the wildly used field-effect transistor limiter 1176LN is sampled from the same 6176 Vintage Channel Strip unit. The limiter samples are recorded with the following settings: attack 800us, release 1100 ms, input level 4, output level 7 and ratio ALL. We use the slowest attack and release settings in order to further test the long-term memory of the models. The compression ratio value of ALL corresponds to all the ratio buttons of an original 1176 being pushed simultaneously. Thus, this setting also introduces distortion due to the variation of attack and release times.

### Github

[https://github.com/mcomunita/tonetwist-afx-dataset](https://github.com/mcomunita/tonetwist-afx-dataset)

### Source

Data used in the following publication:

```@article{martinez2020deep,
  title={Deep learning for black-box modeling of audio effects},
  author={Mart{\'\i}nez Ram{\'\i}rez, Marco A and Benetos, Emmanouil and Reiss, Joshua D},
  journal={Applied Sciences},
  volume={10},
  number={2},
  pages={638},
  year={2020},
  publisher={MDPI}}
```

### Links

web: [https://mchijmma.github.io/DL-AFx/](https://mchijmma.github.io/DL-AFx)

code: N.A.

original data: [https://zenodo.org/records/3562442](https://zenodo.org/records/3562442)