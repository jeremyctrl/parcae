<div align="center">

# parcae

<div>
   <a href="https://pypi.org/project/parcae/"><img src="https://img.shields.io/pypi/v/parcae.svg" alt="PyPI"></a>
   <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License"></a>
</div>

<a href="https://heyjeremy.vercel.app/blog/when-do-you-sleep"><img src="https://img.shields.io/badge/blog-when%20do%20you%20sleep%3F%20learning%20a%20pattern%20of%20life%20from%20message%20timestamps-blue" alt="Blog"></a>

Infer daily rhythm and sleep schedule from message timestamps

</div>

`parcae` is a command-line tool and Python library that analyzes nothing but timestamps and infers a user's likely timezone offset and their typical sleep window.

## How It Works

`parcae` models human behavior as a very small Hidden Markov Model with two hidden states:

- Awake (A)
- Sleep (S)

The only observation is "was there at least one message in this time bin?". The model is trained globally across many users to learn:

- how likely people are to send messages while "awake"
- how unlikely they are to send messages while "asleep"
- how often they switch between the two states

At inference time, Parcae:

1. Tries many possible timezone offsets
2. Picks the offset that makes the timeline most explainable by a "human with one long sleep per day"
3. Decodes the most likely sleep/awake sequence
4. Extracts daily sleep blocks
5. Computes a typical schedule and regularity statistics

## Installation

You can install `parcae` using [pipx](https://pypa.github.io/pipx):

```bash
pipx install parcae
```

## Usage

### API

```python
from parcae import Parcae

p = Parcae()

timestamps = [
    "2025-09-01T05:43:12+00:00",
    "2025-09-01T18:22:10+00:00",
    ...
]

print(p.analyze(timestamps))
```

### CLI

Parcae expects a CSV file with one user's timestamps:

```csv
timestamp
2025-09-01T05:43:12+00:00
2025-09-01T07:58:33+00:00
2025-09-01T18:22:10+00:00
```

```bash
parcae analyze user_timestamps.csv
```

Compare two fingerprints to estimate whether they belong to the same person:

```bash
parcae compare parcae:v1:<fp1> parcae:v1:<fp2>
```

#### Examples

```bash
+ Parcae analysis

~ inferred timezone: UTC+3

+ typical schedule:
        - sleep: 23:52 -> 06:34  (≈ 8h 30m)
        - awake: 06:34 -> 23:52
        - variability: ±175m

+ activity profile (24h):
        ▁▁▁▁▁▁▁▁▅▇▅█▆▁▅▄▅▆▁▇▇▆▆▇
        |     |     |     |     
        00    06    12    18    

+ fingerprint:
        parcae:v1:AAAAAAAAAAAAAAAAAAAAAD0AWQA6AGMAQQAAADoAMAA6AEcAAABWAFUATgBMAFsAd__-D9QPqP12BPEBqwU=

~ based on 30 days of data
~ bin size: 15 minutes
```