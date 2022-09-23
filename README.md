# ConLSTM-CRF

An implementation based on the ConLSTM-CRF model described in:
Olafsson, S., Wallace, B. C., & Bickmore, T. W. (2020, May). Towards a Computational Framework for Automating Substance Use Counseling with Virtual Agents. In AAMAS (pp. 966-974).

Expected format for the data:

| Section	| Tag |	Speaker	| Text |
|---|---|---|--|
| 1	| x |	A |	The quick brown fox |
| 1	| y	| A	| The quick brown fox      |
| 1	| z	| B |	Jumps over the lazy frog |
| 1	| x	| A	| The quick brown fox |
| 1	| y	| B	| Jumps over the lazy frog |
| 2	| z	| B	| Jumps over the lazy frog |
| 2 |	x	| A	| The quick brown fox |
| 2	| y	| B	| Jumps over the lazy frog |
| 2	| z |	A	| The quick brown fox |
| 2	| x	| B	| Jumps over the lazy frog |
...
