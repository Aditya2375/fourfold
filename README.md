# Fourfold, edition II

[Open the live laboratory](https://aditya2375.github.io/fourfold/)

A real policy/value network trained by PUCT self-play, a visible search tree trace, legal-position and recorded-match tools, and a separate exact endgame prover. The original checkpoint is retained unchanged. No expert data, solved-game tables or tactical rescue are used by the neural agent.

## Run

```sh
python -m http.server 8000
# Open http://localhost:8000
```

Static same-origin files. All model inference, search and proof run locally in a worker. No accounts, uploads, analytics, storage or private data. JSON checkpoints are data, never executable code. CSP restricts scripts, workers and fetch to the origin.

## What is learned

Both editions have 3,272 parameters: 42 signed board inputs → 64 ReLU → 7 policy logits and one tanh value. Board has six rows and seven columns, row zero at top. Inputs are from the side-to-move's perspective. Legal moves are masked before search. No positional features or expert examples.

- Edition I: original 240 self-play games, 24 simulations per move, 120 optimizer steps per 20 games, learning rate 0.002, replay cap 12,000. Unchanged `model.json` includes its original benchmarks.
- Edition II: fixed 1,200 games, 64 simulations per move, 160 optimizer steps per 20 games, rate 0.001, replay cap 24,000. Final replay has 22,648 positions. Initialization and self-play seed 311. No checkpoint selection.
- Both: AdamW, weight decay 0.0001, batch up to 128, gradient clip 5, reflection augmentation. Policy cross-entropy to root visit proportions plus value MSE to terminal outcome. PUCT 1.5, Dirichlet root noise alpha 0.3 and fraction 0.25, sampled visits in the first ten plies then greedy. Terminal leaf loss −1 from the next player's view. Backups alternate signs. Ties choose the lowest legal column.

This is an AlphaZero-style educational loop, not a reproduction of AlphaZero's architecture, scale or performance. More training, simulations and optimizer settings change together; this is not an isolated causal study of game count.

## Frozen match evidence

100 unique random four-move openings, generated after training with NumPy seed 7621. Each is played twice with colors swapped. Both models use exactly 96 simulations per move and the same PUCT implementation. Board states can overlap self-play. Openings were not used for checkpoint selection.

| Result for edition II | Games |
|---|---:|
| Win | 92 |
| Draw | 34 |
| Loss | 74 |

200 games. Mean outcome (win +1, draw 0, loss −1): 0.090. A 10,000-resample bootstrap over the 100 opening pairs, seed 172, gives 95% interval [−0.015, 0.195]. It includes zero: this study does **not** establish a reliable strength increase. It is not an Elo estimate or solved-game/against-human claim. Shared search code means the opponents are not independent systems. The interface says this next to the score. Every transcript and summary is in `comparison.json`; the page can replay any match.

## Search, not proof

Play either side or both sides. Switch the neural checkpoint at the same position, inspect with 96/256/768 simulations, view raw network value, child Q, legal prior versus visit shares, and an up-to-eight-ply most-visited continuation. Every number comes from the actual search. Neither values nor visit shares are calibrated win probabilities. A continuation is a trace, not a guaranteed plan.

Undo removes a whole human/opponent turn; undoing an opening when playing second correctly restarts the opponent. New game cancels in-flight search with request IDs. Imported sequences enforce gravity, alternating turns, full-column rejection and no moves after a terminal state. UI columns are one-based; stored model/match columns are zero-based.

## Exact endgame prover

A distinct classical tool, never used for training, neural action choice or tactical rescue. At 14 or fewer empty cells it runs full-depth exhaustive negamax with alpha-beta and bound-typed transposition entries, evaluating terminal wins, draws and losses only. It checks each legal root move. A 300,000-node global cap returns unknown for unfinished moves. A proved win can be reported even if other options remain unknown. It never substitutes a heuristic score for a proof. It does not solve the starting board or claim perfect neural play.

Twelve legal late-game positions in `puzzles.json` were generated independently using seed 9871, separate from training. The Python reference uses a memoized exhaustive recurrence without alpha-beta. The JavaScript solver agrees on every legal move, not only the best one. These puzzle answers are not available to the network/search. All options appear only after an explicit proof request. A node cap of one and shallow depth limits are tested for honest unknowns.

## Reproduce

Python 3.10+, NumPy and PyTorch; CPU is sufficient. Tested with PyTorch 2.14.1+cpu. Exact retraining can vary with CPU/library versions.

```sh
python -m pip install -r requirements.txt
python train_refined.py --chunk 1200
python benchmark.py
python puzzles.py
python -m unittest -v
node test_deep.cjs
node test_proof.cjs
```

For bounded runs, repeat `python train_refined.py --chunk 200` until it reports 1,200. Local `selfplay-resume.pt` saves replay, optimizer, model, RNG, history and completed game count every 20 games. Load only a resume created locally; never untrusted pickle. Remove that locally generated file to restart from scratch. `benchmark.py` also resumes via its own generated `benchmark-progress.json`, with RNG and complete color-swapped pairs. Remove that file for a fresh complete run. Final `refined-model.json` includes inference/search fixtures; the benchmark regenerates those fixtures idempotently. Original `train.py` and `evaluate.py` remain for baseline reproduction.

## Verification

Nine Python tests check rule edge cases, both model exports, model size, training history, all independent puzzle proofs, paired opening generation and bootstrap statistics. JavaScript checks NumPy parity (max refined fixture error 1.26e-7), exact 96-simulation search proportions, legal most-visited variations, all 200 transcript outcomes and six complete Python/JS-identical match replays. Solver tests cover every root move for twelve independent endgames, node-limit/depth unknowns, terminal/full-column/input rejection and immediate win.

`browser-check.cjs` exercises first/second/manual play, opponent search, full-turn undo, second-player opening undo, old/new model selection, budgets, inspect, new game during search, legal/invalid imports, terminal controls, exact proofs, match replay/loading, and 320/390px plus desktop layout. Node and Playwright required (`npm install --no-save playwright`). Set `CHROME_PATH` if Chrome is not at `/usr/bin/google-chrome`. Local test server binds only to localhost.

## Reference

[Silver et al., mastering games through self-play](https://arxiv.org/html/1712.01815v1).

Training losses are training-batch measurements, not held-out results. The model can make tactical mistakes and both search and proof have finite budgets. The exact prover's authority is limited to the moves it actually completes.
