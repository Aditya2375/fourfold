# Fourfold

Connect Four with a small policy/value network trained by PUCT self-play, and a browser search inspector. Inspired by the AlphaZero loop, not a reproduction of AlphaZero's architecture, scale or performance. No expert games, solved-game tables or positional heuristics are used by the trained agent.

Run `python -m http.server 8000`, then open http://localhost:8000. Static browser app with same-origin JSON inference, no accounts or remote APIs.

## Reproduce

Python 3.10+: `python -m pip install -r requirements.txt`, `python train.py`, `python evaluate.py`, `python -m unittest -v`, `node test_engine.cjs`. CPU training uses NumPy inference and PyTorch training. Tested with PyTorch 2.14.1+cpu. Exact training weights can vary across library/CPU versions; seeds and every benchmark transcript are included.

42 signed board inputs, 64 ReLU units, 7 policy logits and one tanh value: 3,272 parameters. Board is 6 rows by 7 columns, row 0 at top. Inputs multiply stones by the player about to move. Legal masking before search. Terminal loss is -1 from the next player's view; backups alternate signs. PUCT constant 1.5. Max-score ties choose the leftmost legal column.

Training: seed 311, 240 self-play games, 24 simulations per move. Dirichlet root noise alpha 0.3, fraction 0.25. Sample normalized visits for the first 10 plies, greedy visits afterward. Every 20 games: 120 AdamW updates, learning rate 0.002, weight decay 0.0001, batch up to 128, gradient clip 5. Targets are search proportions and terminal outcome. Policy cross-entropy plus value MSE. Horizontal reflection augmentation. Replay max 12,000 positions; final 4,462 positions. Policy loss 1.7716, value loss 0.6050 are final training-batch means, not held-out metrics.

Evaluation alternates agent first/second, with seeds recorded in model.json. Trained agent at 64 simulations: 60/0/0 vs random, 51/5/4 vs tactical. Untrained network at 64 simulations: 47/1/12 vs tactical. Trained network without search: 49/0/11 vs random. Each sample has 60 games. Tactical opponent takes an immediate win, else blocks an immediate loss, else chooses a random legal move. These small fixed samples do not establish a reliable strength increase or Elo rating. Search does much of the work. No claim of perfect play.

## UI and tests

Play either side, with 64, 192 or 512 simulations. Inspect a move without placing it; undo your whole turn; reset during search safely using request IDs and cancellation. Undoing the opponent's opening move when playing second restarts the opponent's search. Keyboard Tab/Enter column buttons. Board has accessible text, visit/value rows have ARIA labels. Browser worker prevents blocking the UI.

Rule tests cover gravity/full-column rejection, horizontal/vertical/both diagonal wins, no row wrap, immediate-win search and export. JavaScript checks policy/value parity with NumPy (max fixture error 5.4e-8), exact 64-simulation search-fixture parity, immediate-win search. Desktop/mobile renders inspected.

`browser-check.cjs` ships the Chromium interaction/layout tests, including undoing the opening move when playing second. Install Node.js and `npm install --no-save playwright`, then run `node browser-check.cjs`. It uses a local Chrome binary at `/usr/bin/google-chrome`; set `CHROME_PATH` for another location. Screenshots go to `/tmp`. The test server binds to localhost only.

Static security: no secrets, accounts, uploads, telemetry or storage; textContent for dynamic strings; same-origin scripts, worker, checkpoint and CSP; input validation in worker. No paid dependencies. Do not load untrusted model files as code.

Reference: https://arxiv.org/html/1712.01815v1
