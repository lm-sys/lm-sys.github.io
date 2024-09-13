---
title: "Introducing RedTeam Arena"
author: "Anastasios Angelopoulos*, Luca Vivona*, Wei-Lin Chiang*, Aryan Vichare, Lisa Dunlap, Salvivona, Pliny, Ion Stoica"
date: "Sep 13, 2024"
previewImg: /images/blog/redteam_arena/badwords.png
---

We are excited to launch [RedTeam Arena](https://redarena.ai), a community-driven redteaming platform, built in collaboration with [Pliny](https://x.com/elder_plinius) and the [BASI](https://discord.gg/Y6GxC59G) community!



<img src="/images/blog/redteam_arena/badwords.png" style="display:block; margin-top: auto; margin-left: auto; margin-right: auto; margin-bottom: auto; width: 80%"></img>
<p style="color:gray; text-align: center;">Figure 1: RedTeam Arena with Bad Words at <a href="https://redarena.ai">redarena.ai</a></p>

RedTeam Arena is an open-source red-teaming platform for LLMs. Our plan is to provide games that people can play to have fun, while sharpening their red-teaming skills. The first game we created is called *[Bad Words](https://redarena.ai)*, challenging players to convince models to say target "bad words”. It already has strong community adoption, with thousands of users participating and competing for the top spot on the jailbreaker leaderboard.

We plan to open the data after a short responsible disclosure delay. We hope this data will help the community determine the boundaries of AI models—how they can be controlled and convinced.

This is not a bug bounty program, and it is not your grandma’s jailbreak arena. Our goal is to serve and grow the redteaming community. To make this one of the most massive crowdsourced red teaming initiatives of all time. From our perspective, models that are easily persuaded are not worse: they are just more controllable, and less resistant to persuasion. This can be good or bad depending on your use-case; it’s not black-and-white.

We need your help. Join our jailbreaking game at [redarena.ai](redarena.ai). All the code is open-sourced on [Github](https://github.com/redteaming-arena/redteam-arena). You can open issues and also send feedback on [Discord](https://discord.gg/qn2JkWbe). You are welcome to propose new games, or new bad words on X (just tag @[lmsysorg](https://x.com/lmsysorg) and @[elder_plinius](https://x.com/elder_plinius) so we see it)!


## The Leaderboard: Extended Elo

<img src="/images/blog/redteam_arena/leaderboard.png" style="display:block; margin-top: auto; margin-left: auto; margin-right: auto; margin-bottom: auto; width: 100%"></img>
<p style="color:gray; text-align: center;">Figure 2. Leaderboard screenshot. Latest version at <a href="https://redarena.ai/leaderboard">redarena.ai/leaderboard</a></p>

People have been asking how we compute the leaderboard of players, models, and prompts. The idea is to treat every round of Bad Words as a 1v1 game between a player and a (prompt, model) combination, and calculate the corresponding Elo score. Doing this naively is sample-inefficient and would result in slow convergence, so we instead designed a new statistical method for this purpose (writeup coming!) and we’ll describe it below.

*Observation model.* Let $T$ be the number of battles (“time-steps”), $M$ be the number of models, $P$ be the number of players, and $R$ be the number of prompts. For each battle $i \in [n]$, we have a player, a model, and a prompt, encoded as following:

* $X_i^{\rm Model} \in \{0,1\}^M$, a one-hot vector with 1 on the entry of the model sampled in battle $i$.
* $X_i^{\rm Player} \in \{0,1\}^P$, a one-hot vector with 1 on the entry of the player in battle $i$.
* $X_i^{\rm Prompt} \in \{0,1\}^R$, a one-hot vector with 1 on the entry of the prompt sampled in battle $i$.
* $Y_i \in \{0,1\}$, a binary outcome taking the value 1 if the player won (or forfeited) and 0 otherwise.

We then model the win probability of the player as
\begin{equation}
	\mathbb{P}(Y_i = 1) = \frac{e^{X_i^{\rm Player}\beta^{\rm Player}}}{e^{X_i^{\rm Player}\beta^{\rm Player}} + e^{X_i^{\rm Model}\beta^{\rm Model} + X_i^{\rm Prompt}\beta^{\rm Prompt}}}.
\end{equation}
This form might look familiar, since it is the same type of model as the Arena Score: a logistic model. This is just a logistic model with a different, _additive_ structure—the model scores $\beta^{\rm Model}$ and prompt scores $\beta^{\rm Prompt}$ combine additively to generate a notion of total strength for the model-prompt pair. The player scores $\beta^{\rm Player}$ have a similar interpretation as the standard Elo score, and we let $\beta$ denote the concatenation $(\beta^{\rm Player}, \beta^{\rm Model}, \beta^{\rm Prompt})$. For lack of a better term, we call this model “Extended Elo”.

What problem is this new model solving that the old Elo algorithm couldn’t? The answer is in the efficiency of estimation. The standard Elo algorithm could apply in our setting by simply calling every model-prompt pair a distinct “opponent” for the purposes of calculating the leaderboard. However, this approach has two issues: 
It cannot disentangle the effectiveness of the prompt versus that of the model. There is a single coefficient for the pair. Instead, extended Elo can assign _strength to each subpart_.
There are $M\times R$ model-prompt pairs, and only $M+R$ distinct models and prompts. Therefore, asymptotically if $M$ and $R$ grow proportionally, the extended Elo procedure has a quadratic sample-size saving over the standard Elo procedure.


Now, we solve this logistic regression problem _online_. That is, letting $\ell(x,y;\beta)$ be the binary cross-entropy loss, we use the iteration
\begin{equation}
  \beta_n = \beta_{n-1} - \nabla_\beta \ell(X_{n-1}, Y_{n-1}; \beta_{n-1}).
\end{equation}
This is a generalization of the Elo update. In fact, if one removes the prompt coefficient, it reduces exactly to the Elo update between players and models, as if these were 1-1 games.

That’s it! After updating the model coefficients in this way, we report them in the tables in the [RedTeam Arena](https://redarena.ai/leaderboard). We also have more plans for this approach: extended Elo can be used not just for 1v2 leaderboards, like this one, but any $N$v$M$-player leaderboards in order to attribute notions of strength to each subpart using binary human preference feedback.


## What’s next?

[RedTeam Arena](https://redarena.ai) is a community-driven project, and we’re eager to grow it further with your help! Whether through raising Github issues, creating PRs [here](https://github.com/redteaming-arena/redteam-arena), or providing feedback on [Discord](https://discord.gg/qn2JkWbe), we welcome all your contributions!
