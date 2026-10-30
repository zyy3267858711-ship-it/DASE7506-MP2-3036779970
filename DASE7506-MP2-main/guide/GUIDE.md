# MP2 — Selecting Reliable LLM Reasoning

**DASE7506 · Advanced Machine Learning Methods · Fall 2026**

Individual coursework · **10% of the course grade**

**Deadline: 31 October 2026, 23:59 (Beijing time, UTC+8).**

[Submit a result](https://xudongwu-0.github.io/courses/dase7506/mp2/) · [Download the complete repository](https://github.com/xudongwu-0/DASE7506-MP2) · [Installation and running instructions](../code/README.md)

A language model produces four solutions to the same maths problem. They can all sound convincing, show several calculations, and still disagree on the answer. Your task is to build a program that reads them and chooses one.

The language model has already done its part. We provide the questions, its four candidate solutions, and fixed numerical representations of those solutions. You decide how to use that information. The project can be done on CPU, and the repository includes the complete data.

The output is simple: **one candidate index per question**. The interesting part is deciding what evidence makes one answer more believable than another.

## A small example

Suppose the question is:

> A bus travels at 60 km/h. It leaves at 9 am and arrives at noon, including a 30-minute stop. How far does it travel?

Two candidate excerpts might read:

| Candidate | Reasoning and answer |
|---|---|
| A | There are 3 hours between 9 am and noon. Distance = 60 × 3 = **180 km**. |
| B | The bus spends 2.5 hours moving. Distance = 60 × 2.5 = **150 km**. |

Both multiplications are correct. Candidate A makes a mistake before the multiplication: it treats the stop as time spent moving. Candidate B accounts for the stop and reaches the correct answer.

Real candidates can be harder to distinguish. An explanation may contain a bad intermediate step but land on the right final answer; a neat, internally consistent calculation may solve a slightly different problem. Our scorer checks the **final numeric answer**. Keep that distinction in mind when interpreting what your method has learned.

## Questions worth investigating

- **Does reading the reasoning actually help?** Compare methods that use the full solution, the question and final answer, or the supplied representations. Which information accounts for the improvement?
- **What makes a wrong answer look convincing?** Does your method learn useful reasoning patterns, or depend on length, formatting and confident wording? Design an ablation that separates these explanations.
- **Should a solution be judged on its own?** A candidate may look plausible until you compare it with the other three. Can comparing candidates expose a missed condition or a conflicting interpretation?
- **Can the method recognise its own uncertainty?** Are high scores associated with correct choices? What happens on questions where different methods disagree?
- **When does more computation pay off?** A more elaborate method or a combination of methods may improve accuracy, or just make the same mistakes more expensively. Measure which happens in your experiments.

Pick a question you care about, state what you expect to happen, and test it. A failed idea with a clear explanation can teach us more than another model with an unexplained score. Your report should make the connection between your design, the evidence and the conclusion.

## What you will build

You write the selection method and any training code it needs. We supply an empty function interface and an independent evaluator. Training references let you determine which candidate answers are correct; validation is for development and selection. At evaluation, your method sees the question, the candidate texts and their fixed representations, then chooses a candidate without consulting reference answers.

Each group has at least one correct final answer, and more than one candidate can be correct. Any correct choice earns the point. This makes the task about recognising a good answer among available solutions. The published score is selection accuracy on this course bank.

The supplied `submission.py` defines the function signature only and raises `NotImplementedError`. No reference algorithm, training routine or trained checkpoint is supplied. The resource limits are **16 MiB of additional inference assets** and **30 seconds for complete CPU evaluation** on the reference machine; exact measurement rules are below.


## 1. What we provide

- **`guide/`:** this assignment guide, including the task, research questions, assessment and submission rules.
- **`code/`:** the complete data, fixed representations, an empty prediction interface and independent evaluation tools. The [code README](../code/README.md) explains the files, input/output formats, installation and commands.
- **Data:** **4,755 training, 798 validation and 467 test questions**, with four candidate solutions per question. The splits do not overlap. Every group has at least one correct final answer; test groups contain at least two different valid parsed answers. Training and validation also include easier groups.
- **Inputs:** each question, its four candidate texts and fixed numerical representations of shape **[N, 4, 1024]**. Reference answers are provided separately: training supervision, validation development and test scoring/reproduction only.
- **Access:** clone the repository or use GitHub's **Code → Download ZIP**. All supplied data is included. No API key or model download is needed. Run the data checker before developing your method.

## 2. Ranking and assessment

- **Leaderboard metric:** selection accuracy on the **complete test bank**, in percent; **higher is better**. Submit the official scorer's `accuracy` field, between 0 and 100.
- **Scoring:** choose exactly one candidate index from 0 to 3 for each question, in bank order. The fixed scorer checks that candidate's final numeric answer, including equivalent fractions and decimals. Any correct candidate earns the point. It does **not** assess whether every reasoning step is valid.
- **Experimental evidence:** compare your initial and improved methods using the same data splits and evaluation protocol. Ablate a key design choice and report resource use alongside accuracy.
- **Analysis matters:** your report, **at most 10 pages**, is used for final grading. Explain your decisions, what the comparisons show, where the method fails and whether improvements justify their computational cost. An unexplained leaderboard score is not a substitute for analysis.
- **Reproduction:** rankings are provisional until checkpoint reproduction and peer review. The instructor may also evaluate the frozen method on a separate audit bank to check generalization.

## 3. Constraints

- **Training data:** fit only on supplied training inputs and references. Training-only augmentation is allowed. Use validation for development and method selection. Test references are for final scoring and reproduction only; do not use them to train, tune, select or construct lookup tables.
- **Fixed benchmark:** keep the supplied data, numeric parser and scorer unchanged. No external datasets or additional pretrained weights. IDs are for alignment only; no answer lookups or cached predictions.
- **Offline selection:** select one of the supplied candidates. No new LLM generation or API calls during evaluation. Your frozen method must accept new bank inputs without retraining, hard-coded sample counts or assumptions about question IDs.
- **CPU evaluation: at most 30 seconds** on the reference **Xeon Platinum 8457C**, using **four threads in total across all libraries and processes**. Measure one fresh prediction process plus the independent scoring process on the complete test bank. Include startup, imports, asset/data loading, preprocessing and every ensemble member. Report your local hardware; the reference CPU determines eligibility.
- **Inference assets: at most 16 MiB uncompressed in total**, excluding supplied data and ordinary source code. Count all weights, learned transformations/statistics, configurations, indexes and ensemble members; arrays embedded in code also count. Compression or reconstruction at startup does not exempt an asset. One MiB is 2^20 bytes. These limits are intended to encourage improvements in the method rather than large lookup indexes.
- **Training resources:** CPU or one GPU no more powerful than an RTX 4090. GPUs are not provided. Inference must run on CPU. Record seeds, dependencies and training/search costs.
- **AI and reuse:** AI coding assistance is allowed. Understand your implementation, acknowledge reused work and disclose substantive AI help. Architecture, features, training and fusion are open within the rules above.

## 4. Submission and peer review

- **Where and when:** submit through the [MP2 website](https://xudongwu-0.github.io/courses/dase7506/mp2/) by **31 October 2026, 23:59 (UTC+8)**. Early entries may contain only your student ID and score. Your **final entry must include code and checkpoint/configuration links before the deadline**. New scores close automatically at the deadline.
- **Finish the submission:** the website prepares a GitHub issue. Click **Submit on GitHub** and create the issue to save your result; preparing the link alone does not submit anything. Use the same GitHub account and student ID for updates. The leaderboard displays the best eligible score for that pair; previous submissions remain in history.
- **Code link:** provide a publicly accessible repository with code that reproduces your training process and result, your report of at most 10 pages, exact installation/training/evaluation commands, the official evaluation JSON, experiment/resource records, seeds, asset hashes and AI/reuse disclosure. Identify the submitted code version and its matching checkpoint.
- **Checkpoint link:** provide all inference assets needed to reproduce the score **without retraining**. Training-free methods may submit a configuration. Include all learned weights, transformations and statistics; follow the size limit above.
- **Link release:** student IDs and scores are public. Links submitted through the form are encrypted for the instructor until release; GitHub records retain your account. After the deadline, instructors manually release links and frozen scores together for **seven days** of public review.
- **Peer review:** use the submitted code version and matching checkpoint, follow its installation instructions, and run the official evaluator. File a **Peer Review Report** with the submission number and your reproduced score, or email **wu.xudong@connect.hku.hk**. The reproduced score is required; Markdown explanations and evidence links are optional. GitHub reports are public.
- **Rewards and decisions:** confirmed discrepancies can earn rewards. Reports alone never change grades. Installation failures or small numerical differences alone do not establish score falsification.
- <span style="color: #b42318;"><strong>Our automated checking agent runs checkpoint evaluation to verify reported scores, alongside instructor spot checks. Confirmed serious score misreporting will be formally recorded.</strong></span>

## Data attribution

Questions and references derive from [GSM8K](https://github.com/openai/grade-school-math), under its retained MIT license. Candidate solutions and fixed representations were prepared with [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B); its revision is recorded in the data manifest. No model weights are distributed. Keep the supplied license and attribution files when redistributing the package.
