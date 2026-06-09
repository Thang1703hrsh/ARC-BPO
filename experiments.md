\section{Experiments}

\subsection{Experimental Setup}
\label{subsec:exp_setup}

\textbf{Data:}
We follow the standard offline preference-optimization protocol and train on publicly curated pairwise preference data. Each backbone is initialized from a released supervised fine-tuning (SFT) checkpoint and then optimized on its corresponding preference corpus. Mistral-7B-v0.1 is initialized from \texttt{mistral-7b-sft-alpha} and trained on UltraFeedback Binarized~\cite{cui2310ultrafeedback}; Llama-3-8B is initialized from \texttt{LLaMA3-SFT-v2} and trained on Llama3-UltraFeedback-ArmoRM~\cite{cui2310ultrafeedback}; and Qwen2.5-7B-Instruct is optimized on UltraFeedback Binarized starting from the released instruct checkpoint. All methods share the same frozen reference policy $\pref$ in each setting, taken as the corresponding initialization, so that differences in performance are attributable to the preference objective rather than to the reference.

\textbf{Models:}
We evaluate three backbones spanning a base 7B model, a base 8B model, and a strongly instruction-tuned 7B model: Mistral-7B-v0.1, Llama-3-8B, and Qwen2.5-7B-Instruct. This range lets us assess whether the chunk-level signal remains effective across model scale and across the gap between weakly and strongly aligned starting points.

\textbf{Chunking:}
ARC-BPO partitions each response with the fixed deterministic chunker described in Section~\ref{subsec:setup}. The chunker is computed once per example and frozen for the entire run. Boundaries are placed at sentence and clause terminators, discourse connectives, list and enumeration markers, and structural delimiters such as line breaks, Markdown blocks, and code fences. A minimum chunk length of $4$ tokens, with sub-floor fragments merged, and a maximum of $64$ tokens, with over-long units split, keep the units numerically comparable. Every boundary is snapped to a model-tokenizer boundary so the chunk spans are exact. The chunker introduces no learnable parameters and no policy dependence.

\textbf{Training:}
All methods are trained for one epoch with RMSProp and a cosine schedule, using identical optimization hyperparameters across objectives to isolate the effect of the loss. The shared training configuration and ARC-BPO-specific hyperparameters are summarized in Table~\ref{tab:training_config}. Since ARC-BPO uses data-anchored one-sided chunk targets, its method-specific hyperparameters are the finite margin source $\Delta^\star$, the detached advantage-shape temperature $T$, the winsorization radius $\kappa$, and the SBA parameters $(\lambda,s)$.

\begin{table}[htbp]
\centering
\small
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.08}
\caption{Training configuration for ARC-BPO. The lower block lists the ARC-BPO-specific hyperparameters introduced in Section~\ref{sec:method}. Values marked as \texttt{TBD} should be replaced by the final validated setting.}
\label{tab:training_config}
\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lccc@{}}
\toprule
\textbf{Setting} & \textbf{Mistral-7B-v0.1} & \textbf{Llama-3-8B} & \textbf{Qwen2.5-7B-Instruct} \\
\midrule
Epochs        & $1$ & $1$ & $1$ \\
Optimizer     & RMSProp & RMSProp & RMSProp \\
Hardware      & $4{\times}$H100 & $4{\times}$H100 & $4{\times}$H100 \\
Batch size    & $32$ & $32$ & $32$ \\
Grad. accum.  & $4$ & $4$ & $4$ \\
LR            & $5{\times}10^{-7}$ & $5{\times}10^{-7}$ & $5{\times}10^{-7}$ \\
LR schedule   & cosine & cosine & cosine \\
Warmup ratio  & $0.05$ & $0.05$ & $0.05$ \\
Max length    & $2048$ & $2048$ & $2048$ \\
\midrule
Pref. scale $\beta$          & $0.1$ & $0.1$ & $0.1$ \\
Margin source $\Delta^\star$ & fixed $\tau_0$ & fixed $\tau_0$ & fixed $\tau_0$ \\
Shape temperature $T$        & \texttt{TBD} & \texttt{TBD} & \texttt{TBD} \\
Winsor radius $\kappa$       & \texttt{TBD} & \texttt{TBD} & \texttt{TBD} \\
SBA exponent $\lambda$       & $1.0$ & $1.0$ & $1.0$ \\
SBA scale $s$                & $4$ & $4$ & $4$ \\
Advantage proxy $\widehat A$ & detached & detached & detached \\
\bottomrule
\end{tabular*}
\end{table}

\textbf{Baselines:}
We compare against two families of preference-optimization methods that share ARC-BPO's offline pairwise data regime. The sequence-level family attaches a single preference signal to an entire response and includes DPO~\cite{rafailov2023direct} and Bregman Preference Optimization (BPO-SBA)~\cite{kim2026preference}. The token-level family distributes supervision across individual tokens and includes TDPO~\cite{zeng2024token}, TIS-DPO~\cite{liu2025tis}, TI-DPO~\cite{yang2025token}, and the two variants of Token-level Bregman Preference Optimization, namely TBPO-Q and TBPO-A~\cite{nguyen2026tokenratio}. TBPO is the most direct comparison: ARC-BPO keeps the fine-grained Bregman ratio-matching goal but replaces cross-state token comparisons with one-sided, data-anchored chunk targets. This removes token-to-token alignment, length-mismatch issues, and the need to estimate the correction term $w_t$. We additionally report the SFT or base checkpoint as the initialization reference. Unless noted otherwise, all baselines use the hyperparameters of their official implementations.

\textbf{Evaluation:}
We assess general capability with the HuggingFace Open LLM Leaderboard~\cite{huggingface_open_llm_leaderboard_v1} suite under the Language Model Evaluation Harness~\cite{eleutherai_lm_eval_harness_concept}, reporting per-task accuracy and the six-task average over HellaSwag, ARC, MMLU, TruthfulQA, Winogrande, and GSM8k. The few-shot counts and metrics follow the standard leaderboard protocol and are summarized in Table~\ref{tab:benchmark_details}. Open-ended generation quality is assessed by pairwise MT-Bench~\cite{zheng2023judging} comparisons with an LLM judge, reported as win/tie/loss percentages.

\begin{table}[ht]
\centering
\small
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{1.12}
\caption{Evaluation tasks used in the Open LLM Leaderboard setting.}
\label{tab:benchmark_details}
\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lccc@{}}
\toprule
\textbf{Dataset} & \textbf{HellaSwag} & \textbf{ARC} & \textbf{MMLU} \\
\midrule
\# Few-shot & 10 & 25 & 5 \\
Metric & \texttt{acc\_norm} & \texttt{acc\_norm} & \texttt{acc} \\
\midrule
\textbf{Dataset} & \textbf{TruthfulQA} & \textbf{Winogrande} & \textbf{GSM8k} \\
\midrule
\# Few-shot & 0 & 5 & 5 \\
Metric & \texttt{mc2} & \texttt{acc} & \texttt{acc} \\
\bottomrule
\end{tabular*}
\end{table}

\subsection{Main Results on Preference Alignment}
\label{subsec:alignment_results}

Tables~\ref{tab:alignment_mistral},~\ref{tab:alignment_llama3}, and~\ref{tab:alignment_qwen25} report the main alignment results across the three backbones. For each table, the best completed score per column is shown in \textbf{bold}. The ARC-BPO rows are left blank in this draft and should be filled only with final ARC-BPO runs.

% TODO: Fill ARC-BPO rows with final ARC-BPO results.

\begin{table*}[t]
\centering
\small
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.08}
\resizebox{\textwidth}{!}{%
\begin{tabular}{l|ccccccc}
\toprule
\textbf{Method} 
& \textbf{HellaSwag} 
& \textbf{ARC} 
& \textbf{MMLU} 
& \textbf{TruthfulQA} 
& \textbf{Winogrande} 
& \textbf{GSM8k} 
& \textbf{Avg.} \\
\midrule
\multicolumn{8}{c}{\textbf{Mistral-7B-v0.1}} \\
\midrule
SFT      & 80.72 & 55.54 & 58.41 & 43.67 & 76.71 & 18.49 & 55.59 \\
\midrule
DPO      & 82.39 & 59.40 & 59.15 & 42.30 & 77.74 & 34.87 & 59.30 \\
BPO      & 81.03 & 57.67 & 59.16 & 41.71 & 77.26 & 32.90 & 58.29 \\
TDPO     & 82.70 & \textbf{61.54} & 57.17 & 43.48 & 76.95 & 28.65 & 58.41 \\
TIS-DPO  & \textbf{83.10} & 59.09 & 57.81 & \textbf{45.91} & 77.66 & 34.42 & 59.66 \\
TI-DPO   & 81.95 & 60.12 & 58.30 & 45.10 & 77.10 & 33.50 & 59.35 \\
TBPO-Q   & 82.72 & 59.74 & \textbf{60.82} & 44.47 & \textbf{78.58} & \textbf{39.34} & \textbf{60.95} \\
TBPO-A   & 82.74 & 59.47 & 60.54 & 44.40 & 78.21 & 39.04 & 60.73 \\
\midrule
\rowcolor{gray!20}
\textbf{ARC-BPO} 
& -- & -- & -- & -- & -- & -- & -- \\
\bottomrule
\end{tabular}%
}
\caption{Main alignment results on the Open LLM Leaderboard using \textbf{Mistral-7B-v0.1} as the trained backbone. Higher scores are better, and the best completed result in each column is shown in bold.}
\label{tab:alignment_mistral}
\end{table*}

\begin{table*}[t]
\centering
\small
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.08}
\resizebox{\textwidth}{!}{%
\begin{tabular}{l|ccccccc}
\toprule
\textbf{Method} 
& \textbf{HellaSwag} 
& \textbf{ARC} 
& \textbf{MMLU} 
& \textbf{TruthfulQA} 
& \textbf{Winogrande} 
& \textbf{GSM8k} 
& \textbf{Avg.} \\
\midrule
\multicolumn{8}{c}{\textbf{Llama-3-8B}} \\
\midrule
SFT      & 79.46 & 51.45 & 61.19 & 46.46 & 75.69 & 68.67 & 63.82 \\
\midrule
DPO      & 81.23 & 56.05 & 61.70 & 48.47 & 75.84 & 74.20 & 66.24 \\
BPO      & 81.68 & 54.95 & 61.48 & 47.88 & 75.13 & 73.22 & 65.72 \\
TDPO     & \textbf{83.29} & 59.04 & 61.86 & 51.76 & 76.55 & 75.10 & 67.91 \\
TIS-DPO  & 81.37 & 58.70 & 60.77 & 49.87 & 75.14 & 75.12 & 66.82 \\
TI-DPO   & 80.85 & 58.20 & 62.10 & 50.95 & 76.10 & 74.80 & 67.17 \\
TBPO-Q   & 81.90 & 64.07 & \textbf{64.92} & \textbf{53.36} & 79.08 & 78.92 & 70.38 \\
TBPO-A   & 81.97 & \textbf{64.08} & 64.84 & 53.29 & \textbf{79.63} & \textbf{79.22} & \textbf{70.50} \\
\midrule
\rowcolor{gray!20}
\textbf{ARC-BPO} 
& -- & -- & -- & -- & -- & -- & -- \\
\bottomrule
\end{tabular}%
}
\caption{Main alignment results on the Open LLM Leaderboard using \textbf{Llama-3-8B} as the trained backbone. Higher scores are better, and the best completed result in each column is shown in bold.}
\label{tab:alignment_llama3}
\end{table*}

\begin{table*}[t]
\centering
\small
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.08}
\resizebox{\textwidth}{!}{%
\begin{tabular}{l|ccccccc}
\toprule
\textbf{Method} 
& \textbf{HellaSwag} 
& \textbf{ARC} 
& \textbf{MMLU} 
& \textbf{TruthfulQA} 
& \textbf{Winogrande} 
& \textbf{GSM8k} 
& \textbf{Avg.} \\
\midrule
\multicolumn{8}{c}{\textbf{Qwen2.5-7B-Instruct}} \\
\midrule
Base     & 81.45 & 65.96 & 73.49 & 64.70 & 75.06 & 68.61 & 71.55 \\
\midrule
DPO      & 81.67 & 66.12 & 73.02 & 63.50 & 74.19 & 67.49 & 71.00 \\
BPO      & 81.80 & 66.40 & 73.60 & 64.10 & 75.20 & 70.50 & 71.93 \\
TDPO     & 81.55 & 65.90 & 72.85 & 63.22 & 74.05 & 67.20 & 70.80 \\
TIS-DPO  & 81.88 & 66.60 & 73.42 & 64.30 & 74.95 & 68.45 & 71.60 \\
TI-DPO   & 80.31 & 66.04 & \textbf{74.68} & \textbf{68.58} & 74.09 & 58.74 & 70.41 \\
TBPO-Q   & \textbf{82.10} & \textbf{67.20} & 74.40 & 66.50 & 78.20 & 73.80 & 73.70 \\
TBPO-A   & 81.95 & 67.10 & 74.30 & 66.69 & \textbf{78.30} & \textbf{73.93} & \textbf{73.71} \\
\midrule
\rowcolor{gray!20}
\textbf{ARC-BPO} 
& -- & -- & -- & -- & -- & -- & -- \\
\bottomrule
\end{tabular}%
}
\caption{Main alignment results on the Open LLM Leaderboard using \textbf{Qwen2.5-7B-Instruct} as the trained backbone. Higher scores are better, and the best completed result in each column is shown in bold.}
\label{tab:alignment_qwen25}
\end{table*}