\begin{table}

\centering

\setlength{\tabcolsep}{2pt}

\footnotesize

\resizebox{\columnwidth}{!}{%

\begin{tabular}{l >{\raggedright\arraybackslash}p{8cm}}

\toprule

\textbf{Category} & \textbf{Features} \\

\midrule

Attention-based (7) & \\
\textit{Model-dependent:} & question\_model\_token\_count \\
 & answer\_model\_token\_count \\
 & attention\_mass\_q\_to\_a\_per\_qtoken \\
 & attention\_mass\_a\_to\_q\_per\_atoken \\
 & focus\_token\_to\_answer\_strength \\
 & answer\_token\_to\_focus\_strength \\
 & focus\_token\_coverage\_ratio \\

\midrule

Lexical Alignment (1) & tfidf\_cosine\_similarity\_q\_a \\

\midrule

Content Word (3) & content\_word\_jaccard\_q\_a \\
 & question\_content\_coverage\_in\_answer \\
 & answer\_content\_word\_ratio \\

\midrule

Surface / Length (1) & answer\_word\_count \\

\midrule

Pattern-based Pragmatics (5) & refusal\_pattern\_match\_count \\
 & clarification\_pattern\_match\_count \\
 & answer\_question\_mark\_count \\
 & answer\_is\_short\_question \\
 & answer\_digit\_groups\_per\_word \\

\midrule

Lexicon-based Pragmatics (2) & answer\_negation\_ratio \\
 & answer\_hedge\_ratio \\

\midrule

Sentiment \& Metadata (6) & question\_sentiment\_polarity \\
 & answer\_sentiment\_polarity \\
 & answer\_char\_per\_sentence \\
 & inaudible \\
 & multiple\_questions \\
 & affirmative\_questions \\

\bottomrule

\end{tabular}}

\captionsetup{

  font={footnotesize,bf},

  skip=2pt

}

\caption{Feature categories and all 25 features used in the pragmatic feature-based classification.}

\label{tab:feature_inventory}

\end{table}
