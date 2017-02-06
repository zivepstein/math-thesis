\ProvidesClass{pom_thesis}[1/16/15]

\RequirePackage{graphicx}
\RequirePackage{amsmath}
\RequirePackage{amssymb}
\RequirePackage{color}
\RequirePackage{fullpage}
\RequirePackage{hyperref}
\RequirePackage{amsthm}


\let\maketitle\undefined

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{report}}
\ProcessOptions\relax
\LoadClass{report}

%\newcommand{\qed}{\hfill \mbox{\raggedright \rule{.07in}{.1in}}}
\newcommand{\HRule}{\rule{\linewidth}{0.5mm}}


\newcommand{\advisor}{\gdef\@advisor}
\advisor{Jo Hardin}
\newcommand{\thedate}{\gdef\@thedate}
\thedate{\today}


\numberwithin{equation}{chapter}



\theoremstyle{plain}{
\newtheorem{thm}{Theorem}[chapter]
\newtheorem{lemma}[thm]{Lemma}
\newtheorem{prop}[thm]{Proposition}
\newtheorem{cor}[thm]{Corollary}
\newtheorem{conj}[thm]{Conjecture}
}

\theoremstyle{definition}{
\newtheorem{defn}[thm]{Definition}
\newtheorem{examp}[thm]{Example}
\newtheorem{rmk}[thm]{Remark}
}



%\newenvironment{proof}{\vspace{1ex}\noindent{\bf
%Proof}\hspace{0.5em}}{\hfill\qed\vspace{1ex}}
%\newenvironment{pfof}[1]{\vspace{1ex}\noindent{\bf Proof of
%#1}\hspace{0.5em}}{\hfill\qed\vspace{1ex}}


\def\maketitle{
\begin{center}
\pagenumbering{gobble}
\includegraphics[width=0.5\textwidth]{pomonalogo.pdf}\\[.75cm]

\textsc{\Large Senior Thesis in Mathematics}\\[0.75cm]
% Title
\HRule \\[0.4cm]
% Put your title here
{ \huge \bfseries  \@title}\\[0.4cm]

\HRule \\[1.5cm]

% Author and supervisor
\begin{minipage}{0.4\textwidth}
\begin{flushleft} \large
\emph{Author:} \\
\@author
\end{flushleft}
\end{minipage}
\begin{minipage}{0.4\textwidth}
\begin{flushright} \large
\emph{Advisor:} \\
Dr. \@advisor
\end{flushright}
\end{minipage}
\vfill
 \vspace{\stretch{1.5}}
      {\normalsize
        Submitted to Pomona College in Partial Fulfillment\\
        of the Degree of Bachelor of Arts\par}
%      \vspace{\stretch{1.5}}
 %     Department of Mathematics
\vfill

% Bottom of the page
{\large \@thedate}
\end{center}
}