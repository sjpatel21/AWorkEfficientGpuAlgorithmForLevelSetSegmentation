%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% ucalgarydiss.cls: Derived from the ubcdiss.cls file by Reid holmes. The ubcdiss
%% was originally created by Brian de Alwis. As much as possible the ucalgary 
%% guidelines are adhered to. These guidelines can be found here:
%% http://www.grad.ucalgary.ca/files/custom/t_thesis_guidelines.pdf
%%
%% A LaTeX document class for the revised UBC Thesis Guidelines
%% Originally created by Brian de Alwis.  I developed this separately from
%% Michael McNeil Forbes's ubcthesis.cls/genthesis.class: unlike Forbes' class,
%% this class builds on top of the standard LaTeX book class and should have
%% better compatibility with standard LaTeX packages.
%%
%% FoGS Thesis Specifications and Formatting Requirements
%% <http://www.grad.ubc.ca/students/thesis/index.asp?menu=000,000,000,000>
%% <http://www.grad.ubc.ca/students/thesis/index.asp?menu=005,000,000,000>
%% 
%% Useful documents:
%%  * LaTeX2e for class and package writers
%%    <http://www.latex-project.org/guides/clsguide.pdf>
%%
%% This file may be distributed and/or modified under the
%% conditions of the LaTeX Project Public License, either version 1.2
%% of this license or (at your option) any later version.
%% The latest version of this license is in
%%    http://www.latex-project.org/lppl.txt
%% and version 1.2 or later is part of all distributions of LaTeX
%% version 1999/12/01 or later.
%% 
%% This file may only be distributed together with a copy of this
%% program. You may, however, distribute this program without generated
%% files such as this one.
%% 

%% Random useful links
%% Custom lists: http://www.troubleshooters.com/linux/lyx/ownlists.htm

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{ucalgarydiss}[2008/09/01]

% everybody loves calc
\RequirePackage{calc}
\RequirePackage{ifthen}

%% Useful command stolen from Forbes' genthesis.cls
\newcommand{\@addto}[2]{
  \expandafter\let\expandafter\old\csname#1\endcsname
  \toks1=\expandafter{\old}
  \toks2=\expandafter{#2}
  \expandafter\xdef\csname#1\endcsname{\the\toks1 \the\toks2 }
}

% cause the text of each page to be centred; only effective if using twosided
\newif\ifb@l@nced\b@l@ncedfalse
\newif\iffogscopy\fogscopyfalse

\DeclareOption{balanced}{\b@l@ncedtrue}

% fogscopy: default is to use onesided
\DeclareOption{fogscopy}{
    \fogscopytrue%
    \PassOptionsToClass{oneside}{book}}

% Set the body text to the specified spacing.  Users can explictly specify
% \singlespacing, \onehalfspacing, and \doublespacing in their document
% and restore the default with \textspacing.
\newcommand{\textspacing}{\singlespacing}
\DeclareOption{singlespacing}{%
    \typeout{Using single spacing}%
    \renewcommand{\textspacing}{\singlespacing}}
\DeclareOption{onehalfspacing}{%
    \typeout{Using one-and-a-half spacing}%
    \renewcommand{\textspacing}{\onehalfspacing}}
\DeclareOption{doublespacing}{%
    \typeout{Using double spacing}%
    \renewcommand{\textspacing}{\doublespacing}}

\DeclareOption{draft}{\PassOptionsToClass{draft}{book}}
\DeclareOption{10pt}{\PassOptionsToClass{10pt}{book}}
\DeclareOption{11pt}{\PassOptionsToClass{11pt}{book}}
\DeclareOption{12pt}{\PassOptionsToClass{12pt}{book}}
\DeclareOption{oneside}{\PassOptionsToClass{oneside}{book}}
\DeclareOption{twoside}{\PassOptionsToClass{twoside,openright}{book}}

\ExecuteOptions{11pt,oneside}
\InputIfFileExists{ucalgarydiss.cfg}{}{}
\ProcessOptions\relax

% fix a likely LaTeX bug: seemingly necessary when using book
\newcounter{parentequation}

\LoadClass{book}
\RequirePackage{setspace}

\ifb@l@nced
    \setlength{\oddsidemargin}{\paperwidth-\textwidth}%
    % TeX's margin parameters are implicitly relative to 1in
    \setlength{\evensidemargin}{0.5\oddsidemargin-1in}%
    \setlength{\oddsidemargin}{\evensidemargin}%
\fi

\RequirePackage{textcase}	% for lower- and upper-casing

% tocbibind: cause list of figures, tables in the toc
%   use nottoc: don't want the toc itself in the toc
\RequirePackage[nottoc]{tocbibind}

% Table of Contents must be labelled "Table of Contents"
\renewcommand{\contentsname}{Table of Contents}

% FoGS wants each line of the ToC/LoT/LoF to have dotted leaders
% rtholmes: add the [subfigure] option to prevent some problems getting subfigure to load
\RequirePackage[subfigure]{tocloft}        % typesetting of ToC, LoT, & LoF
\renewcommand{\cftchapdotsep}{\cftdotsep}

% From <http://dcwww.camd.dtu.dk/~schiotz/comp/LatexTips/LatexTips.html#figplacement>
% LaTeX per default only allows a part of the top of a text-page
% (70%) to contain figures, and requires at least 20% of a page to
% be text when text and figures share a page. These parameters should
% be set to more reasonable values, for example 85% and 10%.  Be careful
% not to make \floatpagefraction larger than \topfraction, then you
% risk to produce a figure that can neither go on the top of a text
% page, nor on a page by itself.
\renewcommand{\topfraction}{0.85}
\renewcommand{\textfraction}{0.1}
\renewcommand{\floatpagefraction}{0.75}

% Pages should have no running header, and footer should have
% page number at centre
%\pagestyle{plain}


%% The title page
\makeatletter
\newcommand{\@subtitle}{}
\newcommand{\@previousdegrees}{}
\newcommand{\@degreetitle}{%
    \ClassError{ucalgarydiss}{Missing degree title}{%
	Specify \protect\degreetitle\space in your dissertation
	metadata section.}
    \emph{missing degree title}}
\newcommand{\@institution}{The University of Calgary}
\newcommand{\f@culty}{Faculty of Graduate Studies}
\newcommand{\dep@rtment}{}
\newcommand{\c@mpus}{%
    \ClassError{ucalgarydiss}{Missing campus name}{%
	Specify \protect\campus\{\}\space in your dissertation
	metadata section.}%
    \emph{missing campus}}
\newcommand{\subtitle}[1]{\renewcommand{\@subtitle}{#1}}
\newcommand{\previousdegree}[1]{\@addto{@previousdegrees}{#1\par}}
\newcommand{\degreetitle}[1]{\renewcommand{\@degreetitle}{#1}}
\newcommand{\institution}[1]{\renewcommand{\@institution}{#1}}
\newcommand{\faculty}[1]{\renewcommand{\f@culty}{#1}}
\newcommand{\department}[1]{\renewcommand{\dep@rtment}{#1}}
\newcommand{\program}[1]{\department{#1}}
\newcommand{\campus}[1]{\renewcommand{\c@mpus}{#1}}

% Try to have some compatibility with Forbes' class
\newcommand{\institutionaddress}[1]{%
    \ClassWarning{ucalgarydiss}{Ignoring \protect\institutionaddress:
	use \protect\campus instead}}
\newcommand{\copyrightyear}[1]{%
    \ClassWarning{ucalgarydiss}{Ignoring \protect\copyrightyear:
	use \protect\date instead}}
\newcommand{\submitdate}[1]{%
    \ClassWarning{ucalgarydiss}{Ignoring \protect\submitdate:
	use \protect\date instead}}
\newcommand{\advisor}[1]{%
    \ClassWarning{ucalgarydiss}{Ignoring \protect\advisor}}
\newcommand{\advisortitle}[1]{%
    \ClassWarning{ucalgarydiss}{Ignoring \protect\advisortitle}}

\renewcommand{\maketitle}{%
    % For some reason FoGS thinks the title page should be page 1
    % but \frontmatter causes the page count to be reset.  So
    % we do it here, and forbid use of \frontmatter
    \iffogscopy\frontmatter\fi
%rtholmes    \pagestyle{empty}	% in case of \\PRTSGoublepage in end-titlepage
    \begin{titlepage}
    \null\vfil\par
    \begin{center}
	\setlength{\baselineskip}{1.5em}

		{\large \vspace{-0em}\@institution}\\[2em]
        {\large \@title}\\[3em]

%% if you're using a subtitle, this might be better
%        {\large \@title:}\\[1em]
%        {\large \@subtitle}\\[2em]

	{\large by}\\[3em]
	{\large \@author}\\[4em]
%	\ifx\@previousdegrees\empty\relax\else\@previousdegrees\vspace{2em}\fi

	{\large {A THESIS}}\\[0.5em]
	{\large {SUBMITTED TO THE FACULTY OF GRADUATE STUDIES}}\\[0.5em]
	{\large {IN PARTIAL FULFILMENT OF THE REQUIREMENTS FOR THE}}\\[0.5em]
	{\large {DEGREE OF MASTER OF SCIENCE}}\\[5em]

	{\large {DEPARTMENT OF \dep@rtment}}\\[5em]
%	\ifx\dep@rtment\empty\relax\else(\dep@rtment)\fi
	
%	{\large \textbf{\@degreetitle}}\\[1em]
%	{\large in}\\[1em]
%	{\Large \textsc{\MakeTextLowercase{\f@culty}}}\\

%	\vspace{5em}

	{\large \c@mpus}\\[1em]
%	{\large {\institutionaddress}}[1em]\\
	{\large \@date}\\[3em]

	{\large \textcopyright\ \@author {~~2010}}\\
%	{\tiny Document last updated:~\today~@~\currenttime}
    \end{center}
    \vfill\null

% just a hack to make the back side of the title page blank
%\newpage
%\mbox{}
%\thispagestyle{empty}

    \end{titlepage}
    \iffogscopy\setcounter{page}{2}\else\frontmatter\fi
%    \iffogscopy\setcounter{page}{2}\else\frontmatter\fi
% rtholmes    \pagestyle{plain}
    \renewcommand{\frontmatter}{%
	\ClassError{ucalgarydiss}{frontmatter should not be used}}
}




\newlength{\signwidth}
\newenvironment{signing}[1]{\settowidth{\signwidth}{#1}%     %width of longest signature
\def\signline{ \null\vskip 1em\makebox[\signwidth]{\hrulefill}\\}	%signature line
\def\newsigncolumn{\end{minipage}\hfill\begin{minipage}[t]{\signwidth}}% allow for more than one column of signatures
\begin{singlespace}
% rtholmes
%\setcounter{page}{2}

\pagestyle{plain}
\noindent\begin{minipage}{\textwidth}

\begin{doublespace}
	\begin{center}
		% this is ugly, but it makes it so the institution name lines up height-wise between pg 0 and 1.
%		\vspace{-.45em}
		\vspace{2.85em}
		{\large \@institution}\\
		{\large \f@culty}\\
	\end{center}


\noindent The undersigned certify that they have read, and recommend to the Faculty of Graduate Studies for acceptance, a thesis entitled ``\@title'' submitted by~\@author~in partial fulfillment of the requirements for the degree of Master of Science.

\end{doublespace}
\raggedleft
\null
\vspace{-1em}
\begin{minipage}[t]{\signwidth}
}{\null\end{minipage}\end{minipage}
\null
\par
\vfill
%\begin{flushright}August 28, 2008\end{flushright}
%\vspace{-0.5em}
\noindent \hfill\rule{2.5in}{.01in}\null
\vspace{-2em}
\\
\begin{flushright}
\noindent Date\end{flushright}

\end{singlespace}
}

