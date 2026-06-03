The Batch Diagnostic Pipeline: Complete Overview (TL;DR)The Core Philosophy: We are moving away from an inefficient "one report per object" pass/fail checklist. The new pipeline produces a single, macro-level Category-Led Batch Report. It groups anomalies by their underlying physical causes, identifies complex physics through coupled flags, and only generates deep-dive plots for objects that truly require human review.1. Energetic & Scaling DeviationsWhat it measures: The fundamental mass, energy, and luminosity budget of the explosion.Luminosity Excess Math: The pipeline calculates the expected plateau magnitude ($M_{exp}$) using a predefined scaling relation based on the inferred $ZAMS/E_k$ ratio. It flags an anomaly if the observed magnitude ($M_{obs}$) deviates significantly: $|M_{obs} - M_{exp}| > 0.75$ mag.Mass Budget Math: Calculates the specific kinetic energy ratio: $E_k / M_{ej}$. If this ratio is $>1.0$, it indicates unphysical ejecta velocities for a standard IIP, triggering a flag.Nickel Overabundance: Evaluates $M_{Ni} / M_{ej}$. If this ratio exceeds standard core-collapse yields (e.g., $>0.01$), the tail luminosity is considered anomalously radioactive.2. Morphological & Light Curve OutliersWhat it measures: Global shape parameters, specifically the length of the hydrogen recombination phase.Plateau Duration Math: Compares the observed plateau length ($t_{obs}$) to the model's expected duration ($t_{exp}$). The pipeline flags the object if the difference is extreme: $|t_{obs} - t_{exp}| > 20$ days.3. Progenitor Environment (Precursor & CSM)What it measures: Pre-explosion eruptive mass loss and the shockwave's initial interaction with that material.Precursor Detection Math: The pipeline takes all photometry prior to $t_{exp} - 10$ days and calculates the rolling baseline mean ($\mu$) and standard deviation ($\sigma$). It scans the immediate pre-explosion window ($t_{exp} - 10$ to $t_{exp}$) and triggers a flag if there are $\ge 2$ consecutive detections where $Flux > \mu + 3\sigma$.Early Rise Excess Math (CSM): Fits a standard fireball expansion curve ($f \propto t^2$) to the first 3 days of data. It calculates the residuals and flags the object if the early observations are $>0.1$ magnitudes brighter than the $t^2$ fit, indicating shock breakout in dense CSM.Arrested Cooling Math: Calculates the rate of color change $\frac{d(g-r)}{dt}$ over the first 15 days. Standard IIPs redden quickly. If the gradient is unusually flat (e.g., $< 0.04$ mag/day), it is flagged for arrested cooling (the ejecta is staying artificially hot and blue).4. Plateau TopographyWhat it measures: Non-standard behavior during the plateau, where the light curve should be relatively flat or steadily declining.Rebrightening Derivative Math: The pipeline calculates the daily rate of change $\frac{dm}{dt}$ during the defined plateau (e.g., days 20 to 70). If the derivative remains strictly negative ($\frac{dm}{dt} < 0$, meaning flux is increasing) for a contiguous span of $>5$ days, it triggers a Rebrightening Bump flag.Linear Residual Math: Fits a simple linear regression to the plateau phase. If a contiguous cluster of residuals peaks $\ge 0.1$ magnitudes above the best-fit line, it is flagged as a topographic anomaly.5. Bayesian Convergence & Parameter HealthWhat it measures: The statistical validity of the MCMC model itself.Prior Piercing Math: Checks if the 1-sigma bounds of the posterior distribution fall within $5\%$ of the hard boundaries defined by the priors. If so, the model is likely hitting a wall.Fractional Uncertainty Math: Calculates the relative error for critical parameters like $t_{exp}$ or $M_{Ni}$ using the formula: $(Upper Bound - Lower Bound) / (2 \times Median)$. If this exceeds $0.5$ ($50\%$), the parameter is flagged as unconstrained.Physically Coupled Anomalies (The Science Flags)The pipeline cross-references the above categories to highlight high-value scientific targets that violate multiple physical rules simultaneously:The "Bright & Slow" Paradox: Flags objects that trigger both a Luminosity Excess (Cat 1) AND an Extreme Extension in plateau length (Cat 2).Environmental Cause & Effect: Flags objects with both a detected $>3\sigma$ Precursor (Cat 3) AND an Early Rise Excess / Arrested Cooling (Cat 3), proving the precursor was an eruptive mass-loss event that the shockwave subsequently hit.

The example latex file is here:
\documentclass[11pt,a4paper]{article}

% Packages
\usepackage[margin=1in]{geometry}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{xcolor}
\usepackage{hyperref}
\usepackage{amsmath}
\usepackage{titlesec}
\usepackage{caption}

% Custom Colors & Hyperlink Setup
\definecolor{outlierred}{RGB}{200,50,50}
\definecolor{warningorange}{RGB}{200,120,0}
\definecolor{headerblue}{RGB}{40,80,150}
\hypersetup{
    colorlinks=true,
    linkcolor=headerblue,
    urlcolor=headerblue,
    pdftitle={Type IIP Batch Diagnostics}
}

% Custom ALeRCE Command
\newcommand{\alerce}[1]{\href{https://alerce.online/object/#1}{#1}}

% Section Formatting
\titleformat{\section}{\Large\bfseries\color{headerblue}}{}{0em}{}[\titlerule]

\begin{document}

% --- TITLE PAGE ---
\begin{center}
    {\Huge \bfseries Type IIP Supernova Diagnostics} \\[0.5cm]
    {\Large Automated Batch Anomaly Report} \\[0.2cm]
    {\normalsize Run Date: \today \quad | \quad Total Objects Analyzed: \textbf{142}}
\end{center}

\vspace{0.5cm}
\tableofcontents
\clearpage

% ==========================================
% PART 1: CATEGORY LEDGERS
% ==========================================
\section{I. Energetic \& Scaling Deviations}
\textit{Objects exhibiting severe luminosity excesses or budget violations.}

\begin{table}[h]
\centering
\begin{tabular}{@{}llccl@{}}
\toprule
\textbf{Object ID} & \textbf{Obs. Mag} & \textbf{Exp. Mag} & \textbf{ZAMS/$E_k$} & \textbf{Primary Flag} \\ \midrule
\alerce{ZTF25absffwy} & -17.33 & -16.47 & 12.78 & \textcolor{outlierred}{Luminosity Excess (+0.86)} \\
\bottomrule
\end{tabular}
\end{table}

% (Your Python script would inject Categories II, III, IV, and V here following the same table structure from the previous template, just using the \alerce{} command for the OIDs).

\clearpage

% ==========================================
% PART 2: COUPLED PHYSICAL ANOMALIES
% ==========================================
\section{Physically Coupled Anomalies}
\textit{Objects triggering multiple anomaly categories that suggest complex, non-standard physics.}

\subsection*{A. The "Bright \& Slow" Paradox}
\textit{Objects with extreme brightness combined with unexpectedly long plateaus. Indicates massive envelopes or sustained secondary power sources.}
\begin{table}[h]
\centering
\begin{tabular}{@{}lccc@{}}
\toprule
\textbf{Object ID} & \textbf{Mag Excess} & \textbf{Plateau Excess} & \textbf{Link to Plots} \\ \midrule
\alerce{ZTF25absffwy} & +0.86 mag & +64.7 days & \hyperref[sec:ZTF25absffwy]{See Profile} \\
\bottomrule
\end{tabular}
\end{table}

\subsection*{B. Environmental Cause \& Effect}
\textit{Objects showing both $>3\sigma$ precursor mass-loss and early-time shock breakout excesses.}
\begin{table}[h]
\centering
\begin{tabular}{@{}lccc@{}}
\toprule
\textbf{Object ID} & \textbf{Precursor} & \textbf{Rise Excess} & \textbf{Link to Plots} \\ \midrule
\alerce{ZTF25abccdde} & Day -12 Bump & +0.5 mag & \hyperref[sec:ZTF25abccdde]{See Profile} \\
\bottomrule
\end{tabular}
\end{table}

\clearpage

% ==========================================
% PART 3: FLAGGED OBJECT DEEP DIVES
% ==========================================
\section{Flagged Object Profiles}
\textit{Detailed MCMC fits, light curves, and diagnostic reasoning for all flagged objects.}

% --- PYTHON LOOP STARTS HERE ---

\subsection{\alerce{ZTF25absffwy}} \label{sec:ZTF25absffwy}

\textbf{Primary Anomalies:} Energetic Deviation (Severe), Morphological Outlier (Extreme Extension). \\
\textbf{Diagnostic Reasoning:} This object was flagged because its observed brightness (-17.33) is 0.86 magnitudes brighter than expected based on its inferred $ZAMS/E_k$ ratio of 12.78. Coupled with a massive +64.7 day extension to its expected plateau duration, this object violates standard recombination cooling models. 

\vspace{0.5cm}

\begin{figure}[h]
    \centering
    \begin{minipage}{0.48\textwidth}
        \centering
        % \includegraphics[width=\linewidth]{lc_ZTF25absffwy.png}
        \includegraphics[width=\linewidth]{example-image-a} % Placeholder
        \caption*{Best-Fit Light Curve}
    \end{minipage}\hfill
    \begin{minipage}{0.48\textwidth}
        \centering
        % \includegraphics[width=\linewidth]{corner_ZTF25absffwy.png}
        \includegraphics[width=\linewidth]{example-image-b} % Placeholder
        \caption*{Posterior Corner Plot}
    \end{minipage}
\end{figure}

\vspace{0.5cm}
\hrule
\vspace{0.5cm}

% --- PYTHON LOOP ENDS HERE ---

\clearpage

% ==========================================
% APPENDIX
% ==========================================
\section*{Appendix A: Filtered Non-IIP Objects}
\addcontentsline{toc}{section}{Appendix A: Filtered Non-IIP Objects}
\textit{The following objects were processed but ultimately rejected by the classification logic as they do not conform to baseline Type IIP parameters.}

\begin{table}[h]
\centering
\begin{tabular}{@{}lll@{}}
\toprule
\textbf{Object ID} & \textbf{Alerce Link} & \textbf{Rejection Reason} \\ \midrule
ZTF24abcdef & \alerce{ZTF24abcdef} & No visible plateau ($>0.05$ mag/day linear decline) \\
ZTF25qwerty & \alerce{ZTF25qwerty} & Rise time too long ($>20$ days, possible IIn) \\
\bottomrule
\end{tabular}
\end{table}

\end{document}

Note: This is a template. You need to follow it for each section as mentioned in the comment and actually populate it with code outputs