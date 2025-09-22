# :ghost: What Lies Beneath: A Call for Distribution-based VQA Datasets :ghost:

Welcome to the repository for the paper "What Lies Beneath: A Call for Distribution-based VQA Datasets", presented at JCDL 2025 (this will be updated with links when applicable).

<!--
<div align="center">
  <a href="TBD"><img src="https://img.shields.io/badge/Paper-arXiv-red" alt="arXiv"></a>
  <a href="https://huggingface.co/datasets/ReadingTimeMachine/visual_qa_multipanel"><img src="https://img.shields.io/badge/Dataset-%F0%9F%A4%97%20Hugging_Face-yellow" alt="Hugging Face"></a>
  <a href="TBD"><img src="https://img.shields.io/badge/JCDL-2025-blue" alt="JCDL 2025"></a>
</div>
-->

<div align="center">
<!--
  <a href="https://huggingface.co/datasets/ReadingTimeMachine/visual_qa_multipanel"><img src="https://img.shields.io/badge/Dataset-%F0%9F%A4%97%20Hugging_Face-yellow" alt="Hugging Face"></a>
-->
  <a href="https://anonymous.4open.science/r/jcdl_visual_qa_histograms-D68A"><img src="https://img.shields.io/badge/Dataset-%F0%9F%A4%97%20Hugging_Face-yellow" alt="Hugging Face"></a>

</div>

(Supplemental Material): A permanent, attributed dataset release will be made upon acceptance.

## Overview

Most VQA datasets focus on real-world images or simple diagrammatic analysis, with few focused on interpreting complex scientific charts. Many VQA datasets for chart analysis don't contain the underlying data behind those charts or assume a 1-to-1 correspondence between chart mark's and underlying data. In reality, charts are transformations (i.e. analysis, simplification, modification) of data. This distinction introduces a reasoning challenge in VQA that the current datasets do not capture.

This paper contributes: 
* An argument for a dedicated VQA benchmark for scientific charts where there is no 1-to-1 correspondence between chart marks and underlying data  
* A review of existing chart VQA datasets with linked data
* Code to generate histograms with a variety of parameters
* A large set of VQA histogram images and questions
* A small exemplary set of VQA images and questions with answers from an LMM (GPT-5) and two human annotators for comparison of humans and LMM VQA capabilities for charts that have an underlying distribution not plotted 1-to-1 on the chart image

## Known issues/TODOs
- Major cleaning of repo needed!
- The ChatGPT-mini may not have all updated VQA questions/answers (not used directly in paper)

## Citation information

If you use this work please cite:
```
TBD
```