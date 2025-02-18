# Awesome-LM-Science-Bench

> An open benchmark list covers LLM's reasoning benchmark for science problems, we focus on LLM evaluation datasets in natural sciences.

Hi👋, if you find this repo helpful, welcome to give a star ⭐️!

As many benchmarks are being released, we will update this repo frequently and **welcome contributions** from the 🏠community!

(last update: Feb 2025)

---

## General Science & Multidisciplinary Benchmarks

### SciEx [2024 June]
* **Description:** A multilingual, multimodal benchmark using university computer science exam questions. Includes free-form questions with images and varying difficulty.
* **Purpose:** Assesses LLMs' ability to handle scientific tasks in university exams, including algorithm writing, database querying, and mathematical proofs.
* **Relevance:**  Essential for evaluating LLMs in academic and research settings, with human expert grading provided for performance evaluation.
* **Performance:**  Even top LLMs face challenges with free-form exams in SciEx, indicating ongoing areas for development.
* **Source:** [SciEx Benchmark](https://www.scienceex.ai/) [arXiv](https://arxiv.org/abs/2406.10421)

### SciBench [2023 July]
* **Description:** A benchmark suite for evaluating college-level scientific problem-solving abilities, featuring problems from mathematics, chemistry, and physics.
* **Purpose:**  To rigorously test LLMs' reasoning on complex scientific problems at the university level.
* **Relevance:** Vital for pushing the boundaries of LLMs in scientific research and discovery, highlighting areas for improvement in advanced reasoning.
* **Results:**  Current LLMs show limited performance, indicating substantial room for improvement in collegiate-level scientific problem-solving.
* **Source:** [arXiv:2307.10635](https://doi.org/10.48550/arXiv.2307.10635)

### SciKnowEval [2024 June]
* **Description:** A benchmark designed to evaluate LLMs across five levels of scientific knowledge, from memory to reasoning, in chemistry and physics. It includes 70,000 scientific problems.
* **Purpose:** Establishes a framework for systematically assessing the depth of scientific knowledge in LLMs.
* **Relevance:** Essential for the detailed evaluation of LLMs in scientific domains, aiming to standardize scientific knowledge benchmarking.
* **Source:** [arXiv:2406.09098](https://arxiv.org/html/2406.09098v3)

### Advanced Reasoning Benchmark (ARB) [2023]
* **Description:**  Focuses on advanced reasoning problems across disciplines like physics and chemistry.
* **Purpose:**  Assesses LLMs' logical deduction and complex problem-solving capabilities in scientific contexts.
* **Relevance:**  Crucial for evaluating the inferential abilities of LLMs in scientific reasoning.
* **Source:** [OpenReview: ARB](https://openreview.net/forum?id=gsZAtAdzkY)

### Massive Multitask Language Understanding (MMLU) [2020 September]
* **Description:** Measures general knowledge across 57 diverse subjects, spanning STEM, social sciences, and humanities.
* **Purpose:** Evaluates LLMs' broad understanding and reasoning capabilities across a wide array of disciplines.
* **Relevance:** Suitable for assessing AI systems requiring extensive world knowledge and versatile problem-solving skills.
* **Source:** [Measuring Massive Multitask Language Understanding](https://arxiv.org/abs/2009.03300)
* **Resources:**
    * [MMLU GitHub](https://github.com/hendrycks/test)
    * [MMLU Dataset](https://people.eecs.berkeley.edu/~hendrycks/data.tar)

### General Language Understanding Evaluation (GLUE) [2018 April]
* **Description:** A benchmark suite comprising nine diverse natural language understanding tasks. These tasks include single-sentence analysis, similarity and paraphrasing detection, and natural language inference.
* **Purpose:** Designed to provide a comprehensive evaluation of language models' ability to understand natural language across different tasks and datasets.
* **Relevance:** Crucial for developing and evaluating NLP systems intended for broad language processing applications like chatbots and content analysis.
* **Source:** [GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding](https://arxiv.org/abs/1804.07461)
* **Resources:**
    * [GLUE Homepage](https://gluebenchmark.com/)
    * [GLUE Dataset: HuggingFace](https://huggingface.co/datasets/glue)
    * [GLUE Benchmark - Kaggle](https://www.kaggle.com/datasets/xhlulu/general-language-understanding-evaluation)

### AI2 Reasoning Challenge (ARC) [2018 March]
* **Description:**  A question-answering dataset from grade 3 to grade 9 science exams, featuring multiple-choice questions that demand reasoning. It includes both an "Easy" set and a more challenging "Challenge" set with questions requiring deeper inference.
* **Purpose:**  Tests LLMs' ability to answer complex science questions that require logical reasoning and common-sense knowledge.
* **Relevance:**  Valuable for educational AI applications, particularly in automated tutoring systems and knowledge assessment for grade-school science.
* **Source:** [Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge](https://arxiv.org/abs/1803.05457)
* **Resources:**
    * [ARC Dataset: HuggingFace](https://huggingface.co/datasets/ai2_arc)
    * [ARC Dataset: Allen Institute](https://allenai.org/data/arc)
    * [ARC Leaderboard](https://leaderboard.allenai.org/arc/submissions/public)

### SciQ [2017 July]
* **Description:** A dataset of 13,679 multiple-choice science exam questions, primarily in physics, chemistry, and biology. Many questions include supporting evidence to aid in answering.
* **Purpose:** Tests LLMs on science-based question answering, evaluating their ability to utilize provided context and scientific knowledge.
* **Relevance:** Useful for developing educational tools and platforms for science education and knowledge assessment.
* **Source:** [Crowdsourcing Multiple Choice Science Questions](https://arxiv.org/abs/1707.06209)
* **Resources:**
    * [SciQ Dataset: HuggingFace](https://huggingface.co/datasets/sciq)
    * [SciQ Benchmark (Text Generation) - Papers With Code](https://paperswithcode.com/sota/text-generation-on-sciq)


## Biology Benchmarks

### LAB-Bench: Language Agent Biology Benchmark [2024 July]
* **Description:** A dataset of over 2,400 multiple-choice questions for biology research capabilities, covering literature recall, figure interpretation, database navigation, and sequence manipulation.
* **Purpose:**  Evaluates AI systems on practical biology research tasks, aiming to develop AI assistants for scientific research.
* **Relevance:**  Crucial for accelerating scientific discovery by enhancing LLMs in biology-related research tasks. Performance is compared against human biology experts.
* **Source:** [LAB-Bench: Measuring Capabilities of Language Models for Biology](https://arxiv.org/abs/2407.10362)
* **Resources:**
    * [LAB-Bench GitHub](https://github.com/Future-House/LAB-Bench)

### BioLLMBench [2023 December]
* **Description:** A benchmarking framework for evaluating LLMs in bioinformatics tasks.
* **Purpose:**  Comprehensively assesses LLMs' capabilities in bioinformatics and biological reasoning.
* **Relevance:**  Useful for evaluating and comparing LLMs in biological research and data analysis contexts.
* **Source:** [BioLLMBench: A Comprehensive Benchmarking of Large Language Models in Biology](https://www.biorxiv.org/content/10.1101/2023.12.19.572483v1)

## Chemistry Benchmarks

### ChemQA [2024]
* **Description:** A multimodal question-answering dataset focused on chemistry reasoning, featuring 5 QA tasks.
* **Purpose:** Evaluates LLMs on chemistry-specific tasks like atom counting, molecular weight calculation, and retrosynthesis planning.
* **Relevance:**  Essential for AI applications in chemistry education, research, and complex chemical problem-solving.
* **Source:** [GitHub - materials-data-facility/matchem-llm](https://github.com/materials-data-facility/matchem-llm)
* **Resources:**
    * [ChemQA Dataset: HuggingFace](https://huggingface.co/datasets/shangzhu/chemqa)

### ChemBench [2024]
* **Description:**  Features over 7000 questions covering a wide range of chemistry topics.
* **Purpose:**  Assesses LLMs' chemistry knowledge and reasoning skills across various chemical domains.
* **Relevance:**  Important for evaluating AI systems designed for chemistry education and research.
* **Source:** [GitHub - materials-data-facility/matchem-llm](https://github.com/materials-data-facility/matchem-llm)

### ChemSafetyBench: LLM Safety in Chemistry [2024 November]
* **Description:**  A benchmark specifically designed to evaluate the safety aspects of LLMs in chemistry-related contexts.
* **Purpose:**  Assesses the safety and reliability of LLMs for chemistry applications, focusing on preventing harmful outputs.
* **Relevance:** Crucial for ensuring the responsible and safe deployment of LLMs in chemistry and related fields.
* **Source:** [ChemSafetyBench: Benchmarking LLM Safety on Chemistry](https://arxiv.org/pdf/2411.16736)

### ChemLLMBench [2024 - NeurIPS 2023 Datasets and Benchmarks Track]
* **Description:** A comprehensive benchmark covering eight distinct chemistry tasks.
* **Purpose:** Provides a thorough evaluation of LLMs' capabilities across different chemistry-related tasks.
* **Relevance:**  Useful for advancing AI applications in chemistry research, development, and education.
* **Source:** [https://github.com/ChemFoundationModels/ChemLLMBench](https://github.com/ChemFoundationModels/ChemLLMBench)

### SMolInstruct: Instruction tuning dataset for chemistry [2024]
* **Description:** An instruction-tuning dataset focused on small molecules, including over 3M samples across 14 tasks like name conversion, property prediction, and reaction prediction.
* **Purpose:**  Enhances LLMs' ability to follow chemistry-specific instructions and improves their performance in chemical tasks.
* **Relevance:**  Important for developing instruction-tuned LLMs for assisting in chemical research and development.
* **Source:** [https://openreview.net/forum?id=lY6XTF9tPv](https://openreview.net/forum?id=lY6XTF9tPv)

### ChemBench4k [2024]
* **Description:** Includes 4100 high-quality single-choice questions across nine core chemistry tasks.
* **Purpose:** Evaluates LLMs' chemistry knowledge and reasoning through a large set of curated questions.
* **Relevance:** Crucial for assessing LLMs' competency in chemistry, particularly in education and knowledge evaluation.
* **Source:** [https://huggingface.co/datasets/AI4Chem/ChemBench4K](https://huggingface.co/datasets/AI4Chem/ChemBench4K)

### Fine-tuning Large Language Models for Chemical Text Mining [2024]
* **Description:** A study and resources for fine-tuning LLMs on chemical text mining tasks like compound recognition and reaction labeling.
* **Purpose:** Demonstrates the effectiveness of fine-tuning LLMs for complex chemical information extraction from text.
* **Relevance:**  Valuable for chemical research by improving LLMs' ability to extract knowledge from chemical literature.
* **Source:** [Chem. Sci., 2024](https://pubs.rsc.org/en/content/articlehtml/2024/sc/d4sc00924j)

### ChemLit-QA [2024]
* **Description:** An expert-validated, open-source dataset with over 1,000 entries designed for chemistry Retrieval-Augmented Generation (RAG) and fine-tuning tasks.
* **Purpose:**  Benchmarks LLMs in chemistry-specific RAG tasks, evaluating their ability to generate context-aware, factual answers from chemistry literature.
* **Relevance:**  Aids in developing and evaluating LLMs for chemistry research, particularly in tasks requiring information retrieval and synthesis from scientific text.
* **Resources:** [ChemLit-QA GitHub](https://github.com/geemi725/ChemLit-QA)

### ScholarChemQA [2024 July]
* **Description:** A large-scale Question Answering dataset constructed from chemical research papers, featuring multi-choice questions based on paper titles and abstracts.
* **Purpose:**  Evaluates LLMs' ability to answer research-level chemical questions, reflecting real-world challenges in chemical information processing.
* **Relevance:**  Benchmarks LLMs on understanding and reasoning over chemical research literature, highlighting areas for improvement in complex chemical QA.
* **Source:** [arXiv:2407.16931](https://arxiv.org/abs/2407.16931)


## Materials Science Benchmarks

### Leveraging Large Language Models for Explaining Material Synthesis Mechanisms [2024 - NeurIPS AI4Mat]
* **Description:** A benchmark dataset of 775 semi-manually created multiple-choice questions focused on gold nanoparticle (AuNPs) synthesis mechanisms.
* **Purpose:** Evaluates LLMs' reasoning about material synthesis mechanisms and their understanding of physicochemical principles.
* **Relevance:** Highlights the potential of LLMs in understanding scientific mechanisms and provides tools for exploring synthesis methods.
* **Source:** [https://github.com/amair-lab/Physicochemical-LMs](https://github.com/amair-lab/Physicochemical-LMs)

### LLM4Mat-Bench: Benchmarking Large Language Models for Materials Property Prediction [2024 November]
* **Description:** The largest benchmark for evaluating LLMs in predicting crystalline material properties.
* **Purpose:** Assesses LLMs' capabilities in materials science, specifically in predicting material properties.
* **Relevance:**  Useful for AI-driven materials research and development, focusing on property prediction tasks.
* **Source:** [LLM4Mat-Bench: Benchmarking Large Language Models for Materials](https://arxiv.org/abs/2411.00177)
* **Source (Alternative):** [arXiv](https://arxiv.org/abs/2411.00177)

### MaterialBENCH: Evaluating College-Level Materials Science Knowledge [2024 September]
* **Description:** A college-level benchmark dataset for materials science, designed to assess knowledge equivalent to that of an undergraduate in the field.
* **Purpose:** Evaluates LLMs' understanding of materials science concepts and problem-solving abilities at the college level.
* **Relevance:** Useful for assessing LLMs' readiness for materials science education and research tasks.
* **Source:** [arXiv](https://arxiv.org/abs/2409.03161)

### MatSci-NLP [2023]
* **Description:** A comprehensive benchmark for NLP models in materials science, covering tasks like property prediction and information extraction from literature.
* **Purpose:** Evaluates NLP models, including LLMs, in materials science-specific tasks, encouraging generalization across different tasks.
* **Relevance:**  A cornerstone benchmark for assessing LLM capabilities in the field of materials science and NLP applications.
* **Source:** [MatSci-NLP](https://mila.quebec/en/article/revolutionizing-materials-science-with-nlp-introducing-matsci-nlp-and-honeybee)

### LLM4Mat-Bench: Benchmarking Large Language Models for Materials [2024 November]
* **Description:**  The largest benchmark focused on evaluating LLMs for predicting properties of crystalline materials.
* **Purpose:**  Specifically assesses LLMs' predictive capabilities in materials science for crystalline structures.
* **Relevance:**  Essential for advancing AI in materials research and development, particularly in property prediction.
* **Source:** [arXiv](https://arxiv.org/abs/2411.00177)


## Medical Benchmarks

### Large Language Model Benchmarks in Medical Tasks [2024 October]
* **Description:** A survey of benchmark datasets for medical LLM tasks, covering text, image, and multimodal data. Includes benchmarks for EHRs, doctor-patient dialogues, medical QA, and image captioning.
* **Purpose:**  Evaluates LLMs in various medical tasks, contributing to the advancement of medical AI.
* **Relevance:**  Vital for progressing multimodal medical AI and enhancing healthcare through AI applications.
* **Source:** [arXiv](https://arxiv.org/html/2410.21348v1)

## Physics Benchmarks

### Physics GRE: Testing an LLM’s performance on the Physics GRE [2023 December]
* **Description:**  Evaluates LLMs' performance on the Physics GRE exam, covering undergraduate physics topics.
* **Purpose:**  Assesses the capabilities and limitations of LLMs in physics education and their understanding of undergraduate-level physics.
* **Relevance:** Important for understanding the potential and risks of using LLMs as educational tools for physics students.
* **Source:** [arXiv](https://arxiv.org/html/2312.04613v1)

---

If you find this markdown helpful, welcome to give a star ⭐️ to the original repository!  And contributions to expand this benchmark list are highly welcome!
