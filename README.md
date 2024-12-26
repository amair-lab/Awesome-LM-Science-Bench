![image](https://github.com/user-attachments/assets/c1f60281-6a28-4ba1-99ee-5827fac35a43)

# Awesome-LM-Science-Bench
> An open benchmark list covers LLM's reasoning benchmark for science problems, we focus on LLM evaluation datasets in natural sciences. 

Hi👋, if you find this repo helpful, welcome to give a star ⭐️! 

As there are many benchmarks being released, we will update this repo frequently and welcome contributions from the community.

---

### Massive Multitask Language Understanding (MMLU, partial for natural science)
* **Description:** Measures general knowledge across 57 different subjects, ranging from STEM to social sciences.
* **Purpose:** To assess the LLM's understanding and reasoning in a wide range of subject areas.
* **Relevance:** Ideal for multifaceted AI systems that require extensive world knowledge and problem-solving ability.
* **Source:** [Measuring Massive Multitask Language Understanding](https://arxiv.org/abs/2009.03300)  
* **Resources:**
  * [MMLU GitHub](https://github.com/hendrycks/test)  
  * [MMLU Dataset](https://people.eecs.berkeley.edu/~hendrycks/data.tar)  

### AI2 Reasoning Challenge (ARC)
* **Description:** Tests LLMs on grade-school science questions, requiring both deep general knowledge and reasoning abilities.
* **Purpose:** To evaluate the ability to answer complex science questions that require logical reasoning.
* **Relevance:** Useful for educational AI applications, automated tutoring systems, and general knowledge assessments.
* **Source:** [Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge](https://arxiv.org/abs/1803.05457)  
* **Resources:**
  * [ARC Dataset: HuggingFace](https://huggingface.co/datasets/ai2_arc)  
  * [ARC Dataset: Allen Institute](https://allenai.org/data/arc)  

### General Language Understanding Evaluation (GLUE)
* **Description:** A collection of various language tasks from multiple datasets, designed to measure overall language understanding.
* **Purpose:** To provide a comprehensive assessment of language understanding abilities in different contexts.
* **Relevance:** Crucial for applications requiring advanced language processing, such as chatbots and content analysis.
* **Source:** [GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding](https://arxiv.org/abs/1804.07461)  
* **Resources:**
  * [GLUE Homepage](https://gluebenchmark.com/)  
  * [GLUE Dataset](https://huggingface.co/datasets/glue)  

### SciQ (multiple-choice questions, physics, chemistry, and biology)
* **Description:** Consists of multiple-choice questions mainly in natural sciences like physics, chemistry, and biology.
* **Purpose:** To test the ability to answer science-based questions, often with additional supporting text.
* **Relevance:** Useful for educational tools, especially in science education and knowledge testing platforms.
* **Source:** [Crowdsourcing Multiple Choice Science Questions](https://arxiv.org/abs/1707.06209)  
* **Resources:**
  * [SciQ Dataset: HuggingFace](https://huggingface.co/datasets/sciq)

### LAB-Bench: Language Agent Biology Benchmark
* **Description:** A broad dataset of over 2,400 multiple-choice questions for evaluating AI systems on a range of practical biology research capabilities.
* **Purpose:** To assess recall and reasoning over literature, interpretation of figures, navigation of databases, and comprehension and manipulation of DNA and protein sequences.
* **Relevance:** Useful for evaluating LLMs in the context of biology research and education.
* **Source:** [LAB-Bench: Measuring Capabilities of Language Models for Biology](https://arxiv.org/abs/2407.10362)

### ChemQA: Chemistry Question-Answering Dataset
* **Description:** A multimodal question-and-answering dataset on chemistry reasoning with 5 QA tasks.
* **Purpose:** To evaluate LLMs' abilities in chemistry-related tasks such as counting atoms, calculating molecular weights, and retrosynthesis planning.
* **Relevance:** Essential for AI applications in chemistry education and research.
* **Source:** [GitHub - materials-data-facility/matchem-llm](https://github.com/materials-data-facility/matchem-llm)

### ChemBench - Lamalab
* **Description:** A benchmark with over 7000 questions curated for various chemical topics.
* **Purpose:** To evaluate LLMs on chemistry knowledge and reasoning abilities.
* **Relevance:** Important for AI systems in chemistry education and research.
* **Source:** [GitHub - materials-data-facility/matchem-llm](https://github.com/materials-data-facility/matchem-llm)

### ChemSafetyBench: LLM Safety in Chemistry
* **Description:** A benchmark designed to evaluate the safety of LLMs in the field of chemistry.
* **Purpose:** To assess the safety and reliability of LLMs in chemistry-related applications.
* **Relevance:** Crucial for ensuring the safe use of LLMs in chemistry.
* **Source:** [ChemSafetyBench: Benchmarking LLM Safety on Chemistry](https://arxiv.org/pdf/2411.16736)

### LLM4Mat-Bench: Benchmarking LLMs for Materials Property Prediction
* **Description:** The largest benchmark for evaluating the performance of LLMs in predicting the properties of crystalline materials.
* **Purpose:** To assess LLMs' capabilities in materials science and chemistry.
* **Relevance:** Useful for AI applications in materials research and development.
* **Source:** [LLM4Mat-Bench: Benchmarking Large Language Models for Materials](https://arxiv.org/abs/2411.00177)

### BioLLMBench: A Comprehensive Benchmarking of Large Language Models in Biology
* **Description:** A novel benchmarking framework for comprehensively evaluating LLMs in solving bioinformatics tasks.
* **Purpose:** To assess LLMs' capabilities in bioinformatics and biological reasoning.
* **Relevance:** Useful for evaluating LLMs in the context of biological research and data analysis.
* **Source:** [BioLLMBench: A Comprehensive Benchmarking of Large Language Models in Biology](https://www.biorxiv.org/content/10.1101/2023.12.19.572483v1)

### NPHardEval: Dynamic Benchmark on Reasoning Ability of LLMs
* **Description:** A new benchmark containing a broad spectrum of 900 algorithmic questions belonging up to the NP-Hard complexity class.
* **Purpose:** To evaluate the reasoning ability of LLMs on complex algorithmic questions.
* **Relevance:** Important for assessing LLMs' capabilities in solving complex problems in natural sciences.
* **Source:** [NPHardEval: Dynamic Benchmark on Reasoning Ability of LLMs](https://aclanthology.org/2024.acl-long.225/) 

### ChemLLMBench: A comprehensive benchmark on eight chemistry tasks
* **Description:** ChemLLMBench covers a range of chemistry tasks, providing a thorough evaluation of LLMs in the chemistry domain.
* **Purpose:** To assess LLMs' capabilities in various chemistry-related tasks.
* **Relevance:** Useful for advancing chemistry research and education with AI.
* **Source:** [GitHub - materials-data-facility/matchem-llm](https://github.com/materials-data-facility/matchem-llm)

### SMolInstruct: Instruction tuning dataset for chemistry
* **Description:** SMolInstruct focuses on small molecules and includes 14 tasks and over 3M samples, covering name conversion, property prediction, molecule description, and chemical reaction prediction.
* **Purpose:** To enhance LLMs with chemistry-specific instructions and improve their performance on chemistry tasks.
* **Relevance:** Important for developing LLMs that can assist in chemical research and development.
* **Source:** [GitHub - materials-data-facility/matchem-llm](https://github.com/materials-data-facility/matchem-llm)

### ChemBench4k: Chemistry competency evaluation benchmark
* **Description:** ChemBench4k includes nine chemistry core tasks and 4100 high-quality single-choice questions and answers.
* **Purpose:** To evaluate the chemistry knowledge and reasoning abilities of LLMs.
* **Relevance:** Crucial for applications in chemistry education and knowledge assessment.
* **Source:** [GitHub - materials-data-facility/matchem-llm](https://github.com/materials-data-facility/matchem-llm)

### Chem-RnD and ChemEDU CLAIRify: Chemistry protocols and instructions
* **Description:** Chem-RnD and ChemEDU CLAIRify provide detailed chemistry protocols for synthesizing organic compounds and everyday educational chemistry instructions.
* **Purpose:** To assess LLMs' ability to understand and generate instructions for chemical processes.
* **Relevance:** Useful for training LLMs in chemical synthesis and education.
* **Source:** [GitHub - materials-data-facility/matchem-llm](https://github.com/materials-data-facility/matchem-llm)

### LLM4Mat-Bench: Benchmarking Large Language Models for Materials
* **Description:** LLM4Mat-Bench is the largest benchmark for evaluating LLMs in predicting properties of crystalline materials.
* **Purpose:** To assess LLMs' capabilities in materials science and property prediction.
* **Relevance:** Essential for materials research and development using AI.
* **Source:** [arXiv](https://arxiv.org/abs/2411.00177)

### SciBench: Evaluating College-Level Scientific Problem-Solving Abilities of Large Language Models
* **Description:** SciBench is an expansive benchmark suite designed to systematically examine the reasoning capabilities required for solving complex scientific problems at the collegiate level. It contains a curated dataset featuring scientific problems from the mathematics, chemistry, and physics domains.
* **Purpose:** To evaluate the performance of LLMs on collegiate-level scientific problem-solving and to identify areas for improvement in reasoning abilities.
* **Relevance:** Crucial for advancing the scientific research and discovery capabilities of LLMs.
* **Results:** Current LLMs show unsatisfactory performance with the best overall score of only 43.22%, indicating significant room for improvement.
* **Error Analysis:** Errors made by LLMs are categorized into ten problem-solving abilities, suggesting no single prompting strategy outperforms others significantly.
* **Source:** [arXiv:2307.10635](https://doi.org/10.48550/arXiv.2307.10635)  
* **Cite as:** arXiv:2307.10635 [cs.CL] (or arXiv:2307.10635v3 [cs.CL] for this version)

### LAB-Bench: Measuring Capabilities of Language Models for Biology Research
* **Description:** LAB-Bench is a broad dataset of over 2,400 multiple-choice questions designed to evaluate AI systems on practical biology research capabilities, including literature recall, figure interpretation, database navigation, and DNA/protein sequence manipulation.
* **Purpose:** To measure the performance of LLMs on tasks required for scientific research and to develop automated research systems.
* **Relevance:** Essential for accelerating scientific discovery across disciplines by augmenting LLMs.
* **Human Expert Comparison:** Performance of several LLMs is measured and compared against human expert biology researchers.
* **Availability:** A public subset of LAB-Bench is available for use.
* **Source:** [arXiv:2407.10362](https://doi.org/10.48550/arXiv.2407.10362)  
* **Cite as:** arXiv:2407.10362 [cs.AI] (or arXiv:2407.10362v3 [cs.AI] for this version)


### SciEx: Benchmarking Large Language Models on Scientific Exams with Human Expert Grading and Automatic Grading

* **Description:** SciEx is a benchmark consisting of university computer science exam questions designed to evaluate the ability of LLMs to solve scientific tasks. It is characterized by being multilingual (containing both English and German exams), multi-modal (including questions with images), and featuring various types of freeform questions with different difficulty levels.
* **Purpose:** To assess the performance of LLMs on scientific tasks typically encountered in university examinations, including writing algorithms, querying databases, and providing mathematical proofs.
* **Relevance:** Essential for evaluating the capabilities of LLMs in scientific domains and their potential as assistants in academic and research settings.
* **Performance:** The best-performing LLM achieves an average exam grade of 59.4%, indicating that free-form exams in SciEx remain challenging for current LLMs.
* **Human Expert Grading:** Human expert grading of LLM outputs on SciEx is provided to evaluate performance, showcasing the difficulty in assessing free-form responses.
* **LLM-as-a-Judge:** The study proposes using LLMs as judges to grade answers on SciEx, with experiments showing a 0.948 Pearson correlation with expert grading, demonstrating their potential as graders despite not being perfect at solving the exams.
* **Source:** [SciEx Benchmark](https://www.scienceex.ai/) [arXiv](https://arxiv.org/abs/2406.10421)

