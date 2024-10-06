## FakeShield: Explainable Image Forgery Detection and Localization via Multi-modal Large Language Models

**Authors**: Zhipei Xu, Xuanyu Zhang, Runyi Li, Zecheng Tang, Qing Huang, Jian Zhang

**Abstract**: The rapid development of generative AI is a double-edged sword, which not
only facilitates content creation but also makes image manipulation easier and
more difficult to detect. Although current image forgery detection and
localization (IFDL) methods are generally effective, they tend to face two
challenges: \textbf{1)} black-box nature with unknown detection principle,
\textbf{2)} limited generalization across diverse tampering methods (e.g.,
Photoshop, DeepFake, AIGC-Editing). To address these issues, we propose the
explainable IFDL task and design FakeShield, a multi-modal framework capable of
evaluating image authenticity, generating tampered region masks, and providing
a judgment basis based on pixel-level and image-level tampering clues.
Additionally, we leverage GPT-4o to enhance existing IFDL datasets, creating
the Multi-Modal Tamper Description dataSet (MMTD-Set) for training FakeShield's
tampering analysis capabilities. Meanwhile, we incorporate a Domain Tag-guided
Explainable Forgery Detection Module (DTE-FDM) and a Multi-modal Forgery
Localization Module (MFLM) to address various types of tamper detection
interpretation and achieve forgery localization guided by detailed textual
descriptions. Extensive experiments demonstrate that FakeShield effectively
detects and localizes various tampering techniques, offering an explainable and
superior solution compared to previous IFDL methods.

**URL**: http://arxiv.org/pdf/2410.02761v1

**Published**: 2024-10-03

## SIEVE: General Purpose Data Filtering System Matching GPT-4o Accuracy at 1% the Cost

**Authors**: Jifan Zhang, Robert Nowak

**Abstract**: Creating specialized large language models requires vast amounts of clean,
special purpose data for training and fine-tuning. With only a handful of
existing large-scale, domain-specific datasets, creation of new datasets is
required in most applications. This requires the development of new
application-specific filtering of web-scale data. Filtering with a
high-performance, general-purpose LLM such as GPT-4o can be highly effective,
but this is extremely expensive at web-scale. This paper proposes SIEVE, a
lightweight alternative that matches GPT-4o accuracy at a fraction of the cost.
SIEVE can perform up to 500 filtering operations for the cost of one GPT-4o
filtering call. The key to SIEVE is a seamless integration of GPT-4o and
lightweight T5 models, using active learning to fine-tune T5 in the background
with a small number of calls to GPT-4o. Once trained, it performs as well as
GPT-4o at a tiny fraction of the cost. We experimentally validate SIEVE on the
OpenWebText dataset, using five highly customized filter tasks targeting high
quality and domain-specific content. Our results demonstrate the effectiveness
and efficiency of our method in curating large, high-quality datasets for
language model training at a substantially lower cost (1%) than existing
techniques. To further validate SIEVE, experiments show that SIEVE and GPT-4o
achieve similar accuracy, with human evaluators preferring SIEVE's filtering
results to those of GPT-4o.

**URL**: http://arxiv.org/pdf/2410.02755v1

**Published**: 2024-10-03

## Training Language Models on Synthetic Edit Sequences Improves Code Synthesis

**Authors**: Ulyana Piterbarg, Lerrel Pinto, Rob Fergus

**Abstract**: Software engineers mainly write code by editing existing programs. In
contrast, large language models (LLMs) autoregressively synthesize programs in
a single pass. One explanation for this is the scarcity of open-sourced edit
data. While high-quality instruction data for code synthesis is already scarce,
high-quality edit data is even scarcer. To fill this gap, we develop a
synthetic data generation algorithm called LintSeq. This algorithm refactors
existing code into a sequence of code edits by using a linter to procedurally
sample across the error-free insertions that can be used to sequentially write
programs. It outputs edit sequences as text strings consisting of consecutive
program diffs. To test LintSeq, we use it to refactor a dataset of instruction
+ program pairs into instruction + program-diff-sequence tuples. Then, we
instruction finetune a series of smaller LLMs ranging from 2.6B to 14B
parameters on both the re-factored and original versions of this dataset,
comparing zero-shot performance on code synthesis benchmarks. We show that
during repeated sampling, edit sequence finetuned models produce more diverse
programs than baselines. This results in better inference-time scaling for
benchmark coverage as a function of samples, i.e. the fraction of problems
"pass@k" solved by any attempt given "k" tries. For example, on HumanEval
pass@50, small LLMs finetuned on synthetic edit sequences are competitive with
GPT-4 and outperform models finetuned on the baseline dataset by +20% (+/-3%)
in absolute score. Finally, we also pretrain our own tiny LMs for code
understanding. We show that finetuning tiny models on synthetic code edits
results in state-of-the-art code synthesis for the on-device model class. Our
150M parameter edit sequence LM matches or outperforms code models with twice
as many parameters, both with and without repeated sampling, including Codex
and AlphaCode.

**URL**: http://arxiv.org/pdf/2410.02749v1

**Published**: 2024-10-03

## CriSPO: Multi-Aspect Critique-Suggestion-guided Automatic Prompt Optimization for Text Generation

**Authors**: Han He, Qianchu Liu, Lei Xu, Chaitanya Shivade, Yi Zhang, Sundararajan Srinivasan, Katrin Kirchhoff

**Abstract**: Large language models (LLMs) can generate fluent summaries across domains
using prompting techniques, reducing the need to train models for summarization
applications. However, crafting effective prompts that guide LLMs to generate
summaries with the appropriate level of detail and writing style remains a
challenge. In this paper, we explore the use of salient information extracted
from the source document to enhance summarization prompts. We show that adding
keyphrases in prompts can improve ROUGE F1 and recall, making the generated
summaries more similar to the reference and more complete. The number of
keyphrases can control the precision-recall trade-off. Furthermore, our
analysis reveals that incorporating phrase-level salient information is
superior to word- or sentence-level. However, the impact on hallucination is
not universally positive across LLMs. To conduct this analysis, we introduce
Keyphrase Signal Extractor (CriSPO), a lightweight model that can be finetuned
to extract salient keyphrases. By using CriSPO, we achieve consistent ROUGE
improvements across datasets and open-weight and proprietary LLMs without any
LLM customization. Our findings provide insights into leveraging salient
information in building prompt-based summarization systems.

**URL**: http://arxiv.org/pdf/2410.02748v1

**Published**: 2024-10-03

## Neutral residues: revisiting adapters for model extension

**Authors**: Franck Signe Talla, Herve Jegou, Edouard Grave

**Abstract**: We address the problem of extending a pretrained large language model to a
new domain that was not seen at training time, like adding a language for which
the original model has seen no or little training data. Popular solutions like
fine-tuning or low-rank adaptation are successful at domain adaptation, but
formally they do not add any extra capacity and degrade the performance in the
original domain.
  Our paper analyzes this extension problem under three angles: data,
architecture and training procedure, which are advantageously considered
jointly. In particular, we improve adapters and make it possible to learn an
entire new language while ensuring that the output of the neural network is
almost unchanged in the original domain. For this purpose, we modify the new
residual blocks in a way that leads each new residual block to output
near-zeros in the original domain.
  This solution of neutral residues, which borrows architectural components
from mixture of experts, is effective: with only 20% extra learnable weights
compared to an original model trained on English, we get results that are
significantly better than concurrent approaches (fine-tuning, low-rank or
vanilla adapters) in terms of the trade-off between learning a new language and
not forgetting English.

**URL**: http://arxiv.org/pdf/2410.02744v1

**Published**: 2024-10-03

## MA-RLHF: Reinforcement Learning from Human Feedback with Macro Actions

**Authors**: Yekun Chai, Haoran Sun, Huang Fang, Shuohuan Wang, Yu Sun, Hua Wu

**Abstract**: Reinforcement learning from human feedback (RLHF) has demonstrated
effectiveness in aligning large language models (LLMs) with human preferences.
However, token-level RLHF suffers from the credit assignment problem over long
sequences, where delayed rewards make it challenging for the model to discern
which actions contributed to successful outcomes. This hinders learning
efficiency and slows convergence. In this paper, we propose MA-RLHF, a simple
yet effective RLHF framework that incorporates macro actions -- sequences of
tokens or higher-level language constructs -- into the learning process. By
operating at this higher level of abstraction, our approach reduces the
temporal distance between actions and rewards, facilitating faster and more
accurate credit assignment. This results in more stable policy gradient
estimates and enhances learning efficiency within each episode, all without
increasing computational complexity during training or inference. We validate
our approach through extensive experiments across various model sizes and
tasks, including text summarization, dialogue generation, question answering,
and program synthesis. Our method achieves substantial performance improvements
over standard RLHF, with performance gains of up to 30% in text summarization
and code generation, 18% in dialogue, and 8% in question answering tasks.
Notably, our approach reaches parity with vanilla RLHF 1.7x to 2x faster in
terms of training time and continues to outperform it with further training. We
will make our code and data publicly available at
https://github.com/ernie-research/MA-RLHF .

**URL**: http://arxiv.org/pdf/2410.02743v1

**Published**: 2024-10-03

## Grounding Large Language Models In Embodied Environment With Imperfect World Models

**Authors**: Haolan Liu, Jishen Zhao

**Abstract**: Despite a widespread success in various applications, large language models
(LLMs) often stumble when tackling basic physical reasoning or executing
robotics tasks, due to a lack of direct experience with the physical nuances of
the real world. To address these issues, we propose a Grounding Large language
model with Imperfect world MOdel (GLIMO), which utilizes proxy world models
such as simulators to collect and synthesize trining data. GLIMO incorporates
an LLM agent-based data generator to automatically create high-quality and
diverse instruction datasets. The generator includes an iterative self-refining
module for temporally consistent experience sampling, a diverse set of
question-answering instruction seeds, and a retrieval-augmented generation
module for reflecting on prior experiences. Comprehensive experiments show that
our approach improve the performance of strong open-source LLMs like LLaMA-3
with a performance boost of 2.04 $\times$, 1.54 $\times$, and 1.82 $\times$
across three different benchmarks, respectively. The performance is able to
compete with or surpass their larger counterparts such as GPT-4.

**URL**: http://arxiv.org/pdf/2410.02742v1

**Published**: 2024-10-03

## Salient Information Prompting to Steer Content in Prompt-based Abstractive Summarization

**Authors**: Lei Xu, Mohammed Asad Karim, Saket Dingliwal, Aparna Elangovan

**Abstract**: Large language models (LLMs) can generate fluent summaries across domains
using prompting techniques, reducing the need to train models for summarization
applications. However, crafting effective prompts that guide LLMs to generate
summaries with the appropriate level of detail and writing style remains a
challenge. In this paper, we explore the use of salient information extracted
from the source document to enhance summarization prompts. We show that adding
keyphrases in prompts can improve ROUGE F1 and recall, making the generated
summaries more similar to the reference and more complete. The number of
keyphrases can control the precision-recall trade-off. Furthermore, our
analysis reveals that incorporating phrase-level salient information is
superior to word- or sentence-level. However, the impact on hallucination is
not universally positive across LLMs. To conduct this analysis, we introduce
Keyphrase Signal Extractor (SigExt), a lightweight model that can be finetuned
to extract salient keyphrases. By using SigExt, we achieve consistent ROUGE
improvements across datasets and open-weight and proprietary LLMs without any
LLM customization. Our findings provide insights into leveraging salient
information in building prompt-based summarization systems.

**URL**: http://arxiv.org/pdf/2410.02741v1

**Published**: 2024-10-03

## Revisit Large-Scale Image-Caption Data in Pre-training Multimodal Foundation Models

**Authors**: Zhengfeng Lai, Vasileios Saveris, Chen Chen, Hong-You Chen, Haotian Zhang, Bowen Zhang, Juan Lao Tebar, Wenze Hu, Zhe Gan, Peter Grasch, Meng Cao, Yinfei Yang

**Abstract**: Recent advancements in multimodal models highlight the value of rewritten
captions for improving performance, yet key challenges remain. For example,
while synthetic captions often provide superior quality and image-text
alignment, it is not clear whether they can fully replace AltTexts: the role of
synthetic captions and their interaction with original web-crawled AltTexts in
pre-training is still not well understood. Moreover, different multimodal
foundation models may have unique preferences for specific caption formats, but
efforts to identify the optimal captions for each model remain limited. In this
work, we propose a novel, controllable, and scalable captioning pipeline
designed to generate diverse caption formats tailored to various multimodal
models. By examining Short Synthetic Captions (SSC) towards Dense Synthetic
Captions (DSC+) as case studies, we systematically explore their effects and
interactions with AltTexts across models such as CLIP, multimodal LLMs, and
diffusion models. Our findings reveal that a hybrid approach that keeps both
synthetic captions and AltTexts can outperform the use of synthetic captions
alone, improving both alignment and performance, with each model demonstrating
preferences for particular caption formats. This comprehensive analysis
provides valuable insights into optimizing captioning strategies, thereby
advancing the pre-training of multimodal foundation models.

**URL**: http://arxiv.org/pdf/2410.02740v1

**Published**: 2024-10-03

## Justice or Prejudice? Quantifying Biases in LLM-as-a-Judge

**Authors**: Jiayi Ye, Yanbo Wang, Yue Huang, Dongping Chen, Qihui Zhang, Nuno Moniz, Tian Gao, Werner Geyer, Chao Huang, Pin-Yu Chen, Nitesh V Chawla, Xiangliang Zhang

**Abstract**: LLM-as-a-Judge has been widely utilized as an evaluation method in various
benchmarks and served as supervised rewards in model training. However, despite
their excellence in many domains, potential issues are under-explored,
undermining their reliability and the scope of their utility. Therefore, we
identify 12 key potential biases and propose a new automated bias
quantification framework-CALM-which systematically quantifies and analyzes each
type of bias in LLM-as-a-Judge by using automated and principle-guided
modification. Our experiments cover multiple popular language models, and the
results indicate that while advanced models have achieved commendable overall
performance, significant biases persist in certain specific tasks. Empirical
results suggest that there remains room for improvement in the reliability of
LLM-as-a-Judge. Moreover, we also discuss the explicit and implicit influence
of these biases and give some suggestions for the reliable application of
LLM-as-a-Judge. Our work highlights the need for stakeholders to address these
issues and remind users to exercise caution in LLM-as-a-Judge applications.

**URL**: http://arxiv.org/pdf/2410.02736v1

**Published**: 2024-10-03

## Adaptive Inference-Time Compute: LLMs Can Predict if They Can Do Better, Even Mid-Generation

**Authors**: Rohin Manvi, Anikait Singh, Stefano Ermon

**Abstract**: Inference-time computation is a powerful paradigm to enhance the performance
of large language models (LLMs), with Best-of-N sampling being a widely used
technique. However, this method is computationally expensive, requiring both
(1) an external reward model and (2) the generation of multiple samples. In
this work, we introduce a new generative self-evaluation scheme designed to
adaptively reduce the number of generated samples while maintaining or even
improving performance. We use a generative reward model formulation, allowing
the LLM to predict mid-generation the probability that restarting the
generation will yield a better response. These predictions are obtained without
an external reward model and can be used to decide whether or not to generate
more samples, prune unpromising samples early on, or to pick the best sample.
This capability is very inexpensive as it involves generating a single
predefined token. Trained using a dataset constructed with real unfiltered
LMSYS user prompts, Llama 3.1 8B's win rate against GPT-4 on AlpacaEval
increases from 21% to 34% with 16 samples and math performance on GSM8K
improves from 84% to 91%. By sampling only when the LLM determines that it is
beneficial to do so and adaptively adjusting temperature annealing, we
demonstrate that 74% of the improvement from using 16 samples can be achieved
with only 1.2 samples on average. We further demonstrate that 50-75% of samples
can be pruned early in generation with minimal degradation in performance.
Overall, our methods enable more efficient and scalable compute utilization
during inference for LLMs.

**URL**: http://arxiv.org/pdf/2410.02725v1

**Published**: 2024-10-03

## Large Language Models as Markov Chains

**Authors**: Oussama Zekri, Ambroise Odonnat, Abdelhakim Benechehab, Linus Bleistein, Nicolas Boullé, Ievgen Redko

**Abstract**: Large language models (LLMs) have proven to be remarkably efficient, both
across a wide range of natural language processing tasks and well beyond them.
However, a comprehensive theoretical analysis of the origins of their
impressive performance remains elusive. In this paper, we approach this
challenging task by drawing an equivalence between generic autoregressive
language models with vocabulary of size $T$ and context window of size $K$ and
Markov chains defined on a finite state space of size $\mathcal{O}(T^K)$. We
derive several surprising findings related to the existence of a stationary
distribution of Markov chains that capture the inference power of LLMs, their
speed of convergence to it, and the influence of the temperature on the latter.
We then prove pre-training and in-context generalization bounds and show how
the drawn equivalence allows us to enrich their interpretation. Finally, we
illustrate our theoretical guarantees with experiments on several recent LLMs
to highlight how they capture the behavior observed in practice.

**URL**: http://arxiv.org/pdf/2410.02724v1

**Published**: 2024-10-03

## Domain-Specific Retrieval-Augmented Generation Using Vector Stores, Knowledge Graphs, and Tensor Factorization

**Authors**: Ryan C. Barron, Ves Grantcharov, Selma Wanna, Maksim E. Eren, Manish Bhattarai, Nicholas Solovyev, George Tompkins, Charles Nicholas, Kim Ø. Rasmussen, Cynthia Matuszek, Boian S. Alexandrov

**Abstract**: Large Language Models (LLMs) are pre-trained on large-scale corpora and excel
in numerous general natural language processing (NLP) tasks, such as question
answering (QA). Despite their advanced language capabilities, when it comes to
domain-specific and knowledge-intensive tasks, LLMs suffer from hallucinations,
knowledge cut-offs, and lack of knowledge attributions. Additionally, fine
tuning LLMs' intrinsic knowledge to highly specific domains is an expensive and
time consuming process. The retrieval-augmented generation (RAG) process has
recently emerged as a method capable of optimization of LLM responses, by
referencing them to a predetermined ontology. It was shown that using a
Knowledge Graph (KG) ontology for RAG improves the QA accuracy, by taking into
account relevant sub-graphs that preserve the information in a structured
manner. In this paper, we introduce SMART-SLIC, a highly domain-specific LLM
framework, that integrates RAG with KG and a vector store (VS) that store
factual domain specific information. Importantly, to avoid hallucinations in
the KG, we build these highly domain-specific KGs and VSs without the use of
LLMs, but via NLP, data mining, and nonnegative tensor factorization with
automatic model selection. Pairing our RAG with a domain-specific: (i) KG
(containing structured information), and (ii) VS (containing unstructured
information) enables the development of domain-specific chat-bots that
attribute the source of information, mitigate hallucinations, lessen the need
for fine-tuning, and excel in highly domain-specific question answering tasks.
We pair SMART-SLIC with chain-of-thought prompting agents. The framework is
designed to be generalizable to adapt to any specific or specialized domain. In
this paper, we demonstrate the question answering capabilities of our framework
on a corpus of scientific publications on malware analysis and anomaly
detection.

**URL**: http://arxiv.org/pdf/2410.02721v1

**Published**: 2024-10-03

## UncertaintyRAG: Span-Level Uncertainty Enhanced Long-Context Modeling for Retrieval-Augmented Generation

**Authors**: Zixuan Li, Jing Xiong, Fanghua Ye, Chuanyang Zheng, Xun Wu, Jianqiao Lu, Zhongwei Wan, Xiaodan Liang, Chengming Li, Zhenan Sun, Lingpeng Kong, Ngai Wong

**Abstract**: We present UncertaintyRAG, a novel approach for long-context
Retrieval-Augmented Generation (RAG) that utilizes Signal-to-Noise Ratio
(SNR)-based span uncertainty to estimate similarity between text chunks. This
span uncertainty enhances model calibration, improving robustness and
mitigating semantic inconsistencies introduced by random chunking. Leveraging
this insight, we propose an efficient unsupervised learning technique to train
the retrieval model, alongside an effective data sampling and scaling strategy.
UncertaintyRAG outperforms baselines by 2.03% on LLaMA-2-7B, achieving
state-of-the-art results while using only 4% of the training data compared to
other advanced open-source retrieval models under distribution shift settings.
Our method demonstrates strong calibration through span uncertainty, leading to
improved generalization and robustness in long-context RAG tasks. Additionally,
UncertaintyRAG provides a lightweight retrieval model that can be integrated
into any large language model with varying context window lengths, without the
need for fine-tuning, showcasing the flexibility of our approach.

**URL**: http://arxiv.org/pdf/2410.02719v1

**Published**: 2024-10-03

## LLMs Know More Than They Show: On the Intrinsic Representation of LLM Hallucinations

**Authors**: Hadas Orgad, Michael Toker, Zorik Gekhman, Roi Reichart, Idan Szpektor, Hadas Kotek, Yonatan Belinkov

**Abstract**: Large language models (LLMs) often produce errors, including factual
inaccuracies, biases, and reasoning failures, collectively referred to as
"hallucinations". Recent studies have demonstrated that LLMs' internal states
encode information regarding the truthfulness of their outputs, and that this
information can be utilized to detect errors. In this work, we show that the
internal representations of LLMs encode much more information about
truthfulness than previously recognized. We first discover that the
truthfulness information is concentrated in specific tokens, and leveraging
this property significantly enhances error detection performance. Yet, we show
that such error detectors fail to generalize across datasets, implying that --
contrary to prior claims -- truthfulness encoding is not universal but rather
multifaceted. Next, we show that internal representations can also be used for
predicting the types of errors the model is likely to make, facilitating the
development of tailored mitigation strategies. Lastly, we reveal a discrepancy
between LLMs' internal encoding and external behavior: they may encode the
correct answer, yet consistently generate an incorrect one. Taken together,
these insights deepen our understanding of LLM errors from the model's internal
perspective, which can guide future research on enhancing error analysis and
mitigation.

**URL**: http://arxiv.org/pdf/2410.02707v1

**Published**: 2024-10-03

## Discovering Clues of Spoofed LM Watermarks

**Authors**: Thibaud Gloaguen, Nikola Jovanović, Robin Staab, Martin Vechev

**Abstract**: LLM watermarks stand out as a promising way to attribute ownership of
LLM-generated text. One threat to watermark credibility comes from spoofing
attacks, where an unauthorized third party forges the watermark, enabling it to
falsely attribute arbitrary texts to a particular LLM. While recent works have
demonstrated that state-of-the-art schemes are in fact vulnerable to spoofing,
they lack deeper qualitative analysis of the texts produced by spoofing
methods. In this work, we for the first time reveal that there are observable
differences between genuine and spoofed watermark texts. Namely, we show that
regardless of their underlying approach, all current spoofing methods
consistently leave observable artifacts in spoofed texts, indicative of
watermark forgery. We build upon these findings to propose rigorous statistical
tests that reliably reveal the presence of such artifacts, effectively
discovering that a watermark was spoofed. Our experimental evaluation shows
high test power across all current spoofing methods, providing insights into
their fundamental limitations, and suggesting a way to mitigate this threat.

**URL**: http://arxiv.org/pdf/2410.02693v1

**Published**: 2024-10-03

## HiddenGuard: Fine-Grained Safe Generation with Specialized Representation Router

**Authors**: Lingrui Mei, Shenghua Liu, Yiwei Wang, Baolong Bi, Ruibin Yuan, Xueqi Cheng

**Abstract**: As Large Language Models (LLMs) grow increasingly powerful, ensuring their
safety and alignment with human values remains a critical challenge. Ideally,
LLMs should provide informative responses while avoiding the disclosure of
harmful or sensitive information. However, current alignment approaches, which
rely heavily on refusal strategies, such as training models to completely
reject harmful prompts or applying coarse filters are limited by their binary
nature. These methods either fully deny access to information or grant it
without sufficient nuance, leading to overly cautious responses or failures to
detect subtle harmful content. For example, LLMs may refuse to provide basic,
public information about medication due to misuse concerns. Moreover, these
refusal-based methods struggle to handle mixed-content scenarios and lack the
ability to adapt to context-dependent sensitivities, which can result in
over-censorship of benign content. To overcome these challenges, we introduce
HiddenGuard, a novel framework for fine-grained, safe generation in LLMs.
HiddenGuard incorporates Prism (rePresentation Router for In-Stream
Moderation), which operates alongside the LLM to enable real-time, token-level
detection and redaction of harmful content by leveraging intermediate hidden
states. This fine-grained approach allows for more nuanced, context-aware
moderation, enabling the model to generate informative responses while
selectively redacting or replacing sensitive information, rather than outright
refusal. We also contribute a comprehensive dataset with token-level
fine-grained annotations of potentially harmful information across diverse
contexts. Our experiments demonstrate that HiddenGuard achieves over 90% in F1
score for detecting and redacting harmful content while preserving the overall
utility and informativeness of the model's responses.

**URL**: http://arxiv.org/pdf/2410.02684v1

**Published**: 2024-10-03

## DailyDilemmas: Revealing Value Preferences of LLMs with Quandaries of Daily Life

**Authors**: Yu Ying Chiu, Liwei Jiang, Yejin Choi

**Abstract**: As we increasingly seek guidance from LLMs for decision-making in daily life,
many of these decisions are not clear-cut and depend significantly on the
personal values and ethical standards of the users. We present DailyDilemmas, a
dataset of 1,360 moral dilemmas encountered in everyday life. Each dilemma
includes two possible actions and with each action, the affected parties and
human values invoked. Based on these dilemmas, we consolidated a set of human
values across everyday topics e.g., interpersonal relationships, workplace, and
environmental issues. We evaluated LLMs on these dilemmas to determine what
action they will take and the values represented by these actions. Then, we
analyzed these values through the lens of five popular theories inspired by
sociology, psychology and philosophy. These theories are: World Value Survey,
Moral Foundation Theory, Maslow's Hierarchy of Needs, Aristotle's Virtues, and
Plutchik Wheel of Emotion. We find that LLMs are most aligned with the
self-expression over survival values in terms of World Value Survey, care over
loyalty in Moral Foundation Theory. Interestingly, we find large preferences
differences in models for some core values such as truthfulness e.g.,
Mixtral-8x7B model tends to neglect it by 9.7% while GPT-4-turbo model tends to
select it by 9.4%. We also study the recent guidance released by OpenAI
(ModelSpec), and Anthropic (Constitutional AI) to understand how their released
principles reflect their actual value prioritization when facing nuanced moral
reasoning in daily-life settings. We find that end users cannot effectively
steer such prioritization using system prompts.

**URL**: http://arxiv.org/pdf/2410.02683v1

**Published**: 2024-10-03

## Distilling an End-to-End Voice Assistant Without Instruction Training Data

**Authors**: William Held, Ella Li, Michael Ryan, Weiyan Shi, Yanzhe Zhang, Diyi Yang

**Abstract**: Voice assistants, such as Siri and Google Assistant, typically model audio
and text separately, resulting in lost speech information and increased
complexity. Recent efforts to address this with end-to-end Speech Large
Language Models (LLMs) trained with supervised finetuning (SFT)
  have led to models ``forgetting" capabilities from text-only LLMs. Our work
proposes an alternative paradigm for training Speech LLMs without instruction
data, using the response of a text-only LLM to transcripts as self-supervision.
Importantly, this process can be performed without annotated responses. We show
that our Distilled Voice Assistant (DiVA) generalizes to Spoken Question
Answering, Classification, and Translation. Furthermore, we show that DiVA
better meets user preferences, achieving a 72\% win rate compared with
state-of-the-art models like Qwen 2 Audio, despite using $>$100x less training
compute.

**URL**: http://arxiv.org/pdf/2410.02678v1

**Published**: 2024-10-03

## CulturalBench: a Robust, Diverse and Challenging Benchmark on Measuring the (Lack of) Cultural Knowledge of LLMs

**Authors**: Yu Ying Chiu, Liwei Jiang, Bill Yuchen Lin, Chan Young Park, Shuyue Stella Li, Sahithya Ravi, Mehar Bhatia, Maria Antoniak, Yulia Tsvetkov, Vered Shwartz, Yejin Choi

**Abstract**: To make large language models (LLMs) more helpful across diverse cultures, it
is essential to have effective cultural knowledge benchmarks to measure and
track our progress. Effective benchmarks need to be robust, diverse, and
challenging. We introduce CulturalBench: a set of 1,227 human-written and
human-verified questions for effectively assessing LLMs' cultural knowledge,
covering 45 global regions including the underrepresented ones like Bangladesh,
Zimbabwe, and Peru. Questions - each verified by five independent annotators -
span 17 diverse topics ranging from food preferences to greeting etiquettes. We
evaluate models on two setups: CulturalBench-Easy and CulturalBench-Hard which
share the same questions but asked differently. We find that LLMs are sensitive
to such difference in setups (e.g., GPT-4o with 27.3% difference). Compared to
human performance (92.6% accuracy), CulturalBench-Hard is more challenging for
frontier LLMs with the best performing model (GPT-4o) at only 61.5% and the
worst (Llama3-8b) at 21.4%. Moreover, we find that LLMs often struggle with
tricky questions that have multiple correct answers (e.g., What utensils do the
Chinese usually use?), revealing a tendency to converge to a single answer. Our
results also indicate that OpenAI GPT-4o substantially outperform other
proprietary and open source models in questions related to all but one region
(Oceania). Nonetheless, all models consistently underperform on questions
related to South America and the Middle East.

**URL**: http://arxiv.org/pdf/2410.02677v1

**Published**: 2024-10-03

## AlphaIntegrator: Transformer Action Search for Symbolic Integration Proofs

**Authors**: Mert Ünsal, Timon Gehr, Martin Vechev

**Abstract**: We present the first correct-by-construction learning-based system for
step-by-step mathematical integration. The key idea is to learn a policy,
represented by a GPT transformer model, which guides the search for the right
mathematical integration rule, to be carried out by a symbolic solver.
Concretely, we introduce a symbolic engine with axiomatically correct actions
on mathematical expressions, as well as the first dataset for step-by-step
integration. Our GPT-style transformer model, trained on this synthetic data,
demonstrates strong generalization by surpassing its own data generator in
accuracy and efficiency, using 50% fewer search steps. Our experimental results
with SoTA LLMs also demonstrate that the standard approach of fine-tuning LLMs
on a set of question-answer pairs is insufficient for solving this mathematical
task. This motivates the importance of discovering creative methods for
combining LLMs with symbolic reasoning engines, of which our work is an
instance.

**URL**: http://arxiv.org/pdf/2410.02666v1

**Published**: 2024-10-03

## Hate Personified: Investigating the role of LLMs in content moderation

**Authors**: Sarah Masud, Sahajpreet Singh, Viktor Hangya, Alexander Fraser, Tanmoy Chakraborty

**Abstract**: For subjective tasks such as hate detection, where people perceive hate
differently, the Large Language Model's (LLM) ability to represent diverse
groups is unclear. By including additional context in prompts, we
comprehensively analyze LLM's sensitivity to geographical priming, persona
attributes, and numerical information to assess how well the needs of various
groups are reflected. Our findings on two LLMs, five languages, and six
datasets reveal that mimicking persona-based attributes leads to annotation
variability. Meanwhile, incorporating geographical signals leads to better
regional alignment. We also find that the LLMs are sensitive to numerical
anchors, indicating the ability to leverage community-based flagging efforts
and exposure to adversaries. Our work provides preliminary guidelines and
highlights the nuances of applying LLMs in culturally sensitive cases.

**URL**: http://arxiv.org/pdf/2410.02657v1

**Published**: 2024-10-03

## Measuring and Improving Persuasiveness of Generative Models

**Authors**: Somesh Singh, Yaman K Singla, Harini SI, Balaji Krishnamurthy

**Abstract**: LLMs are increasingly being used in workflows involving generating content to
be consumed by humans (e.g., marketing) and also in directly interacting with
humans (e.g., through chatbots). The development of such systems that are
capable of generating verifiably persuasive messages presents both
opportunities and challenges for society. On the one hand, such systems could
positively impact domains like advertising and social good, such as addressing
drug addiction, and on the other, they could be misused for spreading
misinformation and shaping political opinions. To channel LLMs' impact on
society, we need to develop systems to measure and benchmark their
persuasiveness. With this motivation, we introduce PersuasionBench and
PersuasionArena, the first large-scale benchmark and arena containing a battery
of tasks to measure the persuasion ability of generative models automatically.
We investigate to what extent LLMs know and leverage linguistic patterns that
can help them generate more persuasive language. Our findings indicate that the
persuasiveness of LLMs correlates positively with model size, but smaller
models can also be made to have a higher persuasiveness than much larger
models. Notably, targeted training using synthetic and natural datasets
significantly enhances smaller models' persuasive capabilities, challenging
scale-dependent assumptions. Our findings carry key implications for both model
developers and policymakers. For instance, while the EU AI Act and California's
SB-1047 aim to regulate AI models based on the number of floating point
operations, we demonstrate that simple metrics like this alone fail to capture
the full scope of AI's societal impact. We invite the community to explore and
contribute to PersuasionArena and PersuasionBench, available at
https://bit.ly/measure-persuasion, to advance our understanding of AI-driven
persuasion and its societal implications.

**URL**: http://arxiv.org/pdf/2410.02653v1

**Published**: 2024-10-03

## Undesirable Memorization in Large Language Models: A Survey

**Authors**: Ali Satvaty, Suzan Verberne, Fatih Turkmen

**Abstract**: While recent research increasingly showcases the remarkable capabilities of
Large Language Models (LLMs), it's vital to confront their hidden pitfalls.
Among these challenges, the issue of memorization stands out, posing
significant ethical and legal risks. In this paper, we presents a
Systematization of Knowledge (SoK) on the topic of memorization in LLMs.
Memorization is the effect that a model tends to store and reproduce phrases or
passages from the training data and has been shown to be the fundamental issue
to various privacy and security attacks against LLMs.
  We begin by providing an overview of the literature on the memorization,
exploring it across five key dimensions: intentionality, degree,
retrievability, abstraction, and transparency. Next, we discuss the metrics and
methods used to measure memorization, followed by an analysis of the factors
that contribute to memorization phenomenon. We then examine how memorization
manifests itself in specific model architectures and explore strategies for
mitigating these effects. We conclude our overview by identifying potential
research topics for the near future: to develop methods for balancing
performance and privacy in LLMs, and the analysis of memorization in specific
contexts, including conversational agents, retrieval-augmented generation,
multilingual language models, and diffusion language models.

**URL**: http://arxiv.org/pdf/2410.02650v1

**Published**: 2024-10-03

## Agent Security Bench (ASB): Formalizing and Benchmarking Attacks and Defenses in LLM-based Agents

**Authors**: Hanrong Zhang, Jingyuan Huang, Kai Mei, Yifei Yao, Zhenting Wang, Chenlu Zhan, Hongwei Wang, Yongfeng Zhang

**Abstract**: Although LLM-based agents, powered by Large Language Models (LLMs), can use
external tools and memory mechanisms to solve complex real-world tasks, they
may also introduce critical security vulnerabilities. However, the existing
literature does not comprehensively evaluate attacks and defenses against
LLM-based agents. To address this, we introduce Agent Security Bench (ASB), a
comprehensive framework designed to formalize, benchmark, and evaluate the
attacks and defenses of LLM-based agents, including 10 scenarios (e.g.,
e-commerce, autonomous driving, finance), 10 agents targeting the scenarios,
over 400 tools, 23 different types of attack/defense methods, and 8 evaluation
metrics. Based on ASB, we benchmark 10 prompt injection attacks, a memory
poisoning attack, a novel Plan-of-Thought backdoor attack, a mixed attack, and
10 corresponding defenses across 13 LLM backbones with nearly 90,000 testing
cases in total. Our benchmark results reveal critical vulnerabilities in
different stages of agent operation, including system prompt, user prompt
handling, tool usage, and memory retrieval, with the highest average attack
success rate of 84.30\%, but limited effectiveness shown in current defenses,
unveiling important works to be done in terms of agent security for the
community. Our code can be found at https://github.com/agiresearch/ASB.

**URL**: http://arxiv.org/pdf/2410.02644v1

**Published**: 2024-10-03

## Attention in Large Language Models Yields Efficient Zero-Shot Re-Rankers

**Authors**: Shijie Chen, Bernal Jiménez Gutiérrez, Yu Su

**Abstract**: Information retrieval (IR) systems have played a vital role in modern digital
life and have cemented their continued usefulness in this new era of generative
AI via retrieval-augmented generation. With strong language processing
capabilities and remarkable versatility, large language models (LLMs) have
become popular choices for zero-shot re-ranking in IR systems. So far,
LLM-based re-ranking methods rely on strong generative capabilities, which
restricts their use to either specialized or powerful proprietary models. Given
these restrictions, we ask: is autoregressive generation necessary and optimal
for LLMs to perform re-ranking? We hypothesize that there are abundant signals
relevant to re-ranking within LLMs that might not be used to their full
potential via generation. To more directly leverage such signals, we propose
in-context re-ranking (ICR), a novel method that leverages the change in
attention pattern caused by the search query for accurate and efficient
re-ranking. To mitigate the intrinsic biases in LLMs, we propose a calibration
method using a content-free query. Due to the absence of generation, ICR only
requires two ($O(1)$) forward passes to re-rank $N$ documents, making it
substantially more efficient than generative re-ranking methods that require at
least $O(N)$ forward passes. Our novel design also enables ICR to be applied to
any LLM without specialized training while guaranteeing a well-formed ranking.
Extensive experiments with two popular open-weight LLMs on standard single-hop
and multi-hop information retrieval benchmarks show that ICR outperforms
RankGPT while cutting the latency by more than 60% in practice. Through
detailed analyses, we show that ICR's performance is specially strong on tasks
that require more complex re-ranking signals. Our findings call for further
exploration on novel ways of utilizing open-weight LLMs beyond text generation.

**URL**: http://arxiv.org/pdf/2410.02642v1

**Published**: 2024-10-03

## Large Language Model for Multi-Domain Translation: Benchmarking and Domain CoT Fine-tuning

**Authors**: Tianxiang Hu, Pei Zhang, Baosong Yang, Jun Xie, Derek F. Wong, Rui Wang

**Abstract**: Achieving consistent high-quality machine translation (MT) across diverse
domains remains a significant challenge, primarily due to the limited and
imbalanced parallel training data available in various domains. While large
language models (LLMs) have demonstrated impressive general understanding and
generation abilities, their potential in multi-domain MT is under-explored. We
establish a comprehensive benchmark for multi-domain translation, featuring 25
German$\Leftrightarrow$English and 22 Chinese$\Leftrightarrow$English test sets
respectively covering 15 domains. Our evaluation of prominent LLMs reveals a
discernible performance gap against traditional MT systems, highlighting domain
overfitting and catastrophic forgetting issues after fine-tuning on
domain-limited corpora. To mitigate this, we propose a domain Chain of Thought
(CoT) fine-tuning technique that utilizes the intrinsic multi-domain
intelligence of LLMs to improve translation performance. This method inspires
the LLM to perceive domain information from the source text, which then serves
as a helpful hint to guide the translation process. Despite being trained on a
small dataset of four domains, our CoT fine-tune approach achieves notable
enhancements in translation accuracy and domain robustness than traditional
fine-tuning, as evidenced by an average 1.53 BLEU score increase in over 20
German$\rightarrow$English distinct out-of-domain tests.

**URL**: http://arxiv.org/pdf/2410.02631v1

**Published**: 2024-10-03

## Agents' Room: Narrative Generation through Multi-step Collaboration

**Authors**: Fantine Huot, Reinald Kim Amplayo, Jennimaria Palomaki, Alice Shoshana Jakobovits, Elizabeth Clark, Mirella Lapata

**Abstract**: Writing compelling fiction is a multifaceted process combining elements such
as crafting a plot, developing interesting characters, and using evocative
language. While large language models (LLMs) show promise for story writing,
they currently rely heavily on intricate prompting, which limits their use. We
propose Agents' Room, a generation framework inspired by narrative theory, that
decomposes narrative writing into subtasks tackled by specialized agents. To
illustrate our method, we introduce Tell Me A Story, a high-quality dataset of
complex writing prompts and human-written stories, and a novel evaluation
framework designed specifically for assessing long narratives. We show that
Agents' Room generates stories that are preferred by expert evaluators over
those produced by baseline systems by leveraging collaboration and
specialization to decompose the complex story writing task into tractable
components. We provide extensive analysis with automated and human-based
metrics of the generated output.

**URL**: http://arxiv.org/pdf/2410.02603v1

**Published**: 2024-10-03

## Towards Implicit Bias Detection and Mitigation in Multi-Agent LLM Interactions

**Authors**: Angana Borah, Rada Mihalcea

**Abstract**: As Large Language Models (LLMs) continue to evolve, they are increasingly
being employed in numerous studies to simulate societies and execute diverse
social tasks. However, LLMs are susceptible to societal biases due to their
exposure to human-generated data. Given that LLMs are being used to gain
insights into various societal aspects, it is essential to mitigate these
biases. To that end, our study investigates the presence of implicit gender
biases in multi-agent LLM interactions and proposes two strategies to mitigate
these biases. We begin by creating a dataset of scenarios where implicit gender
biases might arise, and subsequently develop a metric to assess the presence of
biases. Our empirical analysis reveals that LLMs generate outputs characterized
by strong implicit bias associations (>= 50\% of the time). Furthermore, these
biases tend to escalate following multi-agent interactions. To mitigate them,
we propose two strategies: self-reflection with in-context examples (ICE); and
supervised fine-tuning. Our research demonstrates that both methods effectively
mitigate implicit biases, with the ensemble of fine-tuning and self-reflection
proving to be the most successful.

**URL**: http://arxiv.org/pdf/2410.02584v1

**Published**: 2024-10-03

## ColaCare: Enhancing Electronic Health Record Modeling through Large Language Model-Driven Multi-Agent Collaboration

**Authors**: Zixiang Wang, Yinghao Zhu, Huiya Zhao, Xiaochen Zheng, Tianlong Wang, Wen Tang, Yasha Wang, Chengwei Pan, Ewen M. Harrison, Junyi Gao, Liantao Ma

**Abstract**: We introduce ColaCare, a framework that enhances Electronic Health Record
(EHR) modeling through multi-agent collaboration driven by Large Language
Models (LLMs). Our approach seamlessly integrates domain-specific expert models
with LLMs to bridge the gap between structured EHR data and text-based
reasoning. Inspired by clinical consultations, ColaCare employs two types of
agents: DoctorAgent and MetaAgent, which collaboratively analyze patient data.
Expert models process and generate predictions from numerical EHR data, while
LLM agents produce reasoning references and decision-making reports within the
collaborative consultation framework. We additionally incorporate the Merck
Manual of Diagnosis and Therapy (MSD) medical guideline within a
retrieval-augmented generation (RAG) module for authoritative evidence support.
Extensive experiments conducted on four distinct EHR datasets demonstrate
ColaCare's superior performance in mortality prediction tasks, underscoring its
potential to revolutionize clinical decision support systems and advance
personalized precision medicine. The code, complete prompt templates, more case
studies, etc. are publicly available at the anonymous link:
https://colacare.netlify.app.

**URL**: http://arxiv.org/pdf/2410.02551v1

**Published**: 2024-10-03

## MedVisionLlama: Leveraging Pre-Trained Large Language Model Layers to Enhance Medical Image Segmentation

**Authors**: Gurucharan Marthi Krishna Kumar, Aman Chadha, Janine Mendola, Amir Shmuel

**Abstract**: Large Language Models (LLMs), known for their versatility in textual data,
are increasingly being explored for their potential to enhance medical image
segmentation, a crucial task for accurate diagnostic imaging. This study
explores enhancing Vision Transformers (ViTs) for medical image segmentation by
integrating pre-trained LLM transformer blocks. Our approach, which
incorporates a frozen LLM transformer block into the encoder of a ViT-based
model, leads to substantial improvements in segmentation performance across
various medical imaging modalities. We propose a Hybrid Attention Mechanism
that combines global and local feature learning with a Multi-Scale Fusion Block
for aggregating features across different scales. The enhanced model shows
significant performance gains, including an average Dice score increase from
0.74 to 0.79 and improvements in accuracy, precision, and the Jaccard Index.
These results demonstrate the effectiveness of LLM-based transformers in
refining medical image segmentation, highlighting their potential to
significantly boost model accuracy and robustness. The source code and our
implementation are available at: https://bit.ly/3zf2CVs

**URL**: http://arxiv.org/pdf/2410.02458v1

**Published**: 2024-10-03

## Intelligence at the Edge of Chaos

**Authors**: Shiyang Zhang, Aakash Patel, Syed A Rizvi, Nianchen Liu, Sizhuang He, Amin Karbasi, Emanuele Zappala, David van Dijk

**Abstract**: We explore the emergence of intelligent behavior in artificial systems by
investigating how the complexity of rule-based systems influences the
capabilities of models trained to predict these rules. Our study focuses on
elementary cellular automata (ECA), simple yet powerful one-dimensional systems
that generate behaviors ranging from trivial to highly complex. By training
distinct Large Language Models (LLMs) on different ECAs, we evaluated the
relationship between the complexity of the rules' behavior and the intelligence
exhibited by the LLMs, as reflected in their performance on downstream tasks.
Our findings reveal that rules with higher complexity lead to models exhibiting
greater intelligence, as demonstrated by their performance on reasoning and
chess move prediction tasks. Both uniform and periodic systems, and often also
highly chaotic systems, resulted in poorer downstream performance, highlighting
a sweet spot of complexity conducive to intelligence. We conjecture that
intelligence arises from the ability to predict complexity and that creating
intelligence may require only exposure to complexity.

**URL**: http://arxiv.org/pdf/2410.02536v1

**Published**: 2024-10-03

## Choices are More Important than Efforts: LLM Enables Efficient Multi-Agent Exploration

**Authors**: Yun Qu, Boyuan Wang, Yuhang Jiang, Jianzhun Shao, Yixiu Mao, Cheems Wang, Chang Liu, Xiangyang Ji

**Abstract**: With expansive state-action spaces, efficient multi-agent exploration remains
a longstanding challenge in reinforcement learning. Although pursuing novelty,
diversity, or uncertainty attracts increasing attention, redundant efforts
brought by exploration without proper guidance choices poses a practical issue
for the community. This paper introduces a systematic approach, termed LEMAE,
choosing to channel informative task-relevant guidance from a knowledgeable
Large Language Model (LLM) for Efficient Multi-Agent Exploration. Specifically,
we ground linguistic knowledge from LLM into symbolic key states, that are
critical for task fulfillment, in a discriminative manner at low LLM inference
costs. To unleash the power of key states, we design Subspace-based Hindsight
Intrinsic Reward (SHIR) to guide agents toward key states by increasing reward
density. Additionally, we build the Key State Memory Tree (KSMT) to track
transitions between key states in a specific task for organized exploration.
Benefiting from diminishing redundant explorations, LEMAE outperforms existing
SOTA approaches on the challenging benchmarks (e.g., SMAC and MPE) by a large
margin, achieving a 10x acceleration in certain scenarios.

**URL**: http://arxiv.org/pdf/2410.02511v1

**Published**: 2024-10-03

## Can Large Language Models Grasp Legal Theories? Enhance Legal Reasoning with Insights from Multi-Agent Collaboration

**Authors**: Weikang Yuan, Junjie Cao, Zhuoren Jiang, Yangyang Kang, Jun Lin, Kaisong Song, tianqianjin lin, Pengwei Yan, Changlong Sun, Xiaozhong Liu

**Abstract**: Large Language Models (LLMs) could struggle to fully understand legal
theories and perform complex legal reasoning tasks. In this study, we introduce
a challenging task (confusing charge prediction) to better evaluate LLMs'
understanding of legal theories and reasoning capabilities. We also propose a
novel framework: Multi-Agent framework for improving complex Legal Reasoning
capability (MALR). MALR employs non-parametric learning, encouraging LLMs to
automatically decompose complex legal tasks and mimic human learning process to
extract insights from legal rules, helping LLMs better understand legal
theories and enhance their legal reasoning abilities. Extensive experiments on
multiple real-world datasets demonstrate that the proposed framework
effectively addresses complex reasoning issues in practical scenarios, paving
the way for more reliable applications in the legal domain.

**URL**: http://arxiv.org/pdf/2410.02507v1

**Published**: 2024-10-03

## Dog-IQA: Standard-guided Zero-shot MLLM for Mix-grained Image Quality Assessment

**Authors**: Kai Liu, Ziqing Zhang, Wenbo Li, Renjing Pei, Fenglong Song, Xiaohong Liu, Linghe Kong, Yulun Zhang

**Abstract**: Image quality assessment (IQA) serves as the golden standard for all models'
performance in nearly all computer vision fields. However, it still suffers
from poor out-of-distribution generalization ability and expensive training
costs. To address these problems, we propose Dog-IQA, a standard-guided
zero-shot mix-grained IQA method, which is training-free and utilizes the
exceptional prior knowledge of multimodal large language models (MLLMs). To
obtain accurate IQA scores, namely scores consistent with humans, we design an
MLLM-based inference pipeline that imitates human experts. In detail, Dog-IQA
applies two techniques. First, Dog-IQA objectively scores with specific
standards that utilize MLLM's behavior pattern and minimize the influence of
subjective factors. Second, Dog-IQA comprehensively takes local semantic
objects and the whole image as input and aggregates their scores, leveraging
local and global information. Our proposed Dog-IQA achieves state-of-the-art
(SOTA) performance compared with training-free methods, and competitive
performance compared with training-based methods in cross-dataset scenarios.
Our code and models will be available at https://github.com/Kai-Liu001/Dog-IQA.

**URL**: http://arxiv.org/pdf/2410.02505v1

**Published**: 2024-10-03

## Defining Knowledge: Bridging Epistemology and Large Language Models

**Authors**: Constanza Fierro, Ruchira Dhar, Filippos Stamatiou, Nicolas Garneau, Anders Søgaard

**Abstract**: Knowledge claims are abundant in the literature on large language models
(LLMs); but can we say that GPT-4 truly "knows" the Earth is round? To address
this question, we review standard definitions of knowledge in epistemology and
we formalize interpretations applicable to LLMs. In doing so, we identify
inconsistencies and gaps in how current NLP research conceptualizes knowledge
with respect to epistemological frameworks. Additionally, we conduct a survey
of 100 professional philosophers and computer scientists to compare their
preferences in knowledge definitions and their views on whether LLMs can really
be said to know. Finally, we suggest evaluation protocols for testing knowledge
in accordance to the most relevant definitions.

**URL**: http://arxiv.org/pdf/2410.02499v1

**Published**: 2024-10-03

## Dynamic Gradient Alignment for Online Data Mixing

**Authors**: Simin Fan, David Grangier, Pierre Ablin

**Abstract**: The composition of training data mixtures is critical for effectively
training large language models (LLMs), as it directly impacts their performance
on downstream tasks. Our goal is to identify an optimal data mixture to
specialize an LLM for a specific task with access to only a few examples.
Traditional approaches to this problem include ad-hoc reweighting methods,
importance sampling, and gradient alignment techniques. This paper focuses on
gradient alignment and introduces Dynamic Gradient Alignment (DGA), a scalable
online gradient alignment algorithm. DGA dynamically estimates the pre-training
data mixture on which the models' gradients align as well as possible with
those of the model on the specific task. DGA is the first gradient alignment
approach that incurs minimal overhead compared to standard pre-training and
outputs a competitive model, eliminating the need for retraining the model.
Experimentally, we demonstrate significant improvements over importance
sampling in two key scenarios: (i) when the pre-training set is small and
importance sampling overfits due to limited data; and (ii) when there is
insufficient specialized data, trapping importance sampling on narrow pockets
of data. Our findings underscore the effectiveness of gradient alignment
methods in optimizing training data mixtures, particularly in data-constrained
environments, and offer a practical solution for enhancing LLM performance on
specific tasks with limited data availability.

**URL**: http://arxiv.org/pdf/2410.02498v1

**Published**: 2024-10-03

## DTVLT: A Multi-modal Diverse Text Benchmark for Visual Language Tracking Based on LLM

**Authors**: Xuchen Li, Shiyu Hu, Xiaokun Feng, Dailing Zhang, Meiqi Wu, Jing Zhang, Kaiqi Huang

**Abstract**: Visual language tracking (VLT) has emerged as a cutting-edge research area,
harnessing linguistic data to enhance algorithms with multi-modal inputs and
broadening the scope of traditional single object tracking (SOT) to encompass
video understanding applications. Despite this, most VLT benchmarks still
depend on succinct, human-annotated text descriptions for each video. These
descriptions often fall short in capturing the nuances of video content
dynamics and lack stylistic variety in language, constrained by their uniform
level of detail and a fixed annotation frequency. As a result, algorithms tend
to default to a "memorize the answer" strategy, diverging from the core
objective of achieving a deeper understanding of video content. Fortunately,
the emergence of large language models (LLMs) has enabled the generation of
diverse text. This work utilizes LLMs to generate varied semantic annotations
(in terms of text lengths and granularities) for representative SOT benchmarks,
thereby establishing a novel multi-modal benchmark. Specifically, we (1)
propose a new visual language tracking benchmark with diverse texts, named
DTVLT, based on five prominent VLT and SOT benchmarks, including three
sub-tasks: short-term tracking, long-term tracking, and global instance
tracking. (2) We offer four granularity texts in our benchmark, considering the
extent and density of semantic information. We expect this multi-granular
generation strategy to foster a favorable environment for VLT and video
understanding research. (3) We conduct comprehensive experimental analyses on
DTVLT, evaluating the impact of diverse text on tracking performance and hope
the identified performance bottlenecks of existing algorithms can support
further research in VLT and video understanding. The proposed benchmark,
experimental results and toolkit will be released gradually on
http://videocube.aitestunion.com/.

**URL**: http://arxiv.org/pdf/2410.02492v1

**Published**: 2024-10-03

## Meta-Models: An Architecture for Decoding LLM Behaviors Through Interpreted Embeddings and Natural Language

**Authors**: Anthony Costarelli, Mat Allen, Severin Field, Joshua Clymer

**Abstract**: As Large Language Models (LLMs) become increasingly integrated into our daily
lives, the potential harms from deceptive behavior underlie the need for
faithfully interpreting their decision-making. While traditional probing
methods have shown some effectiveness, they remain best for narrowly scoped
tasks while more comprehensive explanations are still necessary. To this end,
we investigate meta-models-an architecture using a "meta-model" that takes
activations from an "input-model" and answers natural language questions about
the input-model's behaviors. We evaluate the meta-model's ability to generalize
by training them on selected task types and assessing their out-of-distribution
performance in deceptive scenarios. Our findings show that meta-models
generalize well to out-of-distribution tasks and point towards opportunities
for future research in this area.

**URL**: http://arxiv.org/pdf/2410.02472v1

**Published**: 2024-10-03

## Response Tuning: Aligning Large Language Models without Instruction

**Authors**: Seokhyun An, Hyounghun Kim

**Abstract**: Instruction tuning-supervised fine-tuning using instruction-response pairs-is
a foundational step in transitioning pre-trained Large Language Models (LLMs)
into helpful and safe chat assistants. Our hypothesis is that establishing an
adequate output space can enable such a transition given the capabilities
inherent in pre-trained LLMs. To verify this, we propose Response Tuning (RT),
which eliminates the instruction-conditioning step in instruction tuning and
solely focuses on response space supervision. Our experiments demonstrate that
RT models, trained only using responses, can effectively respond to a wide
range of instructions and exhibit helpfulness comparable to that of their
instruction-tuned counterparts. Furthermore, we observe that controlling the
training response distribution can significantly improve their user preference
or elicit target behaviors such as refusing assistance for unsafe queries. Our
findings illuminate the role of establishing an adequate output space in
alignment, highlighting the potential of the extensive inherent capabilities of
pre-trained LLMs.

**URL**: http://arxiv.org/pdf/2410.02465v1

**Published**: 2024-10-03

## Strong Preferences Affect the Robustness of Value Alignment

**Authors**: Ziwei Xu, Mohan Kankanhalli

**Abstract**: Value alignment, which aims to ensure that large language models (LLMs) and
other AI agents behave in accordance with human values, is critical for
ensuring safety and trustworthiness of these systems. A key component of value
alignment is the modeling of human preferences as a representation of human
values. In this paper, we investigate the robustness of value alignment by
examining the sensitivity of preference models. Specifically, we ask: how do
changes in the probabilities of some preferences affect the predictions of
these models for other preferences? To answer this question, we theoretically
analyze the robustness of widely used preference models by examining their
sensitivities to minor changes in preferences they model. Our findings reveal
that, in the Bradley-Terry and the Placket-Luce model, the probability of a
preference can change significantly as other preferences change, especially
when these preferences are dominant (i.e., with probabilities near 0 or 1). We
identify specific conditions where this sensitivity becomes significant for
these models and discuss the practical implications for the robustness and
safety of value alignment in AI systems.

**URL**: http://arxiv.org/pdf/2410.02451v1

**Published**: 2024-10-03

## Optimizing Adaptive Attacks against Content Watermarks for Language Models

**Authors**: Abdulrahman Diaa, Toluwani Aremu, Nils Lukas

**Abstract**: Large Language Models (LLMs) can be \emph{misused} to spread online spam and
misinformation. Content watermarking deters misuse by hiding a message in
model-generated outputs, enabling their detection using a secret watermarking
key. Robustness is a core security property, stating that evading detection
requires (significant) degradation of the content's quality. Many LLM
watermarking methods have been proposed, but robustness is tested only against
\emph{non-adaptive} attackers who lack knowledge of the watermarking method and
can find only suboptimal attacks. We formulate the robustness of LLM
watermarking as an objective function and propose preference-based optimization
to tune \emph{adaptive} attacks against the specific watermarking method. Our
evaluation shows that (i) adaptive attacks substantially outperform
non-adaptive baselines. (ii) Even in a non-adaptive setting, adaptive attacks
optimized against a few known watermarks remain highly effective when tested
against other unseen watermarks, and (iii) optimization-based attacks are
practical and require less than seven GPU hours. Our findings underscore the
need to test robustness against adaptive attackers.

**URL**: http://arxiv.org/pdf/2410.02440v1

**Published**: 2024-10-03

## Better Call SAUL: Fluent and Consistent Language Model Editing with Generation Regularization

**Authors**: Mingyang Wang, Lukas Lange, Heike Adel, Jannik Strötgen, Hinrich Schütze

**Abstract**: To ensure large language models contain up-to-date knowledge, they need to be
updated regularly. However, model editing is challenging as it might also
affect knowledge that is unrelated to the new data. State-of-the-art methods
identify parameters associated with specific knowledge and then modify them via
direct weight updates. However, these locate-and-edit methods suffer from heavy
computational overhead and lack theoretical validation. In contrast, directly
fine-tuning the model on requested edits affects the model's behavior on
unrelated knowledge, and significantly damages the model's generation fluency
and consistency. To address these challenges, we propose SAUL, a streamlined
model editing method that uses sentence concatenation with augmented random
facts for generation regularization. Evaluations on three model editing
benchmarks show that SAUL is a practical and reliable solution for model
editing outperforming state-of-the-art methods while maintaining generation
quality and reducing computational overhead.

**URL**: http://arxiv.org/pdf/2410.02433v1

**Published**: 2024-10-03

## IoT-LLM: Enhancing Real-World IoT Task Reasoning with Large Language Models

**Authors**: Tuo An, Yunjiao Zhou, Han Zou, Jianfei Yang

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities across
textual and visual domains but often generate outputs that violate physical
laws, revealing a gap in their understanding of the physical world. Inspired by
human cognition, where perception is fundamental to reasoning, we explore
augmenting LLMs with enhanced perception abilities using Internet of Things
(IoT) sensor data and pertinent knowledge for IoT task reasoning in the
physical world. In this work, we systematically study LLMs capability to
address real-world IoT tasks by augmenting their perception and knowledge base,
and then propose a unified framework, IoT-LLM, to enhance such capability. In
IoT-LLM, we customize three steps for LLMs: preprocessing IoT data into formats
amenable to LLMs, activating their commonsense knowledge through
chain-of-thought prompting and specialized role definitions, and expanding
their understanding via IoT-oriented retrieval-augmented generation based on
in-context learning. To evaluate the performance, We design a new benchmark
with five real-world IoT tasks with different data types and reasoning
difficulties and provide the benchmarking results on six open-source and
close-source LLMs. Experimental results demonstrate the limitations of existing
LLMs with naive textual inputs that cannot perform these tasks effectively. We
show that IoT-LLM significantly enhances the performance of IoT tasks reasoning
of LLM, such as GPT-4, achieving an average improvement of 65% across various
tasks against previous methods. The results also showcase LLMs ability to
comprehend IoT data and the physical law behind data by providing a reasoning
process. Limitations of our work are claimed to inspire future research in this
new era.

**URL**: http://arxiv.org/pdf/2410.02429v1

**Published**: 2024-10-03

## Collective Critics for Creative Story Generation

**Authors**: Minwook Bae, Hyounghun Kim

**Abstract**: Generating a long story of several thousand words with narrative coherence
using Large Language Models (LLMs) has been a challenging task. Previous
research has addressed this challenge by proposing different frameworks that
create a story plan and generate a long story based on that plan. However,
these frameworks have been mainly focusing on maintaining narrative coherence
in stories, often overlooking creativity in story planning and the
expressiveness of the stories generated from those plans, which are desirable
properties to captivate readers' interest. In this paper, we propose Collective
Critics for Creative Story Generation framework (CritiCS), which is composed of
plan refining stage (CrPlan) and story generation stage (CrText), to integrate
a collective revision mechanism that promotes those properties into long-form
story generation process. Specifically, in each stage, a group of LLM critics
and one leader collaborate to incrementally refine drafts of plan and story
throughout multiple rounds. Extensive human evaluation shows that the CritiCS
can significantly enhance story creativity and reader engagement, while also
maintaining narrative coherence. Furthermore, the design of the framework
allows active participation from human writers in any role within the critique
process, enabling interactive human-machine collaboration in story writing.

**URL**: http://arxiv.org/pdf/2410.02428v1

**Published**: 2024-10-03

## LLM-Pilot: Characterize and Optimize Performance of your LLM Inference Services

**Authors**: Małgorzata Łazuka, Andreea Anghel, Thomas Parnell

**Abstract**: As Large Language Models (LLMs) are rapidly growing in popularity, LLM
inference services must be able to serve requests from thousands of users while
satisfying performance requirements. The performance of an LLM inference
service is largely determined by the hardware onto which it is deployed, but
understanding of which hardware will deliver on performance requirements
remains challenging. In this work we present LLM-Pilot - a first-of-its-kind
system for characterizing and predicting performance of LLM inference services.
LLM-Pilot performs benchmarking of LLM inference services, under a realistic
workload, across a variety of GPUs, and optimizes the service configuration for
each considered GPU to maximize performance. Finally, using this
characterization data, LLM-Pilot learns a predictive model, which can be used
to recommend the most cost-effective hardware for a previously unseen LLM.
Compared to existing methods, LLM-Pilot can deliver on performance requirements
33% more frequently, whilst reducing costs by 60% on average.

**URL**: http://arxiv.org/pdf/2410.02425v1

**Published**: 2024-10-03

## Parameter Competition Balancing for Model Merging

**Authors**: Guodong Du, Junlin Lee, Jing Li, Runhua Jiang, Yifei Guo, Shuyang Yu, Hanting Liu, Sim Kuan Goh, Ho-Kin Tang, Daojing He, Min Zhang

**Abstract**: While fine-tuning pretrained models has become common practice, these models
often underperform outside their specific domains. Recently developed model
merging techniques enable the direct integration of multiple models, each
fine-tuned for distinct tasks, into a single model. This strategy promotes
multitasking capabilities without requiring retraining on the original
datasets. However, existing methods fall short in addressing potential
conflicts and complex correlations between tasks, especially in parameter-level
adjustments, posing a challenge in effectively balancing parameter competition
across various tasks. This paper introduces an innovative technique named
PCB-Merging (Parameter Competition Balancing), a lightweight and training-free
technique that adjusts the coefficients of each parameter for effective model
merging. PCB-Merging employs intra-balancing to gauge parameter significance
within individual tasks and inter-balancing to assess parameter similarities
across different tasks. Parameters with low importance scores are dropped, and
the remaining ones are rescaled to form the final merged model. We assessed our
approach in diverse merging scenarios, including cross-task, cross-domain, and
cross-training configurations, as well as out-of-domain generalization. The
experimental results reveal that our approach achieves substantial performance
enhancements across multiple modalities, domains, model sizes, number of tasks,
fine-tuning forms, and large language models, outperforming existing model
merging methods. The code is publicly available at:
\url{https://github.com/duguodong7/pcb-merging}.

**URL**: http://arxiv.org/pdf/2410.02396v1

**Published**: 2024-10-03

## Towards Comprehensive Detection of Chinese Harmful Memes

**Authors**: Junyu Lu, Bo Xu, Xiaokun Zhang, Hongbo Wang, Haohao Zhu, Dongyu Zhang, Liang Yang, Hongfei Lin

**Abstract**: This paper has been accepted in the NeurIPS 2024 D & B Track. Harmful memes
have proliferated on the Chinese Internet, while research on detecting Chinese
harmful memes significantly lags behind due to the absence of reliable datasets
and effective detectors. To this end, we focus on the comprehensive detection
of Chinese harmful memes. We construct ToxiCN MM, the first Chinese harmful
meme dataset, which consists of 12,000 samples with fine-grained annotations
for various meme types. Additionally, we propose a baseline detector,
Multimodal Knowledge Enhancement (MKE), incorporating contextual information of
meme content generated by the LLM to enhance the understanding of Chinese
memes. During the evaluation phase, we conduct extensive quantitative
experiments and qualitative analyses on multiple baselines, including LLMs and
our MKE. The experimental results indicate that detecting Chinese harmful memes
is challenging for existing models while demonstrating the effectiveness of
MKE. The resources for this paper are available at
https://github.com/DUT-lujunyu/ToxiCN_MM.

**URL**: http://arxiv.org/pdf/2410.02378v1

**Published**: 2024-10-03

## AlphaEdit: Null-Space Constrained Knowledge Editing for Language Models

**Authors**: Junfeng Fang, Houcheng Jiang, Kun Wang, Yunshan Ma, Xiang Wang, Xiangnan He, Tat-seng Chua

**Abstract**: Large language models (LLMs) often exhibit hallucinations due to incorrect or
outdated knowledge. Hence, model editing methods have emerged to enable
targeted knowledge updates. To achieve this, a prevailing paradigm is the
locating-then-editing approach, which first locates influential parameters and
then edits them by introducing a perturbation. While effective, current studies
have demonstrated that this perturbation inevitably disrupt the originally
preserved knowledge within LLMs, especially in sequential editing scenarios. To
address this, we introduce AlphaEdit, a novel solution that projects
perturbation onto the null space of the preserved knowledge before applying it
to the parameters. We theoretically prove that this projection ensures the
output of post-edited LLMs remains unchanged when queried about the preserved
knowledge, thereby mitigating the issue of disruption. Extensive experiments on
various LLMs, including LLaMA3, GPT2-XL, and GPT-J, show that AlphaEdit boosts
the performance of most locating-then-editing methods by an average of 36.4%
with a single line of additional code for projection solely. Our code is
available at: https://github.com/jianghoucheng/AlphaEdit.

**URL**: http://arxiv.org/pdf/2410.02355v1

**Published**: 2024-10-03

## Listening to the Wise Few: Select-and-Copy Attention Heads for Multiple-Choice QA

**Authors**: Eduard Tulchinskii, Laida Kushnareva, Kristian Kuznetsov, Anastasia Voznyuk, Andrei Andriiainen, Irina Piontkovskaya, Evgeny Burnaev, Serguei Barannikov

**Abstract**: A standard way to evaluate the abilities of LLM involves presenting a
multiple-choice question and selecting the option with the highest logit as the
model's predicted answer. However, such a format for evaluating LLMs has
limitations, since even if the model knows the correct answer, it may struggle
to select the corresponding letter simply due to difficulties in following this
rigid format. To address this, we introduce new scores that better capture and
reveal model's underlying knowledge: the Query-Key Score (QK-score), derived
from the interaction between query and key representations in attention heads,
and the Attention Score, based on attention weights. These scores are extracted
from specific \textit{select-and-copy} heads, which show consistent performance
across popular Multi-Choice Question Answering (MCQA) datasets. Based on these
scores, our method improves knowledge extraction, yielding up to 16\% gain for
LLaMA2-7B and up to 10\% for larger models on popular MCQA benchmarks. At the
same time, the accuracy on a simple synthetic dataset, where the model
explicitly knows the right answer, increases by almost 60\%, achieving nearly
perfect accuracy, therefore demonstrating the method's efficiency in mitigating
MCQA format limitations. To support our claims, we conduct experiments on
models ranging from 7 billion to 70 billion parameters in both zero- and
few-shot setups.

**URL**: http://arxiv.org/pdf/2410.02343v1

**Published**: 2024-10-03

## How Much Can RAG Help the Reasoning of LLM?

**Authors**: Jingyu Liu, Jiaen Lin, Yong Liu

**Abstract**: Retrieval-Augmented Generation (RAG) has gained significant popularity in
modern Large Language Models (LLMs) due to its effectiveness in introducing new
knowledge and reducing hallucinations. However, the deep understanding of RAG
remains limited, how does RAG help the reasoning process and can RAG help
improve the reasoning capability remains question. While external documents are
typically considered as a method to incorporate domain-specific information,
they also contain intermediate reasoning results related to the query, this
suggests that documents could enhance the reasoning capability of LLMs, which
has not been previously explored. In this paper, we investigate this issue in
depth and find that while RAG can assist with reasoning, the help is limited.
If we conceptualize the reasoning process as a tree with fixed depth, then RAG
struggles to assist LLMs in performing deeper reasoning. Additionally, the
information in the documents requires preprocessing to filter out noise. We
demonstrate that this preprocessing is difficult to achieve simply fine-tuning
of the LLM, it often necessitates numerous additional transformer layers to
solve the problem. To simplify the problem, we propose DPrompt tuning, which
effectively resolves the issue within just limited transformer layers, leading
to improved performance.

**URL**: http://arxiv.org/pdf/2410.02338v1

**Published**: 2024-10-03

## Llama SLayer 8B: Shallow Layers Hold the Key to Knowledge Injection

**Authors**: Tianxiang Chen, Zhentao Tan, Tao Gong, Yue Wu, Qi Chu, Bin Liu, Jieping Ye, Nenghai Yu

**Abstract**: As a manner to augment pre-trained large language models (LLM), knowledge
injection is critical to develop vertical domain large models and has been
widely studied. Although most current approaches, including parameter-efficient
fine-tuning (PEFT) and block expansion methods, uniformly apply knowledge
across all LLM layers, it raises the question: are all layers equally crucial
for knowledge injection? We begin by evaluating the importance of each layer in
finding the optimal layer range for knowledge injection. Intuitively, the more
important layers should play a more critical role in knowledge injection and
deserve a denser injection. We observe performance dips in question-answering
benchmarks after the removal or expansion of the shallow layers, and the
degradation shrinks as the layer gets deeper, indicating that the shallow
layers hold the key to knowledge injection. This insight leads us to propose
the S strategy, a post-pretraining strategy of selectively enhancing shallow
layers while pruning the less effective deep ones. Based on this strategy, we
introduce Llama Slayer-8B and Llama Slayer-8B-Instruct. We experimented on the
corpus of code $\&$ math and demonstrated the effectiveness of our strategy.
Further experiments across different LLM, Mistral-7B, and a legal corpus
confirmed the general applicability of the approach, underscoring its
wide-ranging efficacy. Our code is available at:
\https://github.com/txchen-USTC/Llama-Slayer

**URL**: http://arxiv.org/pdf/2410.02330v1

**Published**: 2024-10-03

## Post-edits Are Preferences Too

**Authors**: Nathaniel Berger, Stefan Riezler, Miriam Exel, Matthias Huck

**Abstract**: Preference Optimization (PO) techniques are currently one of the state of the
art techniques for fine-tuning large language models (LLMs) on pairwise
preference feedback from human annotators. However, in machine translation,
this sort of feedback can be difficult to solicit. Additionally, Kreutzer et
al. (2018) have shown that, for machine translation, pairwise preferences are
less reliable than other forms of human feedback, such as 5-point ratings.
  We examine post-edits to see if they can be a source of reliable human
preferences by construction. In PO, a human annotator is shown sequences $s_1$
and $s_2$ and asked for a preference judgment, %$s_1 > s_2$; while for
post-editing, editors \emph{create} $s_1$ and know that it should be better
than $s_2$. We attempt to use these implicit preferences for PO and show that
it helps the model move towards post-edit-like hypotheses and away from machine
translation-like hypotheses. Furthermore, we show that best results are
obtained by pre-training the model with supervised fine-tuning (SFT) on
post-edits in order to promote post-edit-like hypotheses to the top output
ranks.

**URL**: http://arxiv.org/pdf/2410.02320v1

**Published**: 2024-10-03

## Traffic Light or Light Traffic? Investigating Phrasal Semantics in Large Language Models

**Authors**: Rui Meng, Ye Liu, Lifu Tu, Daqing He, Yingbo Zhou, Semih Yavuz

**Abstract**: Phrases are fundamental linguistic units through which humans convey
semantics. This study critically examines the capacity of API-based large
language models (LLMs) to comprehend phrase semantics, utilizing three
human-annotated datasets. We assess the performance of LLMs in executing phrase
semantic reasoning tasks guided by natural language instructions and explore
the impact of common prompting techniques, including few-shot demonstrations
and Chain-of-Thought reasoning. Our findings reveal that LLMs greatly
outperform traditional embedding methods across the datasets; however, they do
not show a significant advantage over fine-tuned methods. The effectiveness of
advanced prompting strategies shows variability. We conduct detailed error
analyses to interpret the limitations faced by LLMs in comprehending phrase
semantics. Code and data can be found at
https://github.com/memray/llm_phrase_semantics.

**URL**: http://arxiv.org/pdf/2410.02308v1

**Published**: 2024-10-03

## Jailbreak Antidote: Runtime Safety-Utility Balance via Sparse Representation Adjustment in Large Language Models

**Authors**: Guobin Shen, Dongcheng Zhao, Yiting Dong, Xiang He, Yi Zeng

**Abstract**: As large language models (LLMs) become integral to various applications,
ensuring both their safety and utility is paramount. Jailbreak attacks, which
manipulate LLMs into generating harmful content, pose significant challenges to
this balance. Existing defenses, such as prompt engineering and safety
fine-tuning, often introduce computational overhead, increase inference
latency, and lack runtime flexibility. Moreover, overly restrictive safety
measures can degrade model utility by causing refusals of benign queries. In
this paper, we introduce Jailbreak Antidote, a method that enables real-time
adjustment of LLM safety preferences by manipulating a sparse subset of the
model's internal states during inference. By shifting the model's hidden
representations along a safety direction with varying strengths, we achieve
flexible control over the safety-utility balance without additional token
overhead or inference delays. Our analysis reveals that safety-related
information in LLMs is sparsely distributed; adjusting approximately 5% of the
internal state is as effective as modifying the entire state. Extensive
experiments on nine LLMs (ranging from 2 billion to 72 billion parameters),
evaluated against ten jailbreak attack methods and compared with six defense
strategies, validate the effectiveness and efficiency of our approach. By
directly manipulating internal states during reasoning, Jailbreak Antidote
offers a lightweight, scalable solution that enhances LLM safety while
preserving utility, opening new possibilities for real-time safety mechanisms
in widely-deployed AI systems.

**URL**: http://arxiv.org/pdf/2410.02298v1

**Published**: 2024-10-03

## Efficient Second-Order Neural Network Optimization via Adaptive Trust Region Methods

**Authors**: James Vo

**Abstract**: Second-order optimization methods offer notable advantages in training deep
neural networks by utilizing curvature information to achieve faster
convergence. However, traditional second-order techniques are computationally
prohibitive, primarily due to the large matrix inversions and high memory
demands they require. While adaptive trust-region methods have been developed
to mitigate these issues, their performance is often hindered by conservative
estimates of key parameters, such as the Lipschitz constant of the Hessian,
resulting in suboptimal outcomes. In this paper, we introduce
SecondOrderAdaptiveAdam (SOAA), a novel optimization algorithm designed to
overcome these limitations. SOAA approximates the Fisher information matrix
using a diagonal representation, reducing computational complexity from
\(O(n^{2})\) to \(O(n)\), thereby making it suitable for large-scale deep
learning models, including large language models (LLMs). Additionally, the
algorithm integrates an adaptive trust-region mechanism that dynamically
adjusts the trust region size based on observed loss reduction, ensuring both
robust convergence and computational efficiency. We empirically demonstrate
that SOAA achieves faster and more stable convergence compared to first-order
optimizers, such as Adam, under similar computational constraints. However, the
diagonal approximation of the Fisher information matrix may be less effective
in capturing higher-order interactions between gradients, suggesting potential
areas for further refinement and future research.

**URL**: http://arxiv.org/pdf/2410.02293v1

**Published**: 2024-10-03

## Morphological evaluation of subwords vocabulary used by BETO language model

**Authors**: Óscar García-Sierra, Ana Fernández-Pampillón Cesteros, Miguel Ortega-Martín

**Abstract**: Subword tokenization algorithms used by Large Language Models are
significantly more efficient and can independently build the necessary
vocabulary of words and subwords without human intervention. However, those
subwords do not always align with real morphemes, potentially impacting the
models' performance, though it remains uncertain when this might occur. In
previous research, we proposed a method to assess the morphological quality of
vocabularies, focusing on the overlap between these vocabularies and the
morphemes of a given language. Our evaluation method was built on three quality
measures, relevance, cohesion, and morphological accuracy, and a procedure for
their assessment. By applying this method to vocabularies created by three
subword tokenization algorithms, BPE, Wordpiece, and Unigram, we concluded that
these vocabularies generally exhibit very low morphological quality. In this
article, we apply this evaluation to the tokenizer of BETO, a BERT language
model trained on large Spanish corpora. This evaluation, along with our
previous results, helped us conclude that its vocabulary has a low
morphological quality, and we also found that training the tokenizer in a
larger corpus does not improve the morphological quality of the generated
vocabulary. Additionally, this evaluation helps clarify the algorithm used by
the tokenizer, that is, Wordpiece, given the inconsistencies between the
authors' claims and the model's configuration.

**URL**: http://arxiv.org/pdf/2410.02283v1

**Published**: 2024-10-03

## CoLLAP: Contrastive Long-form Language-Audio Pretraining with Musical Temporal Structure Augmentation

**Authors**: Junda Wu, Warren Li, Zachary Novack, Amit Namburi, Carol Chen, Julian McAuley

**Abstract**: Modeling temporal characteristics plays a significant role in the
representation learning of audio waveform. We propose Contrastive Long-form
Language-Audio Pretraining (\textbf{CoLLAP}) to significantly extend the
perception window for both the input audio (up to 5 minutes) and the language
descriptions (exceeding 250 words), while enabling contrastive learning across
modalities and temporal dynamics. Leveraging recent Music-LLMs to generate
long-form music captions for full-length songs, augmented with musical temporal
structures, we collect 51.3K audio-text pairs derived from the large-scale
AudioSet training dataset, where the average audio length reaches 288 seconds.
We propose a novel contrastive learning architecture that fuses language
representations with structured audio representations by segmenting each song
into clips and extracting their embeddings. With an attention mechanism, we
capture multimodal temporal correlations, allowing the model to automatically
weigh and enhance the final fusion score for improved contrastive alignment.
Finally, we develop two variants of the CoLLAP model with different types of
backbone language models. Through comprehensive experiments on multiple
long-form music-text retrieval datasets, we demonstrate consistent performance
improvement in retrieval accuracy compared with baselines. We also show the
pretrained CoLLAP models can be transferred to various music information
retrieval tasks, with heterogeneous long-form multimodal contexts.

**URL**: http://arxiv.org/pdf/2410.02271v1

**Published**: 2024-10-03

## SCA: Highly Efficient Semantic-Consistent Unrestricted Adversarial Attack

**Authors**: Zihao Pan, Weibin Wu, Yuhang Cao, Zibin Zheng

**Abstract**: Unrestricted adversarial attacks typically manipulate the semantic content of
an image (e.g., color or texture) to create adversarial examples that are both
effective and photorealistic. Recent works have utilized the diffusion
inversion process to map images into a latent space, where high-level semantics
are manipulated by introducing perturbations. However, they often results in
substantial semantic distortions in the denoised output and suffers from low
efficiency. In this study, we propose a novel framework called
Semantic-Consistent Unrestricted Adversarial Attacks (SCA), which employs an
inversion method to extract edit-friendly noise maps and utilizes Multimodal
Large Language Model (MLLM) to provide semantic guidance throughout the
process. Under the condition of rich semantic information provided by MLLM, we
perform the DDPM denoising process of each step using a series of edit-friendly
noise maps, and leverage DPM Solver++ to accelerate this process, enabling
efficient sampling with semantic consistency. Compared to existing methods, our
framework enables the efficient generation of adversarial examples that exhibit
minimal discernible semantic changes. Consequently, we for the first time
introduce Semantic-Consistent Adversarial Examples (SCAE). Extensive
experiments and visualizations have demonstrated the high efficiency of SCA,
particularly in being on average 12 times faster than the state-of-the-art
attacks. Our code can be found at
https://github.com/Pan-Zihao/SCA}{https://github.com/Pan-Zihao/SCA.

**URL**: http://arxiv.org/pdf/2410.02240v1

**Published**: 2024-10-03

## SEAL: SEmantic-Augmented Imitation Learning via Language Model

**Authors**: Chengyang Gu, Yuxin Pan, Haotian Bai, Hui Xiong, Yize Chen

**Abstract**: Hierarchical Imitation Learning (HIL) is a promising approach for tackling
long-horizon decision-making tasks. While it is a challenging task due to the
lack of detailed supervisory labels for sub-goal learning, and reliance on
hundreds to thousands of expert demonstrations. In this work, we introduce
SEAL, a novel framework that leverages Large Language Models (LLMs)'s powerful
semantic and world knowledge for both specifying sub-goal space and
pre-labeling states to semantically meaningful sub-goal representations without
prior knowledge of task hierarchies. SEAL employs a dual-encoder structure,
combining supervised LLM-guided sub-goal learning with unsupervised Vector
Quantization (VQ) for more robust sub-goal representations. Additionally, SEAL
incorporates a transition-augmented low-level planner for improved adaptation
to sub-goal transitions. Our experiments demonstrate that SEAL outperforms
state-of-the-art HIL methods and LLM-based planning approaches, particularly in
settings with small expert datasets and complex long-horizon tasks.

**URL**: http://arxiv.org/pdf/2410.02231v1

**Published**: 2024-10-03

## CodePMP: Scalable Preference Model Pretraining for Large Language Model Reasoning

**Authors**: Huimu Yu, Xing Wu, Weidong Yin, Debing Zhang, Songlin Hu

**Abstract**: Large language models (LLMs) have made significant progress in natural
language understanding and generation, driven by scalable pretraining and
advanced finetuning. However, enhancing reasoning abilities in LLMs,
particularly via reinforcement learning from human feedback (RLHF), remains
challenging due to the scarcity of high-quality preference data, which is
labor-intensive to annotate and crucial for reward model (RM) finetuning. To
alleviate this issue, we introduce CodePMP, a scalable preference model
pretraining (PMP) pipeline that utilizes a large corpus of synthesized
code-preference pairs from publicly available high-quality source code. CodePMP
improves RM finetuning efficiency by pretraining preference models on
large-scale synthesized code-preference pairs. We evaluate CodePMP on
mathematical reasoning tasks (GSM8K, MATH) and logical reasoning tasks (ReClor,
LogiQA2.0), consistently showing significant improvements in reasoning
performance of LLMs and highlighting the importance of scalable preference
model pretraining for efficient reward modeling.

**URL**: http://arxiv.org/pdf/2410.02229v1

**Published**: 2024-10-03

## EmbedLLM: Learning Compact Representations of Large Language Models

**Authors**: Richard Zhuang, Tianhao Wu, Zhaojin Wen, Andrew Li, Jiantao Jiao, Kannan Ramchandran

**Abstract**: With hundreds of thousands of language models available on Huggingface today,
efficiently evaluating and utilizing these models across various downstream,
tasks has become increasingly critical. Many existing methods repeatedly learn
task-specific representations of Large Language Models (LLMs), which leads to
inefficiencies in both time and computational resources. To address this, we
propose EmbedLLM, a framework designed to learn compact vector representations,
of LLMs that facilitate downstream applications involving many models, such as
model routing. We introduce an encoder-decoder approach for learning such
embeddings, along with a systematic framework to evaluate their effectiveness.
Empirical results show that EmbedLLM outperforms prior methods in model routing
both in accuracy and latency. Additionally, we demonstrate that our method can
forecast a model's performance on multiple benchmarks, without incurring
additional inference cost. Extensive probing experiments validate that the
learned embeddings capture key model characteristics, e.g. whether the model is
specialized for coding tasks, even without being explicitly trained on them. We
open source our dataset, code and embedder to facilitate further research and
application.

**URL**: http://arxiv.org/pdf/2410.02223v1

**Published**: 2024-10-03

## Buckle Up: Robustifying LLMs at Every Customization Stage via Data Curation

**Authors**: Xiaoqun Liu, Jiacheng Liang, Luoxi Tang, Chenyu You, Muchao Ye, Zhaohan Xi

**Abstract**: Large language models (LLMs) are extensively adapted for downstream
applications through a process known as "customization," with fine-tuning being
a common method for integrating domain-specific expertise. However, recent
studies have revealed a vulnerability that tuning LLMs with malicious samples
can compromise their robustness and amplify harmful content, an attack known as
"jailbreaking." To mitigate such attack, we propose an effective defensive
framework utilizing data curation to revise commonsense texts and enhance their
safety implication from the perspective of LLMs. The curated texts can mitigate
jailbreaking attacks at every stage of the customization process: before
customization to immunize LLMs against future jailbreak attempts, during
customization to neutralize jailbreaking risks, or after customization to
restore the compromised models. Since the curated data strengthens LLMs through
the standard fine-tuning workflow, we do not introduce additional modules
during LLM inference, thereby preserving the original customization process.
Experimental results demonstrate a substantial reduction in jailbreaking
effects, with up to a 100% success in generating responsible responses.
Notably, our method is effective even with commonsense texts, which are often
more readily available than safety-relevant data. With the every-stage
defensive framework and supporting experimental performance, this work
represents a significant advancement in mitigating jailbreaking risks and
ensuring the secure customization of LLMs.

**URL**: http://arxiv.org/pdf/2410.02220v1

**Published**: 2024-10-03

## Multi-modal clothing recommendation model based on large model and VAE enhancement

**Authors**: Bingjie Huang, Qingyu Lu, Shuaishuai Huang, Xue-she Wang, Haowei Yang

**Abstract**: Accurately recommending products has long been a subject requiring in-depth
research. This study proposes a multimodal paradigm for clothing
recommendations. Specifically, it designs a multimodal analysis method that
integrates clothing description texts and images, utilizing a pre-trained large
language model to deeply explore the hidden meanings of users and products.
Additionally, a variational encoder is employed to learn the relationship
between user information and products to address the cold start problem in
recommendation systems. This study also validates the significant performance
advantages of this method over various recommendation system methods through
extensive ablation experiments, providing crucial practical guidance for the
comprehensive optimization of recommendation systems.

**URL**: http://arxiv.org/pdf/2410.02219v1

**Published**: 2024-10-03

## Calibrate to Discriminate: Improve In-Context Learning with Label-Free Comparative Inference

**Authors**: Wei Cheng, Tianlu Wang, Yanmin Ji, Fan Yang, Keren Tan, Yiyu Zheng

**Abstract**: While in-context learning with large language models (LLMs) has shown
impressive performance, we have discovered a unique miscalibration behavior
where both correct and incorrect predictions are assigned the same level of
confidence. We refer to this phenomenon as indiscriminate miscalibration. We
found that traditional calibration metrics, such as Expected Calibrated Errors
(ECEs), are unable to capture this behavior effectively. To address this issue,
we propose new metrics to measure the severity of indiscriminate
miscalibration. Additionally, we develop a novel in-context comparative
inference method to alleviate miscalibrations and improve classification
performance. Through extensive experiments on five datasets, we demonstrate
that our proposed method can achieve more accurate and calibrated predictions
compared to regular zero-shot and few-shot prompting.

**URL**: http://arxiv.org/pdf/2410.02210v1

**Published**: 2024-10-03

## Measuring, Evaluating and Improving Logical Consistency in Large Language Models

**Authors**: Yinhong Liu, Zhijiang Guo, Tianya Liang, Ehsan Shareghi, Ivan Vulić, Nigel Collier

**Abstract**: Recent research in Large Language Models (LLMs) has shown promising progress
related to LLM alignment with human preferences. LLM-empowered decision-making
systems are expected to be predictable, reliable and trustworthy, which implies
being free from paradoxes or contradictions that could undermine their
credibility and validity. However, LLMs still exhibit inconsistent and biased
behaviour when making decisions or judgements. In this work, we focus on
studying logical consistency of LLMs as a prerequisite for more reliable and
trustworthy systems. Logical consistency ensures that decisions are based on a
stable and coherent understanding of the problem, reducing the risk of erratic
or contradictory outputs. We first propose a universal framework to quantify
the logical consistency via three fundamental proxies: transitivity,
commutativity and negation invariance. We then evaluate logical consistency,
using the defined measures, of a wide range of LLMs, demonstrating that it can
serve as a strong proxy for overall robustness. Additionally, we introduce a
data refinement and augmentation technique that enhances the logical
consistency of LLMs without sacrificing alignment to human preferences. It
augments noisy and sparse pairwise-comparison annotations by estimating a
partially or totally ordered preference rankings using rank aggregation
methods. Finally, we show that logical consistency impacts the performance of
LLM-based logic-dependent algorithms, where LLMs serve as logical operators.

**URL**: http://arxiv.org/pdf/2410.02205v1

**Published**: 2024-10-03

## GraphIC: A Graph-Based In-Context Example Retrieval Model for Multi-Step Reasoning

**Authors**: Jiale Fu, Yaqing Wang, Simeng Han, Jiaming Fan, Chen Si, Xu Yang

**Abstract**: In-context learning (ICL) enables large language models (LLMs) to generalize
to new tasks by incorporating a few in-context examples (ICEs) directly in the
input, without updating parameters. However, the effectiveness of ICL heavily
relies on the selection of ICEs, and conventional text-based embedding methods
are often inadequate for tasks that require multi-step reasoning, such as
mathematical and logical problem solving. This is due to the bias introduced by
shallow semantic similarities that fail to capture the deeper reasoning
structures required for these tasks. We present GraphIC, a novel approach that
leverages graph-based representations of reasoning processes, coupled with
Bayesian Networks (BNs) to select ICEs. Graph structures inherently filter out
shallow semantics while preserving the core reasoning structure. Importantly,
BNs capture the dependency of a node's attributes on its parent nodes, closely
mirroring the hierarchical nature of human cognition-where each thought is
shaped by preceding ones. This makes BNs particularly well-suited for
multi-step reasoning tasks, aligning the process more closely with human-like
reasoning. Extensive experiments across three types of reasoning tasks
(mathematical reasoning, code generation, and logical reasoning) demonstrate
that GraphIC outperforms both training-free and training-based models in
selecting ICEs, excelling in terms of both effectiveness and efficiency. We
show that GraphIC enhances ICL's performance and interoperability,
significantly advancing ICE selection for multi-step reasoning tasks.

**URL**: http://arxiv.org/pdf/2410.02203v1

**Published**: 2024-10-03

## G2T-LLM: Graph-to-Tree Text Encoding for Molecule Generation with Fine-Tuned Large Language Models

**Authors**: Zhaoning Yu, Xiangyang Xu, Hongyang Gao

**Abstract**: We introduce G2T-LLM, a novel approach for molecule generation that uses
graph-to-tree text encoding to transform graph-based molecular structures into
a hierarchical text format optimized for large language models (LLMs). This
encoding converts complex molecular graphs into tree-structured formats, such
as JSON and XML, which LLMs are particularly adept at processing due to their
extensive pre-training on these types of data. By leveraging the flexibility of
LLMs, our approach allows for intuitive interaction using natural language
prompts, providing a more accessible interface for molecular design. Through
supervised fine-tuning, G2T-LLM generates valid and coherent chemical
structures, addressing common challenges like invalid outputs seen in
traditional graph-based methods. While LLMs are computationally intensive, they
offer superior generalization and adaptability, enabling the generation of
diverse molecular structures with minimal task-specific customization. The
proposed approach achieved comparable performances with state-of-the-art
methods on various benchmark molecular generation datasets, demonstrating its
potential as a flexible and innovative tool for AI-driven molecular design.

**URL**: http://arxiv.org/pdf/2410.02198v1

**Published**: 2024-10-03

## A Survey on Point-of-Interest Recommendation: Models, Architectures, and Security

**Authors**: Qianru Zhang, Peng Yang, Junliang Yu, Haixin Wang, Xingwei He, Siu-Ming Yiu, Hongzhi Yin

**Abstract**: The widespread adoption of smartphones and Location-Based Social Networks has
led to a massive influx of spatio-temporal data, creating unparalleled
opportunities for enhancing Point-of-Interest (POI) recommendation systems.
These advanced POI systems are crucial for enriching user experiences, enabling
personalized interactions, and optimizing decision-making processes in the
digital landscape. However, existing surveys tend to focus on traditional
approaches and few of them delve into cutting-edge developments, emerging
architectures, as well as security considerations in POI recommendations. To
address this gap, our survey stands out by offering a comprehensive, up-to-date
review of POI recommendation systems, covering advancements in models,
architectures, and security aspects. We systematically examine the transition
from traditional models to advanced techniques such as large language models.
Additionally, we explore the architectural evolution from centralized to
decentralized and federated learning systems, highlighting the improvements in
scalability and privacy. Furthermore, we address the increasing importance of
security, examining potential vulnerabilities and privacy-preserving
approaches. Our taxonomy provides a structured overview of the current state of
POI recommendation, while we also identify promising directions for future
research in this rapidly advancing field.

**URL**: http://arxiv.org/pdf/2410.02191v1

**Published**: 2024-10-03

## POSIX: A Prompt Sensitivity Index For Large Language Models

**Authors**: Anwoy Chatterjee, H S V N S Kowndinya Renduchintala, Sumit Bhatia, Tanmoy Chakraborty

**Abstract**: Despite their remarkable capabilities, Large Language Models (LLMs) are found
to be surprisingly sensitive to minor variations in prompts, often generating
significantly divergent outputs in response to minor variations in the prompts,
such as spelling errors, alteration of wording or the prompt template. However,
while assessing the quality of an LLM, the focus often tends to be solely on
its performance on downstream tasks, while very little to no attention is paid
to prompt sensitivity. To fill this gap, we propose POSIX - a novel PrOmpt
Sensitivity IndeX as a reliable measure of prompt sensitivity, thereby offering
a more comprehensive evaluation of LLM performance. The key idea behind POSIX
is to capture the relative change in loglikelihood of a given response upon
replacing the corresponding prompt with a different intent-preserving prompt.
We provide thorough empirical evidence demonstrating the efficacy of POSIX in
capturing prompt sensitivity and subsequently use it to measure and thereby
compare prompt sensitivity of various open-source LLMs. We find that merely
increasing the parameter count or instruction tuning does not necessarily
reduce prompt sensitivity whereas adding some few-shot exemplars, even just
one, almost always leads to significant decrease in prompt sensitivity. We also
find that alterations to prompt template lead to the highest sensitivity in the
case of MCQtype tasks, whereas paraphrasing results in the highest sensitivity
in open-ended generation tasks. The code for reproducing our results is
open-sourced at https://github.com/kowndinyarenduchintala/POSIX.

**URL**: http://arxiv.org/pdf/2410.02185v1

**Published**: 2024-10-03

## CodeJudge: Evaluating Code Generation with Large Language Models

**Authors**: Weixi Tong, Tianyi Zhang

**Abstract**: Large Language Models (LLMs) have shown promising performance in code
generation. However, how to reliably evaluate code generated by LLMs remains an
unresolved problem. This paper presents CodeJudge, a code evaluation framework
that leverages LLMs to evaluate the semantic correctness of generated code
without the need for test cases. We investigate different ways to guide the LLM
in performing "slow thinking" to arrive at an in-depth and reliable evaluation.
We experimented with four LLMs as evaluators on four code generation datasets
and five programming languages. The results show that CodeJudge significantly
outperformed existing methods in most settings. Furthermore, compared with a
SOTA GPT-3.5-based code evaluation method, CodeJudge achieved better results
even when using a much smaller model, Llama-3-8B-Instruct. Our code and
datasets are available on GitHub https://github.com/VichyTong/CodeJudge.

**URL**: http://arxiv.org/pdf/2410.02184v1

**Published**: 2024-10-03

## Efficiently Deploying LLMs with Controlled Risk

**Authors**: Michael J. Zellinger, Matt Thomson

**Abstract**: Deploying large language models in production requires simultaneous attention
to efficiency and risk control. Prior work has shown the possibility to cut
costs while maintaining similar accuracy, but has neglected to focus on risk
control. By contrast, here we present hierarchical chains with multi-level
abstention (HCMA), which use model-intrinsic uncertainty to delegate queries
along the LLM intelligence hierarchy, enabling training-free model switching
based solely on black-box API calls. Our framework presents novel trade-offs
between efficiency and risk. For example, deploying HCMA on MMLU cuts the error
rate of Llama3 405B by 30% when the model is allowed to abstain on 20% of the
queries. To calibrate HCMA for optimal performance, our approach uses
data-efficient logistic regressions (based on a simple nonlinear feature
transformation), which require only 50 or 100 labeled examples to achieve
excellent calibration error (ECE), cutting ECE by 50% compared to naive Platt
scaling. On free-form generation tasks, we find that chain-of-thought is
ineffectual for selective prediction, whereas zero-shot prompting drives error
to 0% on TruthfulQA at high abstention rates. As LLMs are increasingly deployed
across computing environments with different capabilities (such as mobile,
laptop, and cloud), our framework paves the way towards maintaining deployment
efficiency while putting in place sharp risk controls.

**URL**: http://arxiv.org/pdf/2410.02173v1

**Published**: 2024-10-03

## Training Nonlinear Transformers for Chain-of-Thought Inference: A Theoretical Generalization Analysis

**Authors**: Hongkang Li, Meng Wang, Songtao Lu, Xiaodong Cui, Pin-Yu Chen

**Abstract**: Chain-of-Thought (CoT) is an efficient prompting method that enables the
reasoning ability of large language models by augmenting the query using
multiple examples with multiple intermediate steps. Despite the empirical
success, the theoretical understanding of how to train a Transformer to achieve
the CoT ability remains less explored. This is primarily due to the technical
challenges involved in analyzing the nonconvex optimization on nonlinear
attention models. To the best of our knowledge, this work provides the first
theoretical study of training Transformers with nonlinear attention to obtain
the CoT generalization capability so that the resulting model can inference on
unseen tasks when the input is augmented by examples of the new task. We first
quantify the required training samples and iterations to train a Transformer
model towards CoT ability. We then prove the success of its CoT generalization
on unseen tasks with distribution-shifted testing data. Moreover, we
theoretically characterize the conditions for an accurate reasoning output by
CoT even when the provided reasoning examples contain noises and are not always
accurate. In contrast, in-context learning (ICL), which can be viewed as
one-step CoT without intermediate steps, may fail to provide an accurate output
when CoT does. These theoretical findings are justified through experiments.

**URL**: http://arxiv.org/pdf/2410.02167v1

**Published**: 2024-10-03

## A LLM-Powered Automatic Grading Framework with Human-Level Guidelines Optimization

**Authors**: Yucheng Chu, Hang Li, Kaiqi Yang, Harry Shomer, Hui Liu, Yasemin Copur-Gencturk, Jiliang Tang

**Abstract**: Open-ended short-answer questions (SAGs) have been widely recognized as a
powerful tool for providing deeper insights into learners' responses in the
context of learning analytics (LA). However, SAGs often present challenges in
practice due to the high grading workload and concerns about inconsistent
assessments. With recent advancements in natural language processing (NLP),
automatic short-answer grading (ASAG) offers a promising solution to these
challenges. Despite this, current ASAG algorithms are often limited in
generalizability and tend to be tailored to specific questions. In this paper,
we propose a unified multi-agent ASAG framework, GradeOpt, which leverages
large language models (LLMs) as graders for SAGs. More importantly, GradeOpt
incorporates two additional LLM-based agents - the reflector and the refiner -
into the multi-agent system. This enables GradeOpt to automatically optimize
the original grading guidelines by performing self-reflection on its errors.
Through experiments on a challenging ASAG task, namely the grading of
pedagogical content knowledge (PCK) and content knowledge (CK) questions,
GradeOpt demonstrates superior performance in grading accuracy and behavior
alignment with human graders compared to representative baselines. Finally,
comprehensive ablation studies confirm the effectiveness of the individual
components designed in GradeOpt.

**URL**: http://arxiv.org/pdf/2410.02165v1

**Published**: 2024-10-03

## Controlled Generation of Natural Adversarial Documents for Stealthy Retrieval Poisoning

**Authors**: Collin Zhang, Tingwei Zhang, Vitaly Shmatikov

**Abstract**: Recent work showed that retrieval based on embedding similarity (e.g., for
retrieval-augmented generation) is vulnerable to poisoning: an adversary can
craft malicious documents that are retrieved in response to broad classes of
queries. We demonstrate that previous, HotFlip-based techniques produce
documents that are very easy to detect using perplexity filtering. Even if
generation is constrained to produce low-perplexity text, the resulting
documents are recognized as unnatural by LLMs and can be automatically filtered
from the retrieval corpus.
  We design, implement, and evaluate a new controlled generation technique that
combines an adversarial objective (embedding similarity) with a "naturalness"
objective based on soft scores computed using an open-source, surrogate LLM.
The resulting adversarial documents (1) cannot be automatically detected using
perplexity filtering and/or other LLMs, except at the cost of significant false
positives in the retrieval corpus, yet (2) achieve similar poisoning efficacy
to easily-detectable documents generated using HotFlip, and (3) are
significantly more effective than prior methods for energy-guided generation,
such as COLD.

**URL**: http://arxiv.org/pdf/2410.02163v1

**Published**: 2024-10-03

## Planning in Strawberry Fields: Evaluating and Improving the Planning and Scheduling Capabilities of LRM o1

**Authors**: Karthik Valmeekam, Kaya Stechly, Atharva Gundawar, Subbarao Kambhampati

**Abstract**: The ability to plan a course of action that achieves a desired state of
affairs has long been considered a core competence of intelligent agents and
has been an integral part of AI research since its inception. With the advent
of large language models (LLMs), there has been considerable interest in the
question of whether or not they possess such planning abilities, but -- despite
the slew of new private and open source LLMs since GPT3 -- progress has
remained slow. OpenAI claims that their recent o1 (Strawberry) model has been
specifically constructed and trained to escape the normal limitations of
autoregressive LLMs -- making it a new kind of model: a Large Reasoning Model
(LRM). In this paper, we evaluate the planning capabilities of two LRMs
(o1-preview and o1-mini) on both planning and scheduling benchmarks. We see
that while o1 does seem to offer significant improvements over autoregressive
LLMs, this comes at a steep inference cost, while still failing to provide any
guarantees over what it generates. We also show that combining o1 models with
external verifiers -- in a so-called LRM-Modulo system -- guarantees the
correctness of the combined system's output while further improving
performance.

**URL**: http://arxiv.org/pdf/2410.02162v1

**Published**: 2024-10-03

## The why, what, and how of AI-based coding in scientific research

**Authors**: Tonghe Zhuang, Zhicheng Lin

**Abstract**: Computer programming (coding) is indispensable for researchers across
disciplines, yet it remains challenging to learn and time-consuming to carry
out. Generative AI, particularly large language models (LLMs), has the
potential to transform coding into intuitive conversations, but best practices
and effective workflows are only emerging. We dissect AI-based coding through
three key lenses: the nature and role of LLMs in coding (why), six types of
coding assistance they provide (what), and a five-step workflow in action with
practical implementation strategies (how). Additionally, we address the
limitations and future outlook of AI in coding. By offering actionable
insights, this framework helps to guide researchers in effectively leveraging
AI to enhance coding practices and education, accelerating scientific progress.

**URL**: http://arxiv.org/pdf/2410.02156v1

**Published**: 2024-10-03

## From Pixels to Tokens: Byte-Pair Encoding on Quantized Visual Modalities

**Authors**: Wanpeng Zhang, Zilong Xie, Yicheng Feng, Yijiang Li, Xingrun Xing, Sipeng Zheng, Zongqing Lu

**Abstract**: Multimodal Large Language Models have made significant strides in integrating
visual and textual information, yet they often struggle with effectively
aligning these modalities. We introduce a novel image tokenizer that bridges
this gap by applying the principle of Byte-Pair Encoding (BPE) to visual data.
Unlike conventional approaches that rely on separate visual encoders, our
method directly incorporates structural prior information into image tokens,
mirroring the successful tokenization strategies used in text-only Large
Language Models. This innovative approach enables Transformer models to more
effectively learn and reason across modalities. Through theoretical analysis
and extensive experiments, we demonstrate that our BPE Image Tokenizer
significantly enhances MLLMs' multimodal understanding capabilities, even with
limited training data. Our method not only improves performance across various
benchmarks but also shows promising scalability, potentially paving the way for
more efficient and capable multimodal foundation models.

**URL**: http://arxiv.org/pdf/2410.02155v1

**Published**: 2024-10-03

## Can LLMs Reliably Simulate Human Learner Actions? A Simulation Authoring Framework for Open-Ended Learning Environments

**Authors**: Amogh Mannekote, Adam Davies, Jina Kang, Kristy Elizabeth Boyer

**Abstract**: Simulating learner actions helps stress-test open-ended interactive learning
environments and prototype new adaptations before deployment. While recent
studies show the promise of using large language models (LLMs) for simulating
human behavior, such approaches have not gone beyond rudimentary
proof-of-concept stages due to key limitations. First, LLMs are highly
sensitive to minor prompt variations, raising doubts about their ability to
generalize to new scenarios without extensive prompt engineering. Moreover,
apparently successful outcomes can often be unreliable, either because domain
experts unintentionally guide LLMs to produce expected results, leading to
self-fulfilling prophecies; or because the LLM has encountered highly similar
scenarios in its training data, meaning that models may not be simulating
behavior so much as regurgitating memorized content. To address these
challenges, we propose Hyp-Mix, a simulation authoring framework that allows
experts to develop and evaluate simulations by combining testable hypotheses
about learner behavior. Testing this framework in a physics learning
environment, we found that GPT-4 Turbo maintains calibrated behavior even as
the underlying learner model changes, providing the first evidence that LLMs
can be used to simulate realistic behaviors in open-ended interactive learning
environments, a necessary prerequisite for useful LLM behavioral simulation.

**URL**: http://arxiv.org/pdf/2410.02110v1

**Published**: 2024-10-03

## ReGenesis: LLMs can Grow into Reasoning Generalists via Self-Improvement

**Authors**: Xiangyu Peng, Congying Xia, Xinyi Yang, Caiming Xiong, Chien-Sheng Wu, Chen Xing

**Abstract**: Post-training Large Language Models (LLMs) with explicit reasoning
trajectories can enhance their reasoning abilities. However, acquiring such
high-quality trajectory data typically demands meticulous supervision from
humans or superior models, which can be either expensive or
license-constrained. In this paper, we explore how far an LLM can improve its
reasoning by self-synthesizing reasoning paths as training data without any
additional supervision. Existing self-synthesizing methods, such as STaR,
suffer from poor generalization to out-of-domain (OOD) reasoning tasks. We
hypothesize it is due to that their self-synthesized reasoning paths are too
task-specific, lacking general task-agnostic reasoning guidance. To address
this, we propose Reasoning Generalist via Self-Improvement (ReGenesis), a
method to self-synthesize reasoning paths as post-training data by progressing
from abstract to concrete. More specifically, ReGenesis self-synthesizes
reasoning paths by converting general reasoning guidelines into task-specific
ones, generating reasoning structures, and subsequently transforming these
structures into reasoning paths, without the need for human-designed
task-specific examples used in existing methods. We show that ReGenesis
achieves superior performance on all in-domain and OOD settings tested compared
to existing methods. For six OOD tasks specifically, while previous methods
exhibited an average performance decrease of approximately 4.6% after post
training, ReGenesis delivers around 6.1% performance improvement. We also
conduct in-depth analysis of our framework and show ReGenesis is effective
across various LLMs and design choices.

**URL**: http://arxiv.org/pdf/2410.02108v1

**Published**: 2024-10-03

## Racing Thoughts: Explaining Large Language Model Contextualization Errors

**Authors**: Michael A. Lepori, Michael Mozer, Asma Ghandeharioun

**Abstract**: The profound success of transformer-based language models can largely be
attributed to their ability to integrate relevant contextual information from
an input sequence in order to generate a response or complete a task. However,
we know very little about the algorithms that a model employs to implement this
capability, nor do we understand their failure modes. For example, given the
prompt "John is going fishing, so he walks over to the bank. Can he make an ATM
transaction?", a model may incorrectly respond "Yes" if it has not properly
contextualized "bank" as a geographical feature, rather than a financial
institution. We propose the LLM Race Conditions Hypothesis as an explanation of
contextualization errors of this form. This hypothesis identifies dependencies
between tokens (e.g., "bank" must be properly contextualized before the final
token, "?", integrates information from "bank"), and claims that
contextualization errors are a result of violating these dependencies. Using a
variety of techniques from mechanistic intepretability, we provide
correlational and causal evidence in support of the hypothesis, and suggest
inference-time interventions to address it.

**URL**: http://arxiv.org/pdf/2410.02102v1

**Published**: 2024-10-02

## A Watermark for Black-Box Language Models

**Authors**: Dara Bahri, John Wieting, Dana Alon, Donald Metzler

**Abstract**: Watermarking has recently emerged as an effective strategy for detecting the
outputs of large language models (LLMs). Most existing schemes require
\emph{white-box} access to the model's next-token probability distribution,
which is typically not accessible to downstream users of an LLM API. In this
work, we propose a principled watermarking scheme that requires only the
ability to sample sequences from the LLM (i.e. \emph{black-box} access), boasts
a \emph{distortion-free} property, and can be chained or nested using multiple
secret keys. We provide performance guarantees, demonstrate how it can be
leveraged when white-box access is available, and show when it can outperform
existing white-box schemes via comprehensive experiments.

**URL**: http://arxiv.org/pdf/2410.02099v1

**Published**: 2024-10-02

## RLEF: Grounding Code LLMs in Execution Feedback with Reinforcement Learning

**Authors**: Jonas Gehring, Kunhao Zheng, Jade Copet, Vegard Mella, Taco Cohen, Gabriel Synnaeve

**Abstract**: Large language models (LLMs) deployed as agents solve user-specified tasks
over multiple steps while keeping the required manual engagement to a minimum.
Crucially, such LLMs need to ground their generations in any feedback obtained
to reliably achieve desired outcomes. We propose an end-to-end reinforcement
learning method for teaching models to leverage execution feedback in the realm
of code synthesis, where state-of-the-art LLMs struggle to improve code
iteratively compared to independent sampling. We benchmark on competitive
programming tasks, where we achieve new start-of-the art results with both
small (8B parameters) and large (70B) models while reducing the amount of
samples required by an order of magnitude. Our analysis of inference-time
behavior demonstrates that our method produces LLMs that effectively leverage
automatic feedback over multiple steps.

**URL**: http://arxiv.org/pdf/2410.02089v1

**Published**: 2024-10-02

## EMMA: Efficient Visual Alignment in Multi-Modal LLMs

**Authors**: Sara Ghazanfari, Alexandre Araujo, Prashanth Krishnamurthy, Siddharth Garg, Farshad Khorrami

**Abstract**: Multi-modal Large Language Models (MLLMs) have recently exhibited impressive
general-purpose capabilities by leveraging vision foundation models to encode
the core concepts of images into representations. These are then combined with
instructions and processed by the language model to generate high-quality
responses. Despite significant progress in enhancing the language component,
challenges persist in optimally fusing visual encodings within the language
model for task-specific adaptability. Recent research has focused on improving
this fusion through modality adaptation modules but at the cost of
significantly increased model complexity and training data needs. In this
paper, we propose EMMA (Efficient Multi-Modal Adaptation), a lightweight
cross-modality module designed to efficiently fuse visual and textual
encodings, generating instruction-aware visual representations for the language
model. Our key contributions include: (1) an efficient early fusion mechanism
that integrates vision and language representations with minimal added
parameters (less than 0.2% increase in model size), (2) an in-depth
interpretability analysis that sheds light on the internal mechanisms of the
proposed method; (3) comprehensive experiments that demonstrate notable
improvements on both specialized and general benchmarks for MLLMs. Empirical
results show that EMMA boosts performance across multiple tasks by up to 9.3%
while significantly improving robustness against hallucinations. Our code is
available at https://github.com/SaraGhazanfari/EMMA

**URL**: http://arxiv.org/pdf/2410.02080v1

**Published**: 2024-10-02

## Inspection and Control of Self-Generated-Text Recognition Ability in Llama3-8b-Instruct

**Authors**: Christopher Ackerman, Nina Panickssery

**Abstract**: It has been reported that LLMs can recognize their own writing. As this has
potential implications for AI safety, yet is relatively understudied, we
investigate the phenomenon, seeking to establish whether it robustly occurs at
the behavioral level, how the observed behavior is achieved, and whether it can
be controlled. First, we find that the Llama3-8b-Instruct chat model - but not
the base Llama3-8b model - can reliably distinguish its own outputs from those
of humans, and present evidence that the chat model is likely using its
experience with its own outputs, acquired during post-training, to succeed at
the writing recognition task. Second, we identify a vector in the residual
stream of the model that is differentially activated when the model makes a
correct self-written-text recognition judgment, show that the vector activates
in response to information relevant to self-authorship, present evidence that
the vector is related to the concept of "self" in the model, and demonstrate
that the vector is causally related to the model's ability to perceive and
assert self-authorship. Finally, we show that the vector can be used to control
both the model's behavior and its perception, steering the model to claim or
disclaim authorship by applying the vector to the model's output as it
generates it, and steering the model to believe or disbelieve it wrote
arbitrary texts by applying the vector to them as the model reads them.

**URL**: http://arxiv.org/pdf/2410.02064v1

**Published**: 2024-10-02

## TPP-LLM: Modeling Temporal Point Processes by Efficiently Fine-Tuning Large Language Models

**Authors**: Zefang Liu, Yinzhu Quan

**Abstract**: Temporal point processes (TPPs) are widely used to model the timing and
occurrence of events in domains such as social networks, transportation
systems, and e-commerce. In this paper, we introduce TPP-LLM, a novel framework
that integrates large language models (LLMs) with TPPs to capture both the
semantic and temporal aspects of event sequences. Unlike traditional methods
that rely on categorical event type representations, TPP-LLM directly utilizes
the textual descriptions of event types, enabling the model to capture rich
semantic information embedded in the text. While LLMs excel at understanding
event semantics, they are less adept at capturing temporal patterns. To address
this, TPP-LLM incorporates temporal embeddings and employs parameter-efficient
fine-tuning (PEFT) methods to effectively learn temporal dynamics without
extensive retraining. This approach improves both predictive accuracy and
computational efficiency. Experimental results across diverse real-world
datasets demonstrate that TPP-LLM outperforms state-of-the-art baselines in
sequence modeling and event prediction, highlighting the benefits of combining
LLMs with TPPs.

**URL**: http://arxiv.org/pdf/2410.02062v1

**Published**: 2024-10-02

## Synthio: Augmenting Small-Scale Audio Classification Datasets with Synthetic Data

**Authors**: Sreyan Ghosh, Sonal Kumar, Zhifeng Kong, Rafael Valle, Bryan Catanzaro, Dinesh Manocha

**Abstract**: We present Synthio, a novel approach for augmenting small-scale audio
classification datasets with synthetic data. Our goal is to improve audio
classification accuracy with limited labeled data. Traditional data
augmentation techniques, which apply artificial transformations (e.g., adding
random noise or masking segments), struggle to create data that captures the
true diversity present in real-world audios. To address this shortcoming, we
propose to augment the dataset with synthetic audio generated from
text-to-audio (T2A) diffusion models. However, synthesizing effective
augmentations is challenging because not only should the generated data be
acoustically consistent with the underlying small-scale dataset, but they
should also have sufficient compositional diversity. To overcome the first
challenge, we align the generations of the T2A model with the small-scale
dataset using preference optimization. This ensures that the acoustic
characteristics of the generated data remain consistent with the small-scale
dataset. To address the second challenge, we propose a novel caption generation
technique that leverages the reasoning capabilities of Large Language Models to
(1) generate diverse and meaningful audio captions and (2) iteratively refine
their quality. The generated captions are then used to prompt the aligned T2A
model. We extensively evaluate Synthio on ten datasets and four simulated
limited-data settings. Results indicate our method consistently outperforms all
baselines by 0.1%-39% using a T2A model trained only on weakly-captioned
AudioSet.

**URL**: http://arxiv.org/pdf/2410.02056v1

**Published**: 2024-10-02

## Emo3D: Metric and Benchmarking Dataset for 3D Facial Expression Generation from Emotion Description

**Authors**: Mahshid Dehghani, Amirahmad Shafiee, Ali Shafiei, Neda Fallah, Farahmand Alizadeh, Mohammad Mehdi Gholinejad, Hamid Behroozi, Jafar Habibi, Ehsaneddin Asgari

**Abstract**: Existing 3D facial emotion modeling have been constrained by limited emotion
classes and insufficient datasets. This paper introduces "Emo3D", an extensive
"Text-Image-Expression dataset" spanning a wide spectrum of human emotions,
each paired with images and 3D blendshapes. Leveraging Large Language Models
(LLMs), we generate a diverse array of textual descriptions, facilitating the
capture of a broad spectrum of emotional expressions. Using this unique
dataset, we conduct a comprehensive evaluation of language-based models'
fine-tuning and vision-language models like Contranstive Language Image
Pretraining (CLIP) for 3D facial expression synthesis. We also introduce a new
evaluation metric for this task to more directly measure the conveyed emotion.
Our new evaluation metric, Emo3D, demonstrates its superiority over Mean
Squared Error (MSE) metrics in assessing visual-text alignment and semantic
richness in 3D facial expressions associated with human emotions. "Emo3D" has
great applications in animation design, virtual reality, and emotional
human-computer interaction.

**URL**: http://arxiv.org/pdf/2410.02049v1

**Published**: 2024-10-02

## Are Large Language Models Good Classifiers? A Study on Edit Intent Classification in Scientific Document Revisions

**Authors**: Qian Ruan, Ilia Kuznetsov, Iryna Gurevych

**Abstract**: Classification is a core NLP task architecture with many potential
applications. While large language models (LLMs) have brought substantial
advancements in text generation, their potential for enhancing classification
tasks remains underexplored. To address this gap, we propose a framework for
thoroughly investigating fine-tuning LLMs for classification, including both
generation- and encoding-based approaches. We instantiate this framework in
edit intent classification (EIC), a challenging and underexplored
classification task. Our extensive experiments and systematic comparisons with
various training approaches and a representative selection of LLMs yield new
insights into their application for EIC. We investigate the generalizability of
these findings on five further classification tasks. To demonstrate the
proposed methods and address the data shortage for empirical edit analysis, we
use our best-performing EIC model to create Re3-Sci2.0, a new large-scale
dataset of 1,780 scientific document revisions with over 94k labeled edits. The
quality of the dataset is assessed through human evaluation. The new dataset
enables an in-depth empirical study of human editing behavior in academic
writing. We make our experimental framework, models and data publicly
available.

**URL**: http://arxiv.org/pdf/2410.02028v1

**Published**: 2024-10-02

## Zodiac: A Cardiologist-Level LLM Framework for Multi-Agent Diagnostics

**Authors**: Yuan Zhou, Peng Zhang, Mengya Song, Alice Zheng, Yiwen Lu, Zhiheng Liu, Yong Chen, Zhaohan Xi

**Abstract**: Large language models (LLMs) have demonstrated remarkable progress in
healthcare. However, a significant gap remains regarding LLMs' professionalism
in domain-specific clinical practices, limiting their application in real-world
diagnostics. In this work, we introduce ZODIAC, an LLM-powered framework with
cardiologist-level professionalism designed to engage LLMs in cardiological
diagnostics. ZODIAC assists cardiologists by extracting clinically relevant
characteristics from patient data, detecting significant arrhythmias, and
generating preliminary reports for the review and refinement by cardiologists.
To achieve cardiologist-level professionalism, ZODIAC is built on a multi-agent
collaboration framework, enabling the processing of patient data across
multiple modalities. Each LLM agent is fine-tuned using real-world patient data
adjudicated by cardiologists, reinforcing the model's professionalism. ZODIAC
undergoes rigorous clinical validation with independent cardiologists,
evaluated across eight metrics that measure clinical effectiveness and address
security concerns. Results show that ZODIAC outperforms industry-leading
models, including OpenAI's GPT-4o, Meta's Llama-3.1-405B, and Google's
Gemini-pro, as well as medical-specialist LLMs like Microsoft's BioGPT. ZODIAC
demonstrates the transformative potential of specialized LLMs in healthcare by
delivering domain-specific solutions that meet the stringent demands of medical
practice. Notably, ZODIAC has been successfully integrated into
electrocardiography (ECG) devices, exemplifying the growing trend of embedding
LLMs into Software-as-Medical-Device (SaMD).

**URL**: http://arxiv.org/pdf/2410.02026v1

**Published**: 2024-10-02

## FLAG: Financial Long Document Classification via AMR-based GNN

**Authors**: Bolun, Xia, Mohammed J. Zaki, Aparna Gupta

**Abstract**: The advent of large language models (LLMs) has initiated much research into
their various financial applications. However, in applying LLMs on long
documents, semantic relations are not explicitly incorporated, and a full or
arbitrarily sparse attention operation is employed. In recent years, progress
has been made in Abstract Meaning Representation (AMR), which is a graph-based
representation of text to preserve its semantic relations. Since AMR can
represent semantic relationships at a deeper level, it can be beneficially
utilized by graph neural networks (GNNs) for constructing effective
document-level graph representations built upon LLM embeddings to predict
target metrics in the financial domain. We propose FLAG: Financial Long
document classification via AMR-based GNN, an AMR graph based framework to
generate document-level embeddings for long financial document classification.
We construct document-level graphs from sentence-level AMR graphs, endow them
with specialized LLM word embeddings in the financial domain, apply a deep
learning mechanism that utilizes a GNN, and examine the efficacy of our
AMR-based approach in predicting labeled target data from long financial
documents. Extensive experiments are conducted on a dataset of quarterly
earnings calls transcripts of companies in various sectors of the economy, as
well as on a corpus of more recent earnings calls of companies in the S&P 1500
Composite Index. We find that our AMR-based approach outperforms fine-tuning
LLMs directly on text in predicting stock price movement trends at different
time horizons in both datasets. Our work also outperforms previous work
utilizing document graphs and GNNs for text classification.

**URL**: http://arxiv.org/pdf/2410.02024v1

**Published**: 2024-10-02

## UlcerGPT: A Multimodal Approach Leveraging Large Language and Vision Models for Diabetic Foot Ulcer Image Transcription

**Authors**: Reza Basiri, Ali Abedi, Chau Nguyen, Milos R. Popovic, Shehroz S. Khan

**Abstract**: Diabetic foot ulcers (DFUs) are a leading cause of hospitalizations and lower
limb amputations, placing a substantial burden on patients and healthcare
systems. Early detection and accurate classification of DFUs are critical for
preventing serious complications, yet many patients experience delays in
receiving care due to limited access to specialized services. Telehealth has
emerged as a promising solution, improving access to care and reducing the need
for in-person visits. The integration of artificial intelligence and pattern
recognition into telemedicine has further enhanced DFU management by enabling
automatic detection, classification, and monitoring from images. Despite
advancements in artificial intelligence-driven approaches for DFU image
analysis, the application of large language models for DFU image transcription
has not yet been explored. To address this gap, we introduce UlcerGPT, a novel
multimodal approach leveraging large language and vision models for DFU image
transcription. This framework combines advanced vision and language models,
such as Large Language and Vision Assistant and Chat Generative Pre-trained
Transformer, to transcribe DFU images by jointly detecting, classifying, and
localizing regions of interest. Through detailed experiments on a public
dataset, evaluated by expert clinicians, UlcerGPT demonstrates promising
results in the accuracy and efficiency of DFU transcription, offering potential
support for clinicians in delivering timely care via telemedicine.

**URL**: http://arxiv.org/pdf/2410.01989v1

**Published**: 2024-10-02

## Financial Sentiment Analysis on News and Reports Using Large Language Models and FinBERT

**Authors**: Yanxin Shen, Pulin Kirin Zhang

**Abstract**: Financial sentiment analysis (FSA) is crucial for evaluating market sentiment
and making well-informed financial decisions. The advent of large language
models (LLMs) such as BERT and its financial variant, FinBERT, has notably
enhanced sentiment analysis capabilities. This paper investigates the
application of LLMs and FinBERT for FSA, comparing their performance on news
articles, financial reports and company announcements. The study emphasizes the
advantages of prompt engineering with zero-shot and few-shot strategy to
improve sentiment classification accuracy. Experimental results indicate that
GPT-4o, with few-shot examples of financial texts, can be as competent as a
well fine-tuned FinBERT in this specialized field.

**URL**: http://arxiv.org/pdf/2410.01987v1

**Published**: 2024-10-02

## Lost-in-Distance: Impact of Contextual Proximity on LLM Performance in Graph Tasks

**Authors**: Hamed Firooz, Maziar Sanjabi, Wenlong Jiang, Xiaoling Zhai

**Abstract**: Despite significant advancements, Large Language Models (LLMs) exhibit blind
spots that impair their ability to retrieve and process relevant contextual
data effectively. We demonstrate that LLM performance in graph tasks with
complexities beyond the "needle-in-a-haystack" scenario-where solving the
problem requires cross-referencing and reasoning across multiple subproblems
jointly-is influenced by the proximity of relevant information within the
context, a phenomenon we term "lost-in-distance". We examine two fundamental
graph tasks: identifying common connections between two nodes and assessing
similarity among three nodes, and show that the model's performance in these
tasks significantly depends on the relative positioning of common edges. We
evaluate three publicly available LLMs-Llama-3-8B, Llama-3-70B, and GPT-4-using
various graph encoding techniques that represent graph structures for LLM
input. We propose a formulation for the lost-in-distance phenomenon and
demonstrate that lost-in-distance and lost-in-the middle phenomenas occur
independently. Results indicate that model accuracy can decline by up to 6x as
the distance between node connections increases, independent of graph encoding
and model size.

**URL**: http://arxiv.org/pdf/2410.01985v1

**Published**: 2024-10-02

## LLM+KG@VLDB'24 Workshop Summary

**Authors**: Arijit Khan, Tianxing Wu, Xi Chen

**Abstract**: The unification of large language models (LLMs) and knowledge graphs (KGs)
has emerged as a hot topic. At the LLM+KG'24 workshop, held in conjunction with
VLDB 2024 in Guangzhou, China, one of the key themes explored was important
data management challenges and opportunities due to the effective interaction
between LLMs and KGs. This report outlines the major directions and approaches
presented by various speakers during the LLM+KG'24 workshop.

**URL**: http://arxiv.org/pdf/2410.01978v1

**Published**: 2024-10-02

## How Reliable Is Human Feedback For Aligning Large Language Models?

**Authors**: Min-Hsuan Yeh, Leitian Tao, Jeffrey Wang, Xuefeng Du, Yixuan Li

**Abstract**: Most alignment research today focuses on designing new learning algorithms
using datasets like Anthropic-HH, assuming human feedback data is inherently
reliable. However, little attention has been given to the qualitative
unreliability of human feedback and its impact on alignment. To address this
gap, we conduct a comprehensive study and provide an in-depth analysis of human
feedback data. We assess feedback reliability using a committee of gold reward
models, revealing that over 25% of the dataset shows low or no agreement with
these models, implying a high degree of unreliability. Through a qualitative
analysis, we identify six key sources of unreliability, such as mis-labeling,
subjective preferences, differing criteria and thresholds for helpfulness and
harmlessness, etc. Lastly, to mitigate unreliability, we propose Source-Aware
Cleaning, an automatic data-cleaning method guided by the insight of our
qualitative analysis, to significantly improve data quality. Extensive
experiments demonstrate that models trained on our cleaned dataset, HH-Clean,
substantially outperform those trained on the original dataset. We release
HH-Clean to support more reliable LLM alignment evaluation in the future.

**URL**: http://arxiv.org/pdf/2410.01957v1

**Published**: 2024-10-02

## Generate then Refine: Data Augmentation for Zero-shot Intent Detection

**Authors**: I-Fan Lin, Faegheh Hasibi, Suzan Verberne

**Abstract**: In this short paper we propose a data augmentation method for intent
detection in zero-resource domains. Existing data augmentation methods rely on
few labelled examples for each intent category, which can be expensive in
settings with many possible intents. We use a two-stage approach: First, we
generate utterances for intent labels using an open-source large language model
in a zero-shot setting. Second, we develop a smaller sequence-to-sequence model
(the Refiner), to improve the generated utterances. The Refiner is fine-tuned
on seen domains and then applied to unseen domains. We evaluate our method by
training an intent classifier on the generated data, and evaluating it on real
(human) data. We find that the Refiner significantly improves the data utility
and diversity over the zero-shot LLM baseline for unseen domains and over
common baseline approaches. Our results indicate that a two-step approach of a
generative LLM in zero-shot setting and a smaller sequence-to-sequence model
can provide high-quality data for intent detection.

**URL**: http://arxiv.org/pdf/2410.01953v1

**Published**: 2024-10-02

## TypedThinker: Typed Thinking Improves Large Language Model Reasoning

**Authors**: Danqing Wang, Jianxin Ma, Fei Fang, Lei Li

**Abstract**: Despite significant advancements in the reasoning capabilities of Large
Language Models (LLMs), the lack of diverse reasoning solutions often makes
them trapped in a limited solution search area. In this paper, we propose
TypedThinker, a novel framework that enhances LLMs' problem-solving abilities
by incorporating multiple reasoning types (deductive, inductive, abductive, and
analogical). Our analysis across four benchmarks reveals that different
reasoning types uniquely solve distinct sets of problems, highlighting the
importance of diverse thinking approaches. TypedThinker addresses two key
challenges: selecting appropriate reasoning types for given problems and
effectively implementing specific reasoning types. Through self-training on
successful experiences, TypedThinker learns an implicit policy for reasoning
type selection and application. Experimental results demonstrate significant
improvements over baseline models, with accuracy increases of 3.4% for Mistral
7B and 16.7% for LLaMA3 8B across four reasoning benchmarks. Notably,
TypedThinker shows effective generalization to new benchmarks and can further
enhance the reasoning capability of powerful models like GPT-4o. The code is
released at https://github.com/dqwang122/ThinkHub.

**URL**: http://arxiv.org/pdf/2410.01952v1

**Published**: 2024-10-02

## CHASE-SQL: Multi-Path Reasoning and Preference Optimized Candidate Selection in Text-to-SQL

**Authors**: Mohammadreza Pourreza, Hailong Li, Ruoxi Sun, Yeounoh Chung, Shayan Talaei, Gaurav Tarlok Kakkar, Yu Gan, Amin Saberi, Fatma Ozcan, Sercan O. Arik

**Abstract**: In tackling the challenges of large language model (LLM) performance for
Text-to-SQL tasks, we introduce CHASE-SQL, a new framework that employs
innovative strategies, using test-time compute in multi-agent modeling to
improve candidate generation and selection. CHASE-SQL leverages LLMs' intrinsic
knowledge to generate diverse and high-quality SQL candidates using different
LLM generators with: (1) a divide-and-conquer method that decomposes complex
queries into manageable sub-queries in a single LLM call; (2) chain-of-thought
reasoning based on query execution plans, reflecting the steps a database
engine takes during execution; and (3) a unique instance-aware synthetic
example generation technique, which offers specific few-shot demonstrations
tailored to test questions.To identify the best candidate, a selection agent is
employed to rank the candidates through pairwise comparisons with a fine-tuned
binary-candidates selection LLM. This selection approach has been demonstrated
to be more robust over alternatives. The proposed generators-selector framework
not only enhances the quality and diversity of SQL queries but also outperforms
previous methods. Overall, our proposed CHASE-SQL achieves the state-of-the-art
execution accuracy of 73.0% and 73.01% on the test set and development set of
the notable BIRD Text-to-SQL dataset benchmark, rendering CHASE-SQL the top
submission of the leaderboard (at the time of paper submission).

**URL**: http://arxiv.org/pdf/2410.01943v1

**Published**: 2024-10-02

## LLM-Augmented Symbolic Reinforcement Learning with Landmark-Based Task Decomposition

**Authors**: Alireza Kheirandish, Duo Xu, Faramarz Fekri

**Abstract**: One of the fundamental challenges in reinforcement learning (RL) is to take a
complex task and be able to decompose it to subtasks that are simpler for the
RL agent to learn. In this paper, we report on our work that would identify
subtasks by using some given positive and negative trajectories for solving the
complex task. We assume that the states are represented by first-order
predicate logic using which we devise a novel algorithm to identify the
subtasks. Then we employ a Large Language Model (LLM) to generate first-order
logic rule templates for achieving each subtask. Such rules were then further
fined tuned to a rule-based policy via an Inductive Logic Programming
(ILP)-based RL agent. Through experiments, we verify the accuracy of our
algorithm in detecting subtasks which successfully detect all of the subtasks
correctly. We also investigated the quality of the common-sense rules produced
by the language model to achieve the subtasks. Our experiments show that our
LLM-guided rule template generation can produce rules that are necessary for
solving a subtask, which leads to solving complex tasks with fewer assumptions
about predefined first-order logic predicates of the environment.

**URL**: http://arxiv.org/pdf/2410.01929v1

**Published**: 2024-10-02

