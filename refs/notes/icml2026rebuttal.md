Paper Decision
Decisionby Program Chairs01 May 2026, 00:58 (modified: 01 May 2026, 03:08)Program Chairs, AuthorsRevisions
Decision: Reject
Comment:
The paper presents a potentially interesting problem, inferring a selection rule from bags where only the aggregate count of satisfying instances is observed. However, several reviewers note deficiencies in the presentation and experimentation: the paper is too domain-specific for a general ML audience; comparisons are missing or weak; and the generality beyond the Littlewood-Richardson setting is unclear. In rebuttal, the authors suggest several potential improvements, but the reviewers are in agreement that these changes would be too substantial to support admission in this review cycle.

Official Review of Submission11692 by Reviewer FW7f
Official Reviewby Reviewer FW7f11 Mar 2026, 21:56 (modified: 07 Apr 2026, 18:45)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer FW7fRevisions
Summary:
The authors introduce a differentiable approach to recover the instance-level selection probabilities of a set of observations for which only the aggregate count is known. To that end, they formulate a generic counting and signed-counting underlying process and describe an implementation to recover instance-level selection probabilities specific to the Littlewood-Richardson formula. The approach is instantiated with a Transformer-based variant, which is evaluated once with and wiagnostic variant generalizes better.hout positional encodings, where empirical results sow that the position-

Strengths And Weaknesses:
Strengths

The method achieves a strong performance on OOD data,
The architectural choices are justified with ablation experiments, though a comparison to other pre-existing methods is missing.
Weaknesses:

The initial problem, as described in Section 3, seems extremely underspecified. From the objective, it is not clear what is supposed to be learned.
The objective of balancing predictive and discovery aspects are formulated extremely informally and thus difficult to assess in terms of soundness.
The specific focus on the Littlewood-Richardson problem is interesting. What is lacking, however, is a justification for whether this is generalizable to other comparable problems. For a revised version I would request the authors to include a general notion of rule-based selection into the problem formulation. The specific selector does not seem to be understandable nor particularly relevant to the general ML audience
The evaluation is also a concern. The work of Shukla et al. is mentioned [1] as a similar approach but not compared to? As far as I understand, it is at least applicable to the unsigned counting problem. In general, the write-up is quite dense and difficult to follow for readers not familiar with the Littlewood Richardson problem. The paper begins with a general formulation but quickly focuses on a very specific mathematical application. I recommend either strengthening the general perspective with additional experiments or fully framing the work as a method tailored to the LR problem.
Soundness: 2: fair
Presentation: 2: fair
Significance: 2: fair
Originality: 2: fair
Key Questions For Authors:
Why do you remove the features 
from the ID variant?
How can the model perform better even though it is missing this information? 
What is the impact of having the entropy regularizer or not, and does the strength need to be tuned? 
The Pieri rule is mentioned and visualized in Figure 2, though I dont fully understand it from that example. Where is its formal definition, and is there a human-understandable interpretation, e.g., “arm length < x & leg length < y”?
Limitations:
Yes

Overall Recommendation: 2: Reject: For instance, a paper with technical flaws, weak evaluation, inadequate reproducibility, incompletely addressed ethical considerations, or writing so poor that it is not possible to understand its key claims.
Confidence: 4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.
Compliance With LLM Reviewing Policy: Affirmed.
Code Of Conduct Acknowledgement: Affirmed.
Final Justification:
The paper tackles an interesting and relevant problem. On the positive side, the proposed solution seems to deliver strong results, including OOD settings. Yet, the weaknesses outweigh the strengths in this paper. Such as an insufficient problem description and an overly specialized focus on a single problem, which call into question the generality of the proposed approach. In its current form, the paper needs major revisions, and even after the rebuttal, these concerns remain.

Rebuttal by Authors
Rebuttalby Authors (Chul-hee Lee, Byung-Hak Hwang, Chanho Min)31 Mar 2026, 18:50 (modified: 31 Mar 2026, 22:29)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, AuthorsRevisions
Rebuttal:
We thank the reviewer for their detailed and constructive assessment. We appreciate the observation that our method performs well on OOD data and that our architectural choices are justified. Below we address your specific concerns and questions.

Regarding Generalizability and the Problem Formulation:

We acknowledge the reviewer's point that the paper's focus shifts quickly from a general framework to the specific Littlewood-Richardson (LR) problem. In our revision, we intend to provide a more rigorous, general definition of Latent Signed Counting. To demonstrate generalizability, we are developing a more accessible benchmark, such as a "Signed MNIST" task (calculating the sum of digits in a bag where different digits have different signed values). This will help bridge the gap between the specific mathematical application and a general ML audience.

Regarding the Comparison to Shukla et al. [1]:

We agree that a direct comparison to Shukla et al. is necessary. Algorithmically, our approach yields identical results for unsigned counting. However, our work generalizes this by viewing the counting process as a convolutional operation, which allows for "Signed Counting" (handling negative contributions and erasure).

We will incorporate a detailed discussion and baseline comparison in the revised version.

Response to Key Questions:

Q1 & Q2: Why remove absolute features and how does performance improve?

In AI for Math applications, models often overfit on absolute coordinates (e.g., the specific row and column numbers of cells in a partition). This memorization prevents the model from learning the underlying mathematical rule.

By removing absolute coordinates and using Relative Position Representations, we force the model to learn the logic of how cells relate to one another using only relative position between two cells. This approach reduces overfitting on the training grid and is precisely why the relative feature generalizes significantly better to Out-of-Distribution (OOD) data.

Q3: Impact of the Entropy Regularizer.Without the entropy regularizer, the selector tends to produce "soft" or ambiguous probabilities (e.g., 
). While these might sum up to the correct label during training, they fail to represent a clear selection rule. The regularizer incentivizes the model to make "hard" 
 or 
 decisions, which is essential for discovering discrete rules. We found that the model’s accuracy is quite robust to the exact strength of this regularizer, as long as it is present to push the predictions toward 0 or 1.

Q4: Formal Definition and Interpretation of the Pieri Rule. When coefficients are given as 
: 
 

We say that 
 is in the Pieri case if 
 for some 
, and the skew diagram 
 is a horizontal strip, i.e., 
 contains at most one cell in each column.

In this case, a classical Pieri rule gives an explicit description of the selected subsets 
 appearing in the factorization of 
.

 Replying to Rebuttal by Authors
Rebuttal Acknowledgement by Reviewer FW7f
Rebuttal Acknowledgementby Reviewer FW7f03 Apr 2026, 20:02Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors
Acknowledgement: (c) Partially resolved or unresolved, but the remaining concerns are not easily addressed in a short rebuttal - Please select this option sparingly and only when you believe that your questions concern the core tenets of the work, and addressing them requires a significant update to the paper.
Reasons:
I thank the authors for their elaborate rebuttal. The proposed changes will significantly improve the paper, but they require major revisions and restructuring. In particular, proposing a new benchmark without presenting the results is beyond the scope of a rebuttal. Thus, I keep my score.

Official Review of Submission11692 by Reviewer gnCt
Official Reviewby Reviewer gnCt10 Mar 2026, 23:48 (modified: 07 Apr 2026, 12:31)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer gnCtRevisions
Summary:
The authors propose a model to predict the aggregate count on the unseen bags. The model combines a differentiable probabilistic convolutional aggregator and a purely relative transformer. The method obtains good performance on a challenge, the conjectural (q,t)-Littlewood-Richardson formula.

Strengths And Weaknesses:
Strengths

The paper takes a good trial to apply the transformer and convolution to the classical mathematical challenges.
Weaknesses

The paper uses a very scientific presentation style. However, some motivation or explanations are missing for readers with different backgrounds. For instance, some explanations for Figure 4 are great appreciated.
The authors use very limited baselines in their experiments. It’s very hard to verify the effectiveness of the proposed model compared with existing methods or neural networks.
Soundness: 2: fair
Presentation: 2: fair
Significance: 2: fair
Originality: 3: good
Key Questions For Authors:
Why do you remove the absolute geometric channels from the token features?
How to understand the 
 in Relative bias.
During the experiment, do you consider the derivation under the different random seeds?
Limitations:
yes

Overall Recommendation: 3: Weak reject: A paper with clear merits, but also some weaknesses, which overall outweigh the merits. Papers in this category require revisions before they can be meaningfully built upon by others. Please use sparingly.
Confidence: 2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.
Compliance With LLM Reviewing Policy: Affirmed.
Code Of Conduct Acknowledgement: Affirmed.
Rebuttal by Authors
Rebuttalby Authors (Chul-hee Lee, Byung-Hak Hwang, Chanho Min)31 Mar 2026, 19:43 (modified: 31 Mar 2026, 22:29)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, AuthorsRevisions
Rebuttal:
We thank the reviewer for recognizing the originality of our work and our effort to apply Transformers and Convolutions to classical mathematical challenges. We appreciate the feedback regarding the presentation and the need for better motivation for a broader audience.

Regarding Motivation and Figure 4:

We acknowledge that the presentation style is dense. The core motivation is to move beyond "black-box" regression and create a model that learns exact, discrete rules. To enable robust rule discovery in mathematical contexts, we have proposed a semi-labeled learning framework designed to pinpoint which specific objects in a bag are significant to the underlying theorem or conjecture.

Central to this selection process is our another proposal: Purely-Relative Transformer (PRT). Unlike standard architectures that often overfit to the specific dimensions of the training set, the PRT is engineered to capture the intrinsic geometric logic of mathematical rules. This allows the model to extrapolate successfully to Out-of-Distribution (OOD) data, discovering rules that remain valid even on mathematical objects much larger or more complex than those seen during training.

Figure 4 Explanation: This figure illustrates our end-to-end pipeline. First, mathematical objects are tokenized. These tokens are processed by a Purely-Relative Transformer which identifies relationships between them without knowing their absolute coordinates. Then, tokens are grouped into "bags" (the instances). Finally, our Convolutional Aggregator takes the selection probabilities of these instances and performs a discrete summation using 1D convolution to match the target aggregate count. We will add a much more detailed caption to this figure in the final version to make it self-contained.

Regarding the Novelty of our Formulation and Baselines:

We would like to emphasize that, to the best of our knowledge, this work introduces a novel formulation of Signed Integer Counting, where the objective is to recover latent instance-level rules under exact integer sum-and-cancellation constraints. Since this problem setting has not been previously established in the literature, there are no existing 'State-of-the-Art' models specifically designed for it. This lack of a direct work makes the selection of standard baselines for signed counting particularly challenging

However, we do now realize that our proposed selector model, the Purely-Relative Transformer (PRT), can be compared with numerous other baselines beyond the basic Transformer. In our future work and revisions, we will provide experimental results comparing our approach with more novel models acting as the selector.

Response to Key Questions:

Q1: Why do you remove the absolute geometric channels from the token features?

In mathematical datasets, the absolute position of an object (like a cell in a partition) often correlates with specific training samples. If we include absolute coordinates, the Transformer tends to memorize which coordinates were selected in the training set rather than learning the geometric rule. By removing these channels and using only relative distances, we force the model to learn rules that are coordinate-invariant, allowing it to generalize to much larger, unseen mathematical structures (OOD extrapolation).

Q2: How to understand the 
 in Relative bias.

Since our model is designed to generalize to mathematical objects of any size, it must be independent of absolute coordinates. Consequently, our Purely-Relative Transformer does not use standard absolute Positional Encodings (which would tie the model to the specific grid size of the training data).

Instead, we inject geometric information directly into the attention mechanism via the relative bias 
. This bias is a function of the relative distance between two cells, 
. By conditioning the attention on 
 rather than the absolute positions 
, we ensure the model learns the structural relationship between cells without being anchored to a specific location on the coordinate plane.

Q3: During the experiment, do you consider the derivation under the different random seeds?

Yes, we considered the impact of initialization. All results reported in our tables represent the mean of 5 independent runs with different random seeds. We also included the standard deviation in parentheses.

We hope these explanations clarify the motivation and technical choices of our work. Thank you for your constructive review.

 Replying to Rebuttal by Authors
Rebuttal Acknowledgement by Reviewer gnCt
Rebuttal Acknowledgementby Reviewer gnCt03 Apr 2026, 23:32Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors
Acknowledgement: (c) Partially resolved or unresolved, but the remaining concerns are not easily addressed in a short rebuttal - Please select this option sparingly and only when you believe that your questions concern the core tenets of the work, and addressing them requires a significant update to the paper.
Reasons:
Thank you for the authors' response. I will maintain my current score and have no further questions.

Official Review of Submission11692 by Reviewer RQKg
Official Reviewby Reviewer RQKg08 Mar 2026, 15:29 (modified: 07 Apr 2026, 12:31)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer RQKgRevisions
Summary:
The manuscript is poorly written and requires substantial revision. The references are old, like in the field of mathmatical discovery and multiple instance learning. No any comparion results in experiments with the methods they mentioned in related works.

Strengths And Weaknesses:
The paper is poorly written. The presentation is confused. The experiments are insufficient.

Soundness: 2: fair
Presentation: 1: poor
Significance: 1: poor
Originality: 2: fair
Key Questions For Authors:
Introduction should be rewritten.

Limitations:
Review the related works more and recently.

Overall Recommendation: 3: Weak reject: A paper with clear merits, but also some weaknesses, which overall outweigh the merits. Papers in this category require revisions before they can be meaningfully built upon by others. Please use sparingly.
Confidence: 1: Your assessment is an educated guess. The submission is not in your area, or the submission was difficult to understand. Math/other details were not carefully checked.
Compliance With LLM Reviewing Policy: Affirmed.
Code Of Conduct Acknowledgement: Affirmed.
Final Justification:
This study proposed a method of instance-level selection, which supports the multi-instance learning. What's i concerns are the presentation problem and the experiments. For presenation, the paper writting is confused. As in the INTRODUCTION where there are gaps between paragraphs, so that the logical line is stacked. For experimets, since the goal in the abstract is for instance selection for MIL, there are no comparsion with other MIL methods. Other comparsion experiments are also less, like the OOD evaluation. Finally, according to author's response to my questions and other questions, i update my score while i stack on my confidence score. Thanks.

Rebuttal by Authors
Rebuttalby Authors (Chul-hee Lee, Byung-Hak Hwang, Chanho Min)31 Mar 2026, 20:57 (modified: 31 Mar 2026, 22:29)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, AuthorsRevisions
Rebuttal:
We thank the reviewer for the feedback. We are sorry that the current draft did not communicate the contribution clearly enough.

We will substantially improve the writing and overall exposition, especially in the Introduction, to make the problem setting, motivation, and main contributions much clearer. We will also revise the presentation throughout the paper so that the method, assumptions, and experimental goals are easier to follow.

We will strengthen the related-work section by including more recent references in mathematical discovery and weakly supervised / count-based learning, and by positioning our work more clearly relative to these directions.

We respectfully believe that the core formulation is novel, but we acknowledge that this was not presented convincingly enough in the current draft. In the revision, we will improve the clarity, positioning, and experimental support of the paper.

 Replying to Rebuttal by Authors
Rebuttal Acknowledgement by Reviewer RQKg
Rebuttal Acknowledgementby Reviewer RQKg01 Apr 2026, 11:40Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors
Acknowledgement: (b) Partially resolved - I have follow-up questions for the authors.
Reasons:
Authors has done revision to my concerns. But i cannot get the revised contents. Thanks for responses.

I can improve my socre to weak reject. However, the result in Tabal1-4, the comparsion requires to have more methods. LIke OOD, there are lost of methods. They can be compared to show the advantage in this application.

Official Review of Submission11692 by Reviewer zMzo
Official Reviewby Reviewer zMzo05 Mar 2026, 05:53 (modified: 07 Apr 2026, 12:31)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer zMzoRevisions
Summary:
The study evaluates the application of an ML architecture in the context of Multiple Instance Learning, i.e. an aggregate count is provided as the label to a series of unseen variables. The method is applied to the (q, t)-Littlewood–Richardson formula with a 98.5% test accuracy.

The described approach consists of an ablated transformer which produces a selection probability for each tokenized input. These PMFs are then passed through a 1D convolutional layer for the final aggregate.

Strengths And Weaknesses:
Strengths:

The problem and approach is well-defined.
The paper is well-structured and well-written.
The assumptions of their method are empirically verified through ablation experiments.
To the best of my knowledge, this approach is entirely novel.
Weaknesses:

In section 3.3, the paper mentions that one of their primary goals is the interpretability of their method, yet this aspect is not mentioned in their results at all.
The results are compared to a few other, naïve approach like a pure transformer, but not to the existing SotA.
The authors do not provide the relevant code for their approach, limiting the reproducibility of their findings.
While a solid, novel contribution, it is unclear the findings listed are significant enough to make much of an impact on the field.
Soundness: 2: fair
Presentation: 3: good
Significance: 2: fair
Originality: 3: good
Key Questions For Authors:
Are there any experiments you can point to that makes use of the interpretability of your approach?

How the current state-of-the-art methods perform on the (q, t)-Littlewood–Richardson formula? For instance, Attention-based Deep Multiple Instance Learning (Ilse et al., 2018) represents a strong baseline for Multiple Instance Learning and cites several more advanced approaches compared to a naive transformer (see Section 2.2 of the paper). While I am not entirely certain about the direct applicability of these methods in your setting, it would be helpful if you could clarify whether such approaches have been considered, and if so, how they compare to your proposed method.

Limitations:
The authors briefly mention a few minor limitations with the approach, but there is no dedicated limitations section and clear cases such as the lack of comparison to the SotA are not mentioned.

Overall Recommendation: 3: Weak reject: A paper with clear merits, but also some weaknesses, which overall outweigh the merits. Papers in this category require revisions before they can be meaningfully built upon by others. Please use sparingly.
Confidence: 2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.
Compliance With LLM Reviewing Policy: Affirmed.
Code Of Conduct Acknowledgement: Affirmed.
Rebuttal by Authors
Rebuttalby Authors (Chul-hee Lee, Byung-Hak Hwang, Chanho Min)31 Mar 2026, 20:34 (modified: 31 Mar 2026, 22:29)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, AuthorsRevisions
Rebuttal:
We thank the reviewer for the thoughtful and constructive feedback. We especially appreciate the reviewer’s recognition of the novelty of our formulation. We also agree that there are aspects of the current draft that should be clarified and strengthened.

Regarding Interpretability:

We agree that the current version does not present the interpretability aspect strongly enough in the Results section. In our setting, interpretability is not primarily intended as a post-hoc explanation of a bag-level prediction, but rather as the discovery of a plausible instance-level selection rule that explains the observed aggregate.

Our intended use of interpretability is that these selected objects may provide mathematicians with intuition about a rule that is consistent across different mathematical structures, and may eventually guide theorem formulation or proof. We will revise the paper to make this motivation clearer and to highlight the current quantitative evidence more explicitly.

Regarding Baselines and Comparison to Existing MIL Methods:

We agree with the reviewer that the current set of baselines is not broad enough. In particular, using only a standard Transformer as the main selector baseline is insufficient, and we now realize that the proposed Purely-Relative Transformer can and should be compared against a wider range of models acting as selectors. Given more time, we intend to include additional experiments with stronger and more modern baselines in future revisions.

At the same time, we would like to clarify that methods such as Attention-based Deep MIL are not directly aligned with our problem setting. While they share the same broad spirit of learning from bag-level supervision, the output space and aggregation objective are significantly different. Attention-based Deep MIL is formulated for binary bag classification: it aggregates instance embeddings to predict whether a bag label is 0 or 1. Our framework instead addresses signed integer counting, where the target is not a binary label but an exact integer-valued aggregate with possible cancellation. For this reason, standard mean/max/attention pooling methods are not direct substitutes for our method.

Nevertheless, we consent with the reviewer’s broader point that stronger MIL-inspired selectors are relevant points of comparison.

Regarding Code and Reproducibility:

We agree that the lack of a public implementation is a weakness of the current submission. At the moment, our code is still in a research-prototype state and requires substantial refactoring and cleaning before release. We are actively working on this and will try to upload the code to GitHub as soon as possible.

Regarding Significance:

We appreciate the reviewer’s concern regarding significance. We would like to clarify that our main goal is not simply to obtain higher accuracy on the (q,t)-Littlewood–Richardson problem itself. Rather, we intend to highlight the broader formulation and tool: namely, that latent counting framework for weakly supervised rule discovery in mathematics and transformer model that is structured for out of distribution prediction. In this sense, the (q,t)-LR problem is intended as a challenging case study rather than the only target application.

Response to Key Question 1: Are there any experiments you can point to that makes use of the interpretability of your approach?

At present, the interpretability of our method is used more as a scientific observation tool. The main interpretable object produced by the model is the instance-level selection pattern, not merely the bag-level prediction. Our view is that these selections can help mathematicians identify a rule that is consistent across many mathematical objects, thereby supporting mathematical intuition.

The clearest current evidence is the Pieri evaluation, where a known reference selection exists. In that setting, we observe strong agreement between the predicted selected set and the classical structure, including 100% inclusion and 66.08% exact-set match without explicit learning on these rules.

Question 2: How the current state-of-the-art methods perform on the (q, t)-Littlewood–Richardson formula? For instance, Attention-based Deep Multiple Instance Learning (Ilse et al., 2018) represents a strong baseline for Multiple Instance Learning and cites several more advanced approaches compared to a naive transformer (see Section 2.2 of the paper).

We realized that we have not yet included a sufficiently broad comparison, and we acknowledge that stronger baselines should be considered. However, we would like to emphasize that methods such as Attention-based Deep MIL are designed for a different output space and objective.

We hope these clarifications address the reviewer’s concerns. We are grateful for the constructive suggestions, and we believe they will help us improve both the clarity and the empirical scope of the paper.

 Replying to Rebuttal by Authors
Rebuttal Acknowledgement by Reviewer zMzo
Rebuttal Acknowledgementby Reviewer zMzo03 Apr 2026, 19:53Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors
Acknowledgement: (c) Partially resolved or unresolved, but the remaining concerns are not easily addressed in a short rebuttal - Please select this option sparingly and only when you believe that your questions concern the core tenets of the work, and addressing them requires a significant update to the paper.
Reasons:
I thank the authors for their thorough rebuttal and for clarifying several of the concerns I raised. While some misunderstandings have been resolved, I remain unconvinced on several of the more substantive issues, particularly those related to reproducibility and novelty.

My overall assessment therefore remains unchanged: although this work represents a solid contribution, I do not believe it fully meets the bar for ICML. As such, I am maintaining my weak reject rating. That said, I believe the paper could still be a valuable contribution in a more suitable venue.
