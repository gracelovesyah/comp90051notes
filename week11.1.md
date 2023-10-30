# U-PGM
This note is completed with the assistance of [ChatGPT](https://chat.openai.com/c/7887d9af-ce1c-4551-8f91-c576874448be)

**Undirected Probabilistic Graphical Models (U-PGMs or Markov Random Fields):**

1. **Definition and Structure:**
   - U-PGMs are undirected graphical models parameterized by positive-valued functions of the variables and a global normalization.
   - They are also known as Markov Random Fields (MRFs).

2. **Difference between Directed and Undirected PGMs:**
   - **Undirected PGMs:**
     - Edges are undirected.
     - Each node represents a random variable.
     - Each clique in the graph has a "factor" associated with it.
     - The joint probability is proportional to the product of these factors.
   - **Directed PGMs:**
     - Edges have a direction.
     - Each node represents a random variable.
     - Each node has a conditional probability based on its parents in the graph.
     - The joint probability is the product of these conditionals.

3. **Maximal Clique:**
   - A clique is a set of fully connected nodes.
   - A maximal clique is the largest clique in the graph, meaning no additional nodes can be added to it while maintaining full connectivity.

4. **Conversion from Directed to Undirected PGMs:**
   - Copy nodes.
   - Make edges undirected.
   - "Moralize" parent nodes by connecting them.

5. **Advantages of U-PGMs:**
   - They are a generalization of directed PGMs.
   - They offer a simpler means of modeling without the need for per-factor normalization.
   - General inference algorithms often use the U-PGM representation.

6. **Disadvantages:**
   - They have slightly weaker independence assumptions.
   - Calculating the global normalization term (Z) can be intractable, except for specific structures like chains or trees.

---

**Conditional Random Fields (CRFs):**

1. **Definition:**
   - CRFs are a type of U-PGM tailored for structured prediction, especially for sequences.

2. **Applications:**
   - Used for tasks like part-of-speech tagging, where observed outputs are words and states are tags like noun, verb, etc.

3. **Discriminative Nature:**
   - CRFs are discriminative models, focusing on modeling the boundary between classes. They model the conditional probability $ P(Q|O) $, where $ Q $ is the sequence of states and $ O $ is the sequence of observations.

---

**Hidden Markov Models (HMMs):**

1. **Definition:**
   - HMMs are generative models representing sequences of observations and their underlying states.

2. **Parameters:**
   - Transition probability matrix (A).
   - Output probability matrix (B).
   - Initial state distribution (Π).

3. **Applications:**
   - Used in speech recognition, biological sequence analysis, and part-of-speech tagging.

4. **Fundamental Tasks:**
   - **Evaluation:** Determine the likelihood of an observation sequence given an HMM.
   - **Decoding:** Find the most probable hidden state sequence for a given observation sequence.
   - **Learning:** Adjust the parameters of the HMM based on observed data.

---

**Applications in Computer Vision:**

1. **Pixel Labeling Tasks:**
   - Tasks include semantic labeling, figure-ground segmentation, and denoising.
   - The hidden states represent the semantics of the image, while pixels are the observed outputs.

2. **Relation to HMMs:**
   - The tasks in computer vision can be seen as spatial versions of HMMs, where the spatial structure of an image is analogous to the temporal structure in HMMs.


### Example
In the context of the sentence "I love Machine Learning" and its inferred sequence of tags "noun, verb, noun, noun" using an HMM:

1. **O (Observation Sequence):**
   - $ O $ represents the sequence of observed data. 
   - In this example, $ O $ is the sentence "I love Machine Learning". It's the data that you can directly observe and want to infer hidden states (tags) for.

2. **μ (HMM Model):**
   - $ \mu $ represents the HMM model itself, which includes:
     - **State Set:** The possible parts of speech tags (e.g., noun, verb, adjective, adverb, etc.).
     - **Transition Probabilities (A):** The probabilities of transitioning from one part of speech to another. For instance, the probability of a noun being followed by a verb, a verb being followed by an adjective, etc.
     - **Emission Probabilities (B):** The probabilities of a particular word being emitted for a given part of speech. For example, the probability that the word "love" is a verb, the probability that "I" is a noun, etc.
     - **Initial State Probabilities (Π):** The probabilities of starting a sentence with each possible part of speech.

In the given example, the HMM model $ \mu $ would have been trained on a large corpus of sentences with known parts of speech tags. This training allows the model to learn the transition and emission probabilities, which it then uses to infer the sequence of tags for new sentences like "I love Machine Learning".

---

In the context of the sentence "I love Machine Learning" and its inferred sequence of tags "noun, verb, noun, noun" using an HMM, let's break down the tasks of Evaluation, Decoding, and Learning:

---
## HMM
### Example cont. Evaluation, Decoding, Learning
1. **Evaluation:**
   - **Task:** Given an HMM model $ \mu $ and an observation sequence $ O $, determine the likelihood $ Pr(O|\mu) $.
   - **In this example:** The task would be to compute the likelihood of observing the sentence "I love Machine Learning" given the HMM model $ \mu $ (which has been trained on a corpus of sentences with known parts of speech tags). Essentially, how probable is this sequence of words given the model we have?

2. **Decoding:**
   - **Task:** Given an HMM model $ \mu $ and an observation sequence $ O $, determine the most probable hidden state sequence $ Q $.
   - **In this example:** The task is to find the most likely sequence of parts of speech tags (hidden states) for the sentence "I love Machine Learning" given the HMM model $ \mu $. The result, in this case, is "noun, verb, noun, noun". This is the sequence of tags that the model believes is most likely to have generated the observed sequence of words.

3. **Learning:**
   - **Task:** Given an observation sequence $ O $ and a set of states, learn the parameters $ A $ (transition probabilities), $ B $ (emission probabilities), and $ \Pi $ (initial state probabilities).
   - **In this example:** If we were given a large corpus of sentences (like "I love Machine Learning") with their corresponding parts of speech tags (like "noun, verb, noun, noun"), the task would be to adjust or train the HMM model $ \mu $ to best fit this data. This involves estimating the transition probabilities between tags, the probabilities of words being emitted for given tags, and the probabilities of starting a sentence with each possible tag.

---

**Summary:**

In the context of the example:
- **Evaluation** is about assessing how well the model explains the observed sentence.
- **Decoding** is about finding the best sequence of tags for the sentence given the model.
- **Learning** is about adjusting the model based on a dataset of sentences and their corresponding tags to make it more accurate for future predictions.

---
## Pixel labelling tasks in Computer Vision

## CRF
A practical example using **Part-of-Speech Tagging** with Conditional Random Fields (CRFs):

---

**Scenario:**
Imagine we have a sentence, and we want to predict the part-of-speech (POS) tag for each word in the sentence. The sentence is:

**Sentence (Observations, O):** "She enjoys reading books."

---

**Task:**
Given the sentence, we want to predict the POS tag for each word. The possible POS tags include nouns (N), verbs (V), pronouns (PR), adverbs (ADV), and prepositions (P), among others.

---

**Using CRFs:**

1. **Features:** CRFs rely on features to make predictions. For POS tagging, features might include:
   - The current word (e.g., "enjoys").
   - The previous word (e.g., "She").
   - The next word (e.g., "reading").
   - Whether the current word is capitalized.
   - Whether the current word is a known verb or noun, etc.

2. **Model Training:** Using a labeled dataset (sentences with known POS tags), we train the CRF to understand the relationships between words and their corresponding POS tags based on the features.

3. **Prediction:** Once trained, given a new sentence, the CRF can predict the POS tags for each word based on the features and the learned relationships.

---

**Predicted Tags (States, Q):**
Using our trained CRF model on the sentence "She enjoys reading books.", we might get the following POS tags:

"She" (PR) - Pronoun
"enjoys" (V) - Verb
"reading" (V) - Verb
"books" (N) - Noun
"." (Punctuation)

---

**Summary:**
In this example, the CRF uses the features of each word in the sentence, along with the learned relationships from the training data, to predict the most likely POS tag for each word. The advantage of CRFs over other models like HMMs is that they can consider a wide range of features and capture complex dependencies between words, making them particularly effective for tasks like POS tagging.