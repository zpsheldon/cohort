# cohort

Text analysis and user-to-user recommendation using NLP and machine learning.


### Goal

Our overall goal is to automatically assess the qualities of someone's profile and recommend similar people for them to meet.

### Data

We used data from tags that each person entered into distinct categories: 

- Experience
- Can Help With (Skills)
- Objectives
- Interests

Note that neither the raw nor processed data are included in the repo for the sake of privacy.

### Model

We used an open-source model that was trained by Google on about 100 billion words, and learned the relationships between words (i.e. king is to man as queen is to woman). 

Download the Google word2vec model here: https://code.google.com/archive/p/word2vec/

The model provides a way to represent words as collections of numbers, thereby allowing us to compare the similarities and differences between text.

### Methods

We cleaned and processed the tag data for each user, then put each tag into the model and found the average representation of that user's profile. 

Then we measured the similarity of each pair of users' profiles and ranked them, generating a list of recommendations. 

Top recommendations were validated by manually comparing how many shared tags they had.

### Caveats

This is a first-pass that was hacked together over the course of a week, so it is certainly not the most powerful model available. 

Better results will be possible with a more sophisticated model, as well as with more data from users who chose not to provide answers in some of the categories.
