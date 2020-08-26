# Movie Genre Classification & Similarity Calculation
Multi-label classification of movie genres and similar movie calculation via a given movie's plot summary.

Logistic regression and SVM classifiers were used for fast training, prediction and baseline establishment.
  For better results, transfer learning on a pre-trained BERT model was done with an added fully connected dense output layer. Similarity calculation was done on embeddings extracted from the first token in the last BERT hidden layer.   
    
  Embeddings were extracted only for english speaking movies.
## Results
#### Home Alone 2: Lost in New York 
##### Plot Summary
> After snarky youth Kevin McCallister (Macaulay Culkin) loses track of his father at the airport, he mistakenly gets 
>on a plane headed for New York City -- while 
>the rest of the McCallisters fly to Florida. 
>Now alone in the Big Apple, Kevin cons his way into a room at the Plaza Hotel and begins his usual antics. But when
> Kevin discovers that the Sticky Bandits (Joe Pesci, Daniel Stern) are on the loose, he struggles to stop them from 
>robbing an elderly man's toy store just before Christmas.
##### Movie Genre Classification
- Logistic Regression - Comedy (37.38%), Drama (19.06%), Indie (11.34%) , Thriller (9.59%)
- SVM - No genres
- BERT - Comedy (92.38%), Family (69.36%), Adventure (68.42%), Western (33.87%) Children's/Family (17.31%)
##### Similarity
BERT - cosine similarity
1) Rover Dangerfield, similarity score: 0.99
2) Dennis the Menace, similarity score: 0.98
3) To Grandmother's House We Go, similarity score: 0.97
4) Herbie Goes Bananas, similarity score: 0.96
5) The Great Muppet Caper, similarity score: 0.95
#### Joker
##### Plot Summary
> In Gotham City, mentally troubled comedian Arthur Fleck is disregarded and mistreated by society. He then embarks on a downward spiral of revolution and bloody crime. This path brings him face-to-face with his alter-ego: the Joker.
##### Genre Prediction
- Logistic Regression - Drama (80.61%), Crime Fiction (11.34%), Comedy (16.35%)
- SVM - Drama (100%), Crime Fiction(100%)
- BERT - Drama (71.26%), Thriller (67.59%), Crime Fiction (62.57%), Indie (60.89%), Action(51.91%)
##### Similarity
BERT (only english language movies)
1) Kill The Moonlight, similarity score: 0.93
2) Hobo With a Shotgun, similarity score: 0.88
3) Ashes to Ashes, similarity score: 0.87
4) I Got the Hook Up, similarity score: 0.87
5) Exterminator 2, similarity score: 0.87

#### The Dark Knight
##### Plot Summary
> With the help of allies Lt. Jim Gordon (Gary Oldman) and DA Harvey Dent (Aaron Eckhart), Batman (Christian Bale) has been able to keep a tight lid on crime in Gotham City. But when a vile young criminal calling himself the Joker (Heath Ledger) suddenly throws the town into chaos, the caped Crusader begins to tread a fine line between heroism and vigilantism.
##### Genre Prediction
- Logistic Regression - Drama (56.10%), Comedy (25.33%), Thriller (24.15%), Action/Adventure(15.11%)
- SVM - Action/Adventure (100%), Drama (100%), Thriller(100%)
- BERT - Action (95.23%), Thriller (73.57%), Action/Adventure (64.26%), Drama (30.51%), Crime Fiction (26.06%)
##### Similarity
BERT
1) Half Past Dead, similarity score: 0.97
2) Double Take, similarity score: 0.96
3) Batman, similarity score: 0.96
4) Wake of Death, similarity score: 0.96
5) Assault on Precinct 13, similarity score: 0.96
## About the Dataset
The [CMU Movie Summary Corpus](http://www.cs.cmu.edu/~ark/personas/) open dataset containing 42,306 movie plot summaries extracted from Wikipedia + aligned metadata extracted from Freebase.

## Methods
### Dataset Pre-procesing
- 53 unique genres out of 363 that appear at least 500 times in the entire dataset were kept
- Plot summaries were cleaned out of links, punctuation and synopsis tags were removed
### Model Training
#### Logistic Regression & Support Vector Machines
- Further pre-processing was done including: the removal of stopwords and stemming.
- Models were trained using a One vs Rest approach
#### Deep Learning
A pre-trained BERT model with an added output layer was trained on googlecolab using pytorch.
#### Training Results
- logistic regression - F1 score = 0.42
- SVM - F1 score = 0.44
- BERT - F1 score = 0.54
## Code Structure
-`movie_analysis.ipynb` - jupyter notebook containing the code used for training logistic regression and SVM classifiers
- `bert_model` - modules used for training a pytorch model
- `scripts` - scripts used for pre-processing/training/predicting
- `utils` - helper function and utils 
- `movie_classifier.py` - class used for predicting similar movies and genres




