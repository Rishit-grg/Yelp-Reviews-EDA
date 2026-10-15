# Yelp Reviews: Exploratory Data Analysis and Text Classification

An exploratory analysis of Yelp reviews, followed by binary text classification of 1-star and 5-star reviews. The project examines rating patterns, review length, review-vote features, and the performance of two Multinomial Naive Bayes approaches.

## Project overview

This notebook analyzes a local CSV file named `yelp.csv` containing 10,000 Yelp review records. It begins with a review of the data and descriptive statistics, then explores how review length and engagement-style vote counts vary by star rating. The final section frames extreme ratings as a text-classification problem: distinguishing 1-star from 5-star reviews.

## Questions explored

- How are review ratings distributed across the 1–5 star scale?
- Does review length vary by star rating?
- How do the average `cool`, `useful`, and `funny` vote counts differ by rating?
- How are these numeric review characteristics related at the rating-group level?
- How well can bag-of-words and TF–IDF Multinomial Naive Bayes models distinguish 1-star from 5-star review text?

## Dataset

The notebook reads `yelp.csv` and contains **10,000 records** with the following original columns:

| Column | Description inferred from the notebook |
| --- | --- |
| `business_id` | Business identifier |
| `date` | Review date (loaded as text) |
| `review_id` | Review identifier |
| `stars` | Integer star rating from 1 to 5 |
| `text` | Review text |
| `type` | Record type (shown as `review` in the preview) |
| `user_id` | Reviewer identifier |
| `cool` | Cool vote count |
| `useful` | Useful vote count |
| `funny` | Funny vote count |

All 10 original columns are non-null in the notebook’s dataset inspection. The analysis creates one additional feature, `text_length`, calculated as the character length of each review.

> The notebook does not record a source URL or licensing terms for `yelp.csv`. Before publishing, add the original dataset link and follow its redistribution terms. Keep the raw CSV out of version control unless redistribution is explicitly allowed.

## Methodology

1. Load and inspect the data with Pandas using `head()`, `info()`, and descriptive statistics.
2. Create `text_length` from the review text.
3. Explore the data with:
   - faceted histograms of text length, one panel per rating;
   - a text-length-by-rating box plot;
   - a rating count plot; and
   - a heatmap of correlations among rating-level means for `cool`, `useful`, `funny`, and `text_length`.
4. Restrict the classification task to reviews rated 1 or 5 stars.
5. Split review text into training and test sets with `test_size=0.33` and `random_state=42`.
6. Train and evaluate:
   - a bag-of-words `CountVectorizer` + `MultinomialNB` model; and
   - a `CountVectorizer` + `TfidfTransformer` + `MultinomialNB` pipeline.

## Notebook visualizations

The notebook produces the following inline charts:

- **Faceted text-length histograms** for each star rating
- **Box plot** of review text length by rating
- **Count plot** of review counts by rating
- **Annotated correlation heatmap** for rating-level mean numeric features

## Key findings

### Review characteristics by rating

The rating-level averages show that review text is longest for 2-star reviews (**842.3 characters** on average) and shortest for 5-star reviews (**625.0 characters** on average). The remaining averages are 826.5 characters for 1-star reviews, 758.5 for 3-star reviews, and 712.9 for 4-star reviews.

Average `useful` votes are highest for 1-star reviews (1.60) and 2-star reviews (1.56). Average `cool` votes are higher for 4-star (0.95) and 5-star (0.94) reviews than for 1-star reviews (0.58). These are descriptive group averages, not causal conclusions.

### Bag-of-words classifier

For the 1-star-versus-5-star test set (1,349 reviews), the count-vectorizer Multinomial Naive Bayes model achieved **0.90 accuracy**. It correctly classified 163 of 261 1-star reviews and 1,056 of 1,088 5-star reviews. Its class-level recall was 0.62 for 1-star reviews and 0.97 for 5-star reviews.

### TF–IDF pipeline result

The TF–IDF pipeline achieved **0.81 accuracy**, but this result is misleading if read alone: it predicted almost every test review as 5-star. It identified all 1,088 5-star reviews but only 1 of 261 1-star reviews (1-star recall rounds to 0.00). This demonstrates why accuracy should be read alongside the confusion matrix and class-level metrics, especially when the classes are unevenly represented.

## Tools

- Python
- Jupyter Notebook
- Pandas and NumPy
- Matplotlib and Seaborn
- scikit-learn (`CountVectorizer`, `TfidfTransformer`, `MultinomialNB`, and evaluation metrics)



