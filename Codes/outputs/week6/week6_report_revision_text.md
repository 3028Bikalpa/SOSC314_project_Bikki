# Week 6 Report Revisions (Professor Feedback Addressed)

## Final Analytic Configuration and Stopping Justification (Replacement)

The dependent variable in this project is binary review sentiment (Y), coded as Y=1 for ratings 4–5 and Y=0 for ratings 1–2; rating 3 is excluded as a neutral/ambiguous midpoint. After filtering, the analytic sample contains 436,836 reviews: 316,271 positive (72.4%) and 120,565 negative (27.6%).

This binary coding was chosen to prioritize interpretability and actionable polarity (clear positive vs negative instructional signals). The tradeoff is loss of granularity from the original rating scale. To address this, a non-binary sensitivity check was run using a 4-class specification (ratings 1/2/4/5): weighted F1 dropped to 0.6717 versus 0.9210 in the binary setup, supporting binary coding for the project's primary explanatory objective while acknowledging reduced granularity.

The final selected model is Logistic Regression with TF-IDF (10,000 features, unigrams+bigrams, class_weight='balanced', C=1.0, solver='liblinear'). Stopping criterion: stop iterating when test-set improvement <=0.005 across two consecutive updates or when additional complexity reduces generalization.

## Limitations and Scope (Replacement)

1. Class imbalance (~2.6:1 positive:negative) may hide minority-class errors in aggregate metrics.
2. Causal inference limitation: this is observational text data, so coefficients are predictive associations and not causal estimates.
3. External validity limitation: data come from a single platform (RateMyProfessors), so transferability to other educational contexts is uncertain without external benchmarking.
4. Finite model-search budget means optimality is conditional on tested models/hyperparameters.
5. Negation sensitivity remains a challenge for bag-of-words assumptions; a controlled negation-scope experiment improved F1 from 0.9482 to 0.9577 and reduced negation-tagged errors by 17.71%.

## Future Improvements (Replacement)

1. Validate the model on external education-review datasets to test external validity.
2. Extend beyond binary outcomes with ordinal/multiclass models when interpretation needs finer rating granularity.
3. Compare the negation-scope enhancement with sequence-aware models covered in Stats 201 to evaluate whether contextual modeling provides additional gains beyond +0.0095 F1 observed here.
