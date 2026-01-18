# Predicting Professor Ratings from Student Reviews

## Research Question

**Can student-written RateMyProfessors comments be used to predict numerical professor ratings?**

This project uses a dataset from Hugging Face containing **480,342 observations** and **five variables**. The project is approached as a **supervised text-as-data task**, where multiple predictive models will be trained using student-written review text to predict numerical professor ratings. The models will be compared to determine which achieves the lowest prediction error.

During data cleaning, variables other than the review text and numerical rating will be removed. The variable **`difficulty`** may be retained if it is found to positively contribute to model performance.

This project aims to connect **subjective human judgment and emotion** (student-written comments) to a **measurable outcome** (numerical professor ratings).

---

## Unit of Analysis

The unit of analysis for this project is an **individual student-written RateMyProfessors review**.

---

## Data Source

This project uses a publicly available dataset from **Hugging Face**:

- Dataset: *RateMyProfessors Reviews*
- Link: https://huggingface.co/datasets/ZephyrUtopia/ratemyprofessors_reviews
- Access Method: Hugging Face Datasets API

The dataset is originally split into training and test sets on Hugging Face. All observations will be combined into a single dataset and later re-split according to project needs.

Although the dataset appears largely clean, additional preprocessing steps will include:
- Removing unused variables and retaining primarily **`text`** and **`rating`**
- Optionally retaining **`difficulty`** if it improves model performance
- Removing observations with empty or missing review text

---

## Feasibility Assessment

The dataset size is sufficient and does not pose a feasibility concern. However, several challenges are anticipated:

- **Sarcasm and nuanced language** in reviews may reduce model accuracy
- **Class imbalance**, particularly for ratings ending in `.5` (e.g., 1.5, 2.5, 3.5, 4.5)

To address these issues, observations with `.5` ratings may be removed, and models will be trained on the remaining data.

Overall, the dataset is well-suited for the project and provides a strong foundation for supervised text-based prediction models.
