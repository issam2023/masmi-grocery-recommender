# Association Rules and Apriori

**Abdelhafid Masmi · Vanier College · Fall 2026**

This homework explores synthetic grocery transactions using Apriori
and builds a recommendation system from saved association rules.

## View the work

- [Documented Jupyter notebook](Abdelhafid-Masmi-Apriori-Documented.ipynb)
- [Online Streamlit application](https://masmi-grocery-recommender.streamlit.app)

## Steps

1. Generate and inspect transactions.
2. Convert baskets into a boolean matrix.
3. Find frequent itemsets using Apriori.
4. Compare support, confidence, and lift.
5. Recommend products using matching rules.
6. Save and reload rules as JSON.
7. Evaluate coverage and discuss limitations.

## Results

| Measure | Result |
|---|---:|
| Shopping baskets | 3,000 |
| Products | 44 |
| Minimum support | 0.02 |
| Minimum confidence | 0.30 |
| Saved rules with lift above 1 | 2,124 |
| Catalogue coverage | 50% |

## Files

- `Abdelhafid-Masmi-Apriori-Documented.ipynb`: code, explanations, and results.
- `transactions.csv`: synthetic transaction dataset.
- `association_rules.json`: saved rules and metadata.
- `app.py`: interactive recommendation application.
- `requirements.txt`: application dependencies.

## Run locally

Install the application dependencies with:
`python -m pip install -r requirements.txt`

Start the application with:
`python -m streamlit run app.py`

For the notebook, install pandas, NumPy, mlxtend, and JupyterLab.
Open the notebook and run all cells from the beginning.

## Limitations

The data is synthetic. Association rules describe co-purchases,
not causation or guaranteed preferences. Lift above 1 alone does
not establish statistical reliability.

Some recommendations use rules with multiple consequent products.
Their metrics describe the complete rule, rather than each product
individually. Popularity fallback suggestions are labelled separately.
