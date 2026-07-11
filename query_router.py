# query_router.py

# words that signal a numeric/table question
NUMERIC_KEYWORDS = [
    "how much", "what is the", "total", "revenue", "sales",
    "profit", "loss", "income", "expense", "margin", "earnings",
    "eps", "debt", "cash", "cost", "percent", "%", "billion",
    "million", "growth", "increase", "decrease", "ratio"
]

def classify_query(question):
    """
    Returns "table" if numeric question, "text" if narrative question
    """
    question_lower = question.lower()

    for keyword in NUMERIC_KEYWORDS:
        if keyword in question_lower:
            return "table"

    return "text"


# test it
if __name__ == "__main__":
    tests = [
        "what is the total revenue",
        "what are the risk factors",
        "how much did operating expenses increase",
        "what does management say about competition",
        "what is the gross margin",
        "what markets does apple operate in"
    ]

    for q in tests:
        result = classify_query(q)
        print(f"{'📊' if result == 'table' else '📝'} [{result}] {q}")