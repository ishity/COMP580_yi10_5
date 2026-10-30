"""The exact rules: tokens, stopwords, TF-IDF scores in decimal arithmetic, ranking and display.

src/plugin/rule.py and verifier/rule.py hold this same file, so the tools and the verifier decide alike.
"""

import re
import unicodedata
from decimal import ROUND_HALF_EVEN, Context, Decimal

NON_ALNUM = re.compile(r"[^a-z0-9]+")
MIN_LENGTH = 3  # a term has at least this many characters
MAX_K = 30  # the task's k is from 1 to MAX_K
PRECISION = 40  # significant digits of every score computation
PLACES = Decimal("0.000001")  # displayed scores are quantized to 6 decimal places, ROUND_HALF_EVEN
CONTEXT = Context(prec=PRECISION, rounding=ROUND_HALF_EVEN)

# Common English function words, already normalized by stem (so "does" is "doe" and "always" is "alway").
# Tokens shorter than MIN_LENGTH are dropped anyway, so none is listed.
STOPWORDS = frozenset(
    {
        "about", "above", "across", "after", "again", "against", "all", "almost", "alone", "along", "already", "also",
        "although", "alway", "among", "and", "another", "any", "anyone", "anything", "are", "around", "because", "been",
        "before", "being", "below", "beside", "between", "both", "but", "can", "cannot", "could", "did", "doe", "doing",
        "done", "down", "during", "each", "either", "else", "enough", "etc", "even", "ever", "every", "few", "for",
        "from", "further", "had", "hardly", "has", "have", "having", "her", "here", "herself", "him", "himself", "his",
        "how", "however", "into", "its", "itself", "just", "least", "less", "like", "many", "may", "might", "more",
        "most", "mostly", "much", "must", "myself", "near", "neither", "never", "nevertheless", "none", "nor", "not",
        "now", "off", "often", "once", "one", "only", "onto", "other", "otherwise", "our", "ourselve", "out", "over",
        "own", "per", "perhap", "quite", "rather", "same", "several", "shall", "she", "should", "since", "some", "such",
        "than", "that", "the", "their", "them", "themselve", "then", "there", "therefore", "these", "they", "this",
        "those", "though", "through", "throughout", "thus", "too", "toward", "under", "unless", "until", "upon", "very",
        "via", "was", "well", "were", "what", "whatever", "when", "whenever", "where", "wherea", "whether", "which",
        "while", "who", "whom", "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your",
        "yourself", "yourselve",
    }
)


def stem(token):
    """A token longer than 3 characters ending in "s", but not "ss", "us" or "is", without that "s"."""
    if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def normalize(text):
    """Plugin 2's normalizer: NFKC, lowercase, every character outside a-z and 0-9 a space, then stem each token."""
    return [stem(token) for token in NON_ALNUM.sub(" ", unicodedata.normalize("NFKC", text).lower()).split()]


def is_term(token):
    """Whether a normalized token is a term: at least MIN_LENGTH characters, not all digits, not a stopword."""
    return len(token) >= MIN_LENGTH and not token.isdigit() and token not in STOPWORDS


def terms(text):
    """The text's terms, in order, repeats kept: its normalized tokens that are terms."""
    return [token for token in normalize(text) if is_term(token)]


def score(tf, df, n):
    """A term's score: its tf summed over the selected papers, times ln(n / df) over a scope of n papers, df of
    which contain it, at PRECISION significant digits."""
    if not 1 <= df <= n or tf < 1:
        raise ValueError(f"a term needs tf >= 1 and 1 <= df <= N, got tf {tf}, df {df}, N {n}")
    return CONTEXT.multiply(Decimal(tf), CONTEXT.divide(Decimal(n), Decimal(df)).ln(CONTEXT))


def ranking_key(term, value):
    """The ranking order: score descending, then term ascending."""
    return (-value, term)


def ranked(counts, n):
    """``counts`` (term -> (tf, df)) as (term, tf, df, score) by ranking_key."""
    scored = [(term, tf, df, score(tf, df, n)) for term, (tf, df) in counts.items()]
    return sorted(scored, key=lambda row: ranking_key(row[0], row[3]))


def display(value):
    """A score as shown: quantized to 6 decimal places, ROUND_HALF_EVEN, as a string such as "12.345679"."""
    return str(value.quantize(PLACES, rounding=ROUND_HALF_EVEN, context=CONTEXT))
