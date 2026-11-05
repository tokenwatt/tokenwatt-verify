"""Versioned thresholds for `conform`. A change here gets a new VERSION; old reports keep the version they
were graded with (it is written into every report), so a reading is never silently re-graded.

The values follow the TokenWatt Inference Endpoint Conformance draft v0 (2026-11-05) where it states one
(P1: ±10 tokens per request, ±1% over the suite; P2: no request above list), and are TokenWatt's choice
otherwise; each is listed with the check that uses it.
"""

VERSION = "conform-thresholds-2026-11-05.1"

THRESHOLDS = {
    # I2 tokenizer fingerprint: billed differential counts of the corpus vs the claimed family's reference.
    "I2": {
        "pass_total_ratio": 0.01,       # |sum(measured) / sum(reference) - 1| <= 1% -> same tokenizer
        "warn_total_ratio": 0.03,       # within 3% -> close but not identical (warn); beyond -> fail
        "pass_median_abs_tokens": 2,    # and the median per-sample difference is at most 2 tokens
        "indistinguishable_ratio": 0.002,  # other families within 0.2% of the claimed one: can't separate
        "min_samples": 4,
    },
    # P1 token counts (draft §6): per request and over the suite.
    "P1": {
        "per_request_tokens": 10,
        "suite_ratio": 0.01,
        "output_echo_tokens": 10,       # completion tokens of a verbatim echo vs the input count of the same text
        "baseline_warn_tokens": 150,    # a one-line message billed above this: something is added to every prompt
    },
    # P2 prices (draft §6): no request above list. Tolerance only for rounding of the reported cost.
    "P2": {"max_over_list_ratio": 0.005, "abs_rounding_usd": 0.000002},
    # P3 cache: share of the repeated prefix that must be reported as cached on a repeat.
    "P3": {"min_cached_share": 0.5, "prefix_chars": 24000, "repeats": 3, "max_over_cached_price_ratio": 0.02},
    # I4 capability fingerprint: one-sided Fisher exact test, endpoint vs reference rates.
    "I4": {"fail_p": 0.01, "warn_p": 0.05, "min_answered": 10},
    # B2 truncation: the request's max_tokens.
    "B2": {"max_tokens": 16},
    # B5 context: default (short) size, and the long-context fill of the advertised window.
    "B5": {"short_prompt_tokens": 8000, "long_fill": 0.95, "depths": [0.25, 0.50, 0.90], "min_reported_share": 0.9},
    # B6 streaming parity.
    "B6": {},
    # S1 service: sample count is a CLI option; this is the output length asked for.
    "S1": {"max_tokens": 400},
}
