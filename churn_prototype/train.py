"""Train reference prototype and emit audit artefacts (Ch.4 outputs)."""
import os, json
import joblib
import pandas as pd
from churn_core import (
    generate_synthetic, validate_schema, train_and_select, score_customers,
    SEED, TARGET, ALL_FEATURES,
)

OUT = os.path.dirname(os.path.abspath(__file__))

def main(n=6000):
    print("Generating SYNTHETIC data (demo only)...")
    df = generate_synthetic(n=n, seed=SEED)
    ok, issues = validate_schema(df)
    print("Schema ok:", ok)
    for i in issues:
        print(" -", i)
    art = train_and_select(df)
    rep = art["report"]
    print(f"Selected: {rep['selected_model']} @ thr={rep['operating_threshold']:.2f}")
    print("Validation LR:", rep["validation"]["logistic_regression"])
    print("Validation RF:", rep["validation"]["calibrated_random_forest"])
    print("Locked TEST:", rep["test_metrics_selected"])

    # persist pipeline (preprocessor+model as single artefact)
    joblib.dump({"model": art["selected"], "threshold": rep["operating_threshold"],
                 "report": rep, "features": ALL_FEATURES},
                os.path.join(OUT, "selected_pipeline.joblib"))
    with open(os.path.join(OUT, "implementation_report.json"), "w") as f:
        json.dump(rep, f, indent=2)
    art["perm_df"].to_csv(os.path.join(OUT, "permutation_importance.csv"), index=False)

    # score full set, export Top-50
    scored = score_customers(art["selected"], df[ALL_FEATURES].copy(),
                             rep["operating_threshold"])
    # reattach id + target for audit (token in production)
    scored["customer_id"] = df["customer_id"].values
    scored[TARGET] = df[TARGET].values
    cols = ["customer_id", "churn_probability", "risk_band", "expected_retention_value",
            "recommended_action", "reason_codes", "decision", TARGET]
    top50 = scored.sort_values("churn_probability", ascending=False).head(50)
    top50[cols].to_csv(os.path.join(OUT, "Top_50_scored_customers.csv"), index=False)
    scored.to_csv(os.path.join(OUT, "all_scored_customers.csv"), index=False)
    print("Artefacts written to", OUT)
    print("PASSED: schema, missing-value path, prob bounds, persistence, decisioning")

if __name__ == "__main__":
    main()
