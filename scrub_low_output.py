"""One-off cleanup: remove "Low Output" prediction rows produced by CLC1/CLC2/CLC3.

These models' Low Output calls underperform and skew the dashboard's pooled
Low Output stats. This deletes ONLY rows where:
    model_name IN ('model_CLC1', 'model_CLC2', 'model_CLC3_LEARN')
    AND note = 'Low Output'

model_CL2 also emits "Low Output" and is intentionally left untouched.

Safe by default: prints what it WOULD delete (with a year/result breakdown) and
exits without changing anything. Pass --apply (or set SCRUB_APPLY=true) to commit
the delete.
"""

import os
import sys
import psycopg2

TARGET_MODELS = ("model_CLCF3")
TARGET_NOTE = "Low Output"

WHERE = "model_name = ANY(%s) AND note = %s"
PARAMS = (list(TARGET_MODELS), TARGET_NOTE)


def main():
    apply = "--apply" in sys.argv or os.getenv("SCRUB_APPLY", "").lower() in ("1", "true", "yes")

    conn = psycopg2.connect(
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT", 5432),
    )
    cur = conn.cursor()

    cur.execute(f"SELECT COUNT(*) FROM predictions WHERE {WHERE};", PARAMS)
    total = cur.fetchone()[0]

    cur.execute(
        f"""
        SELECT EXTRACT(YEAR FROM date)::int AS yr,
               COALESCE(result, 'UNGRADED') AS result,
               COUNT(*) AS n,
               COALESCE(SUM(profit), 0) AS net_profit
        FROM predictions
        WHERE {WHERE}
        GROUP BY yr, result
        ORDER BY yr, result;
        """,
        PARAMS,
    )
    breakdown = cur.fetchall()

    print(f"Target: note = '{TARGET_NOTE}' for models {TARGET_MODELS}")
    print(f"Matching rows: {total}\n")
    if breakdown:
        print(f"{'year':>6} {'result':>10} {'rows':>6} {'net_profit':>12}")
        print("-" * 38)
        for yr, result, n, net in breakdown:
            print(f"{yr:>6} {result:>10} {n:>6} {float(net):>12.2f}")
        print()

    if total == 0:
        print("Nothing to delete.")
        cur.close()
        conn.close()
        return

    if not apply:
        print("DRY RUN — no rows deleted. Re-run with --apply (or SCRUB_APPLY=true) to commit.")
        cur.close()
        conn.close()
        return

    cur.execute(f"DELETE FROM predictions WHERE {WHERE};", PARAMS)
    deleted = cur.rowcount
    conn.commit()
    print(f"DELETED {deleted} row(s).")

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
