"""One-off cleanup: VOID model_CL2's Flau'Jae Johnson rows.

Before the 2026-07 fix, CL2's fuzzy name matching couldn't find her (props
spell "Flau'Jae", the box CSVs "Flau'jae", and she's a 2026 rookie absent
from the 2009-2025 files CL2 loaded) and fell back to blending every player
named Johnson since 2009. Those bets were generated from other players'
careers, so they're voided: result = 'VOID', profit = 0 (stake returned),
keeping the rows as an audit trail instead of deleting them.

Only CL2 is affected — other models either matched her correctly (CLCF1/2)
or correctly skipped her.

Safe by default: prints what it WOULD void and exits. Pass --apply (or set
VOID_APPLY=true) to commit.
"""

import os
import sys
import psycopg2

WHERE = "model_name = 'model_CL2' AND player_name = %s"
PARAMS = ("Flau'Jae Johnson",)


def main():
    apply = "--apply" in sys.argv or os.getenv("VOID_APPLY", "").lower() in ("1", "true", "yes")

    conn = psycopg2.connect(
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT", 5432),
    )
    cur = conn.cursor()

    cur.execute(
        f"""
        SELECT date, bet, COALESCE(result, 'UNGRADED'), amount, profit
        FROM predictions
        WHERE {WHERE}
        ORDER BY date;
        """,
        PARAMS,
    )
    rows = cur.fetchall()

    print(f"Target: model_CL2 rows for {PARAMS[0]}")
    print(f"Matching rows: {len(rows)}\n")
    if rows:
        print(f"{'date':>12} {'bet':>6} {'result':>8} {'amount':>7} {'profit':>8}")
        print("-" * 46)
        net = 0.0
        for d, bet, result, amount, profit in rows:
            net += float(profit or 0)
            print(f"{str(d):>12} {str(bet):>6} {result:>8} "
                  f"{'' if amount is None else f'{float(amount):.0f}':>7} "
                  f"{'' if profit is None else f'{float(profit):+.2f}':>8}")
        print(f"\nNet profit being voided: {net:+.2f}")

    if not rows:
        print("Nothing to void.")
        cur.close()
        conn.close()
        return

    if not apply:
        print("\nDRY RUN — no rows changed. Re-run with --apply (or VOID_APPLY=true) to commit.")
        cur.close()
        conn.close()
        return

    cur.execute(
        f"""
        UPDATE predictions
        SET result = 'VOID',
            profit = CASE WHEN amount IS NULL THEN NULL ELSE 0 END
        WHERE {WHERE};
        """,
        PARAMS,
    )
    changed = cur.rowcount
    conn.commit()
    print(f"\nVOIDED {changed} row(s).")

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
