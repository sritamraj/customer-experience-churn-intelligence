from pathlib import Path
import duckdb

DB = Path("data/olist.duckdb")
SQL_FILE = Path("sql/05_target_leakage_audit.sql")


def main():

    con = duckdb.connect(str(DB))

    print("=" * 70)
    print("TARGET LEAKAGE AUDIT")
    print("=" * 70)

    sql = SQL_FILE.read_text(encoding="utf-8")

    result = con.execute(sql).fetchdf()

    print("\nAudit results:\n")
    print(result.to_string(index=False))

    con.close()


if __name__ == "__main__":
    main()