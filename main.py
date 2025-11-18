import sqlite3


def main():
    # check_env.py

    try:
        import sqlite_vec

        print(f"sqlite-vec version: {sqlite_vec.__version__}")
    except ImportError as e:
        print("sqlite-vec import failed:", e)
        exit(1)

    conn = sqlite3.connect(":memory:")
    conn.enable_load_extension(True)
    try:
        sqlite_vec.load(conn)
        print("✓ sqlite-vec extension loaded successfully")
    except Exception as e:
        print("Extension load warning (expected on some platforms):", e)

    conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS temp_vec USING vec0(x float[4])")
    print("✓ Created vec0 virtual table")

    conn.close()
    print("Environment is 100% ready – uv style!")


if __name__ == "__main__":
    main()
