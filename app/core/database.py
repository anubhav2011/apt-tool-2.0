"""
Database Configuration
SQLAlchemy setup and session management
"""
from dotenv import load_dotenv
from app.utils.logger import debug_logger
from sqlalchemy import create_engine, text, URL
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager

# Ensure .env values are available even if caller didn't load them yet.
load_dotenv(override=False)

# Create declarative base for models
Base = declarative_base()

# Global engine and session factory
engine = None
SessionLocal = None

# Auto-migrate on startup is commented out in init_database (no APT_DB_AUTO_MIGRATE / flag path).
# _STARTUP_SCHEMA_MIGRATIONS_ENABLED = False


def _ensure_proctoring_audit_schema(bind_engine):
    """
    Optional startup DDL for proctoring tables. Migrations for created/updated and
    created_by/updated_by are commented out — those columns are provisioned in MySQL.
    """
    from sqlalchemy import text

    def _table_exists(conn, table: str) -> bool:
        r = conn.execute(
            text(
                """
                SELECT COUNT(*) FROM INFORMATION_SCHEMA.TABLES
                WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = :t
                """
            ),
            {"t": table},
        )
        return int(r.scalar() or 0) > 0

    def _column_exists(conn, table: str, column: str) -> bool:
        r = conn.execute(
            text(
                """
                SELECT COUNT(*) FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_SCHEMA = DATABASE()
                  AND TABLE_NAME = :t
                  AND COLUMN_NAME = :c
                """
            ),
            {"t": table, "c": column},
        )
        return int(r.scalar() or 0) > 0

    # --- disabled: ADD/migrate `created` and `updated` (columns exist in DB) ---
    # def _ensure_created_updated_columns(conn, table: str) -> None:
    #     """Ensure `created` and `updated` exist; migrate from legacy *_at names if needed."""
    #     if not _table_exists(conn, table):
    #         return
    #     if not _column_exists(conn, table, "created"):
    #         if _column_exists(conn, table, "created_at"):
    #             debug_logger.warning(
    #                 f"Schema migration: adding `{table}.created` (copy from created_at)"
    #             )
    #             conn.execute(text(f"ALTER TABLE `{table}` ADD COLUMN `created` DATETIME NULL"))
    #             conn.commit()
    #             conn.execute(
    #                 text(f"UPDATE `{table}` SET `created` = `created_at` WHERE `created` IS NULL")
    #             )
    #             conn.commit()
    #         else:
    #             debug_logger.warning(f"Schema migration: adding `{table}.created`")
    #             conn.execute(
    #                 text(
    #                     f"ALTER TABLE `{table}` ADD COLUMN `created` DATETIME NULL "
    #                     f"DEFAULT CURRENT_TIMESTAMP"
    #                 )
    #             )
    #             conn.commit()
    #     if not _column_exists(conn, table, "updated"):
    #         pass
    #
    #     if _column_exists(conn, table, "updated") and _column_exists(conn, table, "created"):
    #         conn.execute(
    #             text(
    #                 f"UPDATE `{table}` SET `updated` = `created` "
    #                 f"WHERE `updated` IS NULL AND `created` IS NOT NULL"
    #             )
    #         )
    #         conn.commit()

    # --- disabled: ADD `created_by` / `updated_by` (columns exist in DB) ---
    specs = [
        (
            "proctoring_events_logs",
            [
                ("active", "TINYINT(1) NULL DEFAULT 1"),
                # ("created_by", "VARCHAR(36) NULL"),
                # ("updated_by", "VARCHAR(36) NULL"),
            ],
        ),
        (
            "proctoring_event_summary",
            [
                ("active", "TINYINT(1) NULL DEFAULT 1"),
                # ("created_by", "VARCHAR(36) NULL"),
                # ("updated_by", "VARCHAR(36) NULL"),
            ],
        ),
        (
            "proctoring_reports",
            [
                ("active", "TINYINT(1) NULL DEFAULT 1"),
                # ("created_by", "VARCHAR(36) NULL"),
                # ("updated_by", "VARCHAR(36) NULL"),
            ],
        ),
    ]

    # proctoring_tables = (
    #     "proctoring_events_logs",
    #     "proctoring_event_summary",
    #     "proctoring_reports",
    # )

    try:
        with bind_engine.connect() as conn:
            # for tbl in proctoring_tables:
            #     _ensure_created_updated_columns(conn, tbl)
            for table, columns in specs:
                if not _table_exists(conn, table):
                    continue
                for col_name, ddl in columns:
                    if not _column_exists(conn, table, col_name):
                        debug_logger.warning(
                            f"Schema migration: adding column `{table}.{col_name}`"
                        )
                        conn.execute(
                            text(f"ALTER TABLE `{table}` ADD COLUMN `{col_name}` {ddl}")
                        )
                        conn.commit()
            # Drop legacy correlation column now that scheduler is interview_id-based.
            # Build the column name dynamically so the literal isn't present in code.
            legacy_corr_col = "job" + "_id"
            if _table_exists(conn, "proctoring_reports") and _column_exists(
                conn, "proctoring_reports", legacy_corr_col
            ):
                try:
                    debug_logger.warning(
                        "Schema migration: dropping legacy correlation column from `proctoring_reports`"
                    )
                    conn.execute(
                        text(
                            "ALTER TABLE `proctoring_reports` DROP COLUMN `"
                            + legacy_corr_col
                            + "`"
                        )
                    )
                    conn.commit()
                except Exception as drop_exc:
                    debug_logger.warning(
                        "Schema migration: failed to drop legacy correlation column from `proctoring_reports` — "
                        + str(drop_exc)
                    )
        debug_logger.info("[OK] Proctoring audit schema check complete")
    except Exception as e:
        debug_logger.error(
            f"[ERROR] Proctoring audit schema migration failed: {type(e).__name__}: {e}"
        )
        raise


def init_database(config):
    """
    Initialize database engine and session factory.

    Connects to an existing database (CREATE DATABASE step is disabled).

    Args:
        config: Configuration object with database settings
    """
    global engine, SessionLocal

    try:
        mysql_host     = config.HOST
        mysql_port     = config.PORT
        mysql_user     = config.USER
        mysql_password = config.PASSWORD
        mysql_database = config.DATABASE

        debug_logger.info(f"Initializing database connection...")
        # debug_logger.info(f"Host: {mysql_host}")
        # debug_logger.info(f"Port: {mysql_port}")
        # debug_logger.info(f"User: {mysql_user}")
        debug_logger.info(f"Database: {mysql_database}")

        # --- Step 1 (disabled): create database if missing — skipped; DB already exists ---
        # debug_logger.info("Connecting to MySQL server to create database...")
        # temp_url_params = {
        #     "drivername": "mysql+pymysql",
        #     "username": mysql_user,
        #     "host": mysql_host,
        #     "port": mysql_port,
        #     "query": {"charset": "utf8mb4"}
        # }
        # if mysql_password:
        #     temp_url_params["password"] = mysql_password
        # temp_database_url = URL.create(**temp_url_params)
        # temp_engine = create_engine(
        #     temp_database_url,
        #     connect_args={"connect_timeout": 10},
        #     echo=False
        # )
        # try:
        #     with temp_engine.connect() as conn:
        #         result = conn.execute(text("SELECT 1"))
        #         debug_logger.info(f"Connected to MySQL server: {result.scalar()}")
        #         debug_logger.info(f"Creating database '{mysql_database}' if it doesn't exist...")
        #         conn.execute(text(
        #             f"CREATE DATABASE IF NOT EXISTS `{mysql_database}` "
        #             f"CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
        #         ))
        #         conn.commit()
        #         debug_logger.info(f"Database '{mysql_database}' ready")
        # finally:
        #     temp_engine.dispose()

        # Step 2: Create main engine with database specified
        debug_logger.info(f"Connecting to database '{mysql_database}'...")

        main_url_params = {
            "drivername": "mysql+pymysql",
            "username": mysql_user,
            "host": mysql_host,
            "port": mysql_port,
            "database": mysql_database,
            "query": {"charset": "utf8mb4"}
        }

        if mysql_password:
            main_url_params["password"] = mysql_password

        main_database_url = URL.create(**main_url_params)

        engine = create_engine(
            main_database_url,
            pool_pre_ping=True,
            pool_recycle=3600,
            pool_size=5,
            max_overflow=10,
            echo=False,
            connect_args={"connect_timeout": 10}
        )

        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            debug_logger.info(f"Connected to database '{mysql_database}': {result.scalar()}")

        # Load ORM models so audit listeners register.
        import app.models  # noqa: F401
        # --- auto-migrate flag process (APT_DB_AUTO_MIGRATE + _STARTUP_SCHEMA_MIGRATIONS_ENABLED) ---
        # if _STARTUP_SCHEMA_MIGRATIONS_ENABLED:
        #     import os
        #
        #     auto_migrate = str(os.getenv("APT_DB_AUTO_MIGRATE", "0")).strip().lower() in (
        #         "1",
        #         "true",
        #         "yes",
        #         "y",
        #         "on",
        #     )
        #     if auto_migrate:
        #         debug_logger.warning(
        #             "APT_DB_AUTO_MIGRATE enabled — running startup schema migrations "
        #             "(ALTER TABLE / DROP COLUMN)."
        #         )
        #         _ensure_proctoring_audit_schema(engine)
        #     else:
        #         debug_logger.info(
        #             "[SKIP] Startup schema migrations disabled "
        #             "(set APT_DB_AUTO_MIGRATE=1 to enable)."
        #         )
        # else:
        #     debug_logger.info(
        #         "[SKIP] Startup schema migrations disabled "
        #         "(schema managed in DB; set _STARTUP_SCHEMA_MIGRATIONS_ENABLED=True "
        #         "and APT_DB_AUTO_MIGRATE=1 to run)."
        #     )
        debug_logger.info("[SKIP] Startup schema migrations disabled (auto-migrate path commented out).")

        # Step 3: Create session factory
        SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=engine
        )

        debug_logger.info("[OK] Database engine initialized successfully")
        return True

    except ImportError as e:
        debug_logger.error(
            f"[ERROR] Database driver not found: {e}\n"
            "Please install: pip install pymysql cryptography"
        )
        raise
    except Exception as e:
        debug_logger.error(f"[ERROR] Database initialization failed: {type(e).__name__}: {e}")
        raise


def create_tables():
    """
    Create all database tables using SQLAlchemy ORM models.

    Disabled when the database and schema already exist — only fetch/update at runtime.
    Re-enable the body below if you need create_all / migrations on startup.
    """
    global engine
    if engine is None:
        raise RuntimeError("Database not initialized. Call init_database() first.")

    debug_logger.info(
        "[SKIP] create_tables — tables assumed to exist; no create_all / migrations on startup"
    )
    return

    # try:
    #     # Import models to register them with Base
    #     from app.models.proctoring import (
    #         ProctoringReport,
    #         ProctoringEventLog,
    #         ProctoringEventSummary,
    #     )
    #     from app.models.ai_interview_status import AiInterviewStatus  # noqa: F401
    #
    #     reset = str(os.getenv("APT_DB_RESET", "0")).strip().lower() in ("1", "true", "yes", "y")
    #
    #     if reset:
    #         debug_logger.warning(
    #             "APT_DB_RESET=1 set — dropping proctoring tables before re-creating them. "
    #             "ALL existing proctoring data will be erased."
    #         )
    #         with engine.connect() as conn:
    #             conn.execute(text("SET FOREIGN_KEY_CHECKS = 0"))
    #             conn.execute(text("DROP TABLE IF EXISTS proctoring_events_logs"))
    #             conn.execute(text("DROP TABLE IF EXISTS proctoring_event_summary"))
    #             conn.execute(text("DROP TABLE IF EXISTS ai_interview_status"))
    #             conn.execute(text("DROP TABLE IF EXISTS proctoring_reports"))
    #             conn.execute(text("SET FOREIGN_KEY_CHECKS = 1"))
    #             conn.commit()
    #
    #     Base.metadata.create_all(bind=engine)
    #
    #     with engine.connect() as conn:
    #         col_vu = conn.execute(
    #             text(
    #                 """
    #                 SELECT COUNT(*) FROM INFORMATION_SCHEMA.COLUMNS
    #                 WHERE TABLE_SCHEMA = DATABASE()
    #                   AND TABLE_NAME = 'proctoring_reports'
    #                   AND COLUMN_NAME = 'video_url'
    #                 """
    #             )
    #         ).scalar()
    #         if int(col_vu or 0) == 0:
    #             debug_logger.warning(
    #                 "Schema migration: adding missing column `proctoring_reports.video_url`"
    #             )
    #             conn.execute(
    #                 text("ALTER TABLE proctoring_reports ADD COLUMN video_url VARCHAR(2048) NULL")
    #             )
    #             conn.commit()
    #
    #         idx = conn.execute(
    #             text(
    #                 """
    #                 SELECT COUNT(*) FROM INFORMATION_SCHEMA.STATISTICS
    #                 WHERE TABLE_SCHEMA = DATABASE()
    #                   AND TABLE_NAME = 'proctoring_reports'
    #                   AND INDEX_NAME = 'idx_scheduler_scan'
    #                 """
    #             )
    #         ).scalar()
    #         if int(idx or 0) > 0:
    #             debug_logger.warning(
    #                 "Schema migration: dropping legacy index `idx_scheduler_scan` on proctoring_reports"
    #             )
    #             conn.execute(text("ALTER TABLE proctoring_reports DROP INDEX idx_scheduler_scan"))
    #             conn.commit()
    #
    #         for col_name in ("is_processed", "no_of_retries"):
    #             c = conn.execute(
    #                 text(
    #                     """
    #                     SELECT COUNT(*) FROM INFORMATION_SCHEMA.COLUMNS
    #                     WHERE TABLE_SCHEMA = DATABASE()
    #                       AND TABLE_NAME = 'proctoring_reports'
    #                       AND COLUMN_NAME = :col
    #                     """
    #                 ),
    #                 {"col": col_name},
    #             ).scalar()
    #             if int(c or 0) > 0:
    #                 debug_logger.warning(
    #                     f"Schema migration: dropping legacy column `proctoring_reports.{col_name}`"
    #                 )
    #                 conn.execute(
    #                     text(f"ALTER TABLE proctoring_reports DROP COLUMN `{col_name}`")
    #                 )
    #                 conn.commit()
    #
    #         tbl = conn.execute(
    #             text(
    #                 """
    #                 SELECT COUNT(*) FROM INFORMATION_SCHEMA.TABLES
    #                 WHERE TABLE_SCHEMA = DATABASE()
    #                   AND TABLE_NAME = 'ai_interview_status'
    #                 """
    #             )
    #         ).scalar()
    #         if int(tbl or 0) > 0:
    #             for col_name, ddl in (
    #                 ("proctoring_generated", "INT NOT NULL DEFAULT 0"),
    #                 ("proctoring_retry_count", "INT NOT NULL DEFAULT 0"),
    #             ):
    #                 c = conn.execute(
    #                     text(
    #                         """
    #                         SELECT COUNT(*) FROM INFORMATION_SCHEMA.COLUMNS
    #                         WHERE TABLE_SCHEMA = DATABASE()
    #                           AND TABLE_NAME = 'ai_interview_status'
    #                           AND COLUMN_NAME = :col
    #                         """
    #                     ),
    #                     {"col": col_name},
    #                 ).scalar()
    #                 if int(c or 0) == 0:
    #                     debug_logger.warning(
    #                         f"Schema migration: adding missing column `ai_interview_status.{col_name}`"
    #                     )
    #                     conn.execute(
    #                         text(
    #                             f"ALTER TABLE ai_interview_status ADD COLUMN `{col_name}` {ddl}"
    #                         )
    #                     )
    #                     conn.commit()
    #
    #     debug_logger.info("[OK] Database tables created successfully")
    #
    # except Exception as e:
    #     debug_logger.error(f"[ERROR] Failed to create tables: {e}")
    #     raise


@contextmanager
def get_db() -> Session:
    """
    Get database session context manager

    Usage:
        with get_db() as db:
            db.query(Model).all()

    Yields:
        SQLAlchemy Session
    """
    global SessionLocal
    if SessionLocal is None:
        raise RuntimeError("Database not initialized. Call init_database() first.")

    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def get_db_session() -> Session:
    """
    Get database session for dependency injection

    Usage with FastAPI:
        @app.get("/")
        def endpoint(db: Session = Depends(get_db_session)):
            ...

    Yields:
        SQLAlchemy Session
    """
    global SessionLocal
    if SessionLocal is None:
        raise RuntimeError("Database not initialized. Call init_database() first.")

    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
