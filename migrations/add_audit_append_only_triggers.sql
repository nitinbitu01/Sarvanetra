-- migrations/add_audit_append_only_triggers.sql
-- Run as part of PostgreSQL database initialization for Day 16.2.
-- Enforces append-only immutable audit logging at the database engine level.

CREATE OR REPLACE FUNCTION prevent_audit_modification()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'UPDATE' THEN
        RAISE EXCEPTION
            'audit_log is append-only. Updates are forbidden. Add a compensating record instead. Row ID: %, Event: %',
            OLD.id, OLD.event_type;
    END IF;
    IF TG_OP = 'DELETE' THEN
        RAISE EXCEPTION
            'audit_log is append-only. Deletion is a compliance violation. Contact your compliance officer. Row ID: %, Event: %',
            OLD.id, OLD.event_type;
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS audit_log_immutable ON audit_log;

CREATE TRIGGER audit_log_immutable
BEFORE UPDATE OR DELETE ON audit_log
FOR EACH ROW
EXECUTE FUNCTION prevent_audit_modification();
