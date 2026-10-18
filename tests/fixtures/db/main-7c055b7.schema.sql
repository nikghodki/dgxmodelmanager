CREATE TABLE users (
	id INTEGER NOT NULL, 
	username VARCHAR(64) NOT NULL, 
	pw_hash VARCHAR(255) NOT NULL, 
	role VARCHAR(16) NOT NULL, 
	must_change_pw BOOLEAN NOT NULL, 
	created_at DATETIME NOT NULL, 
	PRIMARY KEY (id), 
	UNIQUE (username)
);
CREATE TABLE deployments (
	id INTEGER NOT NULL, 
	name VARCHAR(64) NOT NULL, 
	engine VARCHAR(16) NOT NULL, 
	model_ref VARCHAR(512) NOT NULL, 
	config JSON NOT NULL, 
	state VARCHAR(16) NOT NULL, 
	container_id VARCHAR(80), 
	port INTEGER, 
	priority INTEGER NOT NULL, 
	alias VARCHAR(64), 
	reserved_bytes BIGINT NOT NULL, 
	estimate JSON NOT NULL, 
	litellm_status VARCHAR(16) NOT NULL, 
	last_error TEXT, 
	exit_code INTEGER, 
	oom_killed BOOLEAN, 
	adopted_from VARCHAR(128), 
	system BOOLEAN, 
	created_by INTEGER, 
	created_at DATETIME NOT NULL, 
	updated_at DATETIME NOT NULL, 
	PRIMARY KEY (id), 
	UNIQUE (name)
);
CREATE TABLE memory_calibration (
	id INTEGER NOT NULL, 
	engine VARCHAR(16) NOT NULL, 
	repo VARCHAR(512) NOT NULL, 
	config_hash VARCHAR(32) NOT NULL, 
	estimated_bytes BIGINT NOT NULL, 
	measured_bytes BIGINT NOT NULL, 
	oom BOOLEAN NOT NULL, 
	ts DATETIME NOT NULL, 
	method VARCHAR(32), 
	quality VARCHAR(16), 
	detail JSON, 
	PRIMARY KEY (id)
);
CREATE INDEX ix_memory_calibration_repo ON memory_calibration (repo);
CREATE TABLE benchmarks (
	id INTEGER NOT NULL, 
	deployment_id INTEGER, 
	engine VARCHAR(16) NOT NULL, 
	model_ref VARCHAR(512) NOT NULL, 
	config_hash VARCHAR(32) NOT NULL, 
	config JSON NOT NULL, 
	concurrency INTEGER NOT NULL, 
	input_len INTEGER NOT NULL, 
	output_len INTEGER NOT NULL, 
	requests INTEGER NOT NULL, 
	errors INTEGER, 
	ttft_p50 FLOAT, 
	ttft_p95 FLOAT, 
	itl_p50 FLOAT, 
	tps_total FLOAT NOT NULL, 
	tps_per_req FLOAT NOT NULL, 
	ts DATETIME NOT NULL, 
	PRIMARY KEY (id)
);
CREATE INDEX ix_benchmarks_model_ref ON benchmarks (model_ref);
CREATE INDEX ix_benchmarks_deployment_id ON benchmarks (deployment_id);
CREATE TABLE benchmark_runs (
	id INTEGER NOT NULL, 
	deployment_id INTEGER, 
	engine VARCHAR(16) NOT NULL, 
	model_ref VARCHAR(512) NOT NULL, 
	config_hash VARCHAR(32) NOT NULL, 
	levels JSON NOT NULL, 
	status VARCHAR(16) NOT NULL, 
	failed_concurrency INTEGER, 
	error TEXT, 
	started_at DATETIME NOT NULL, 
	finished_at DATETIME, 
	PRIMARY KEY (id)
);
CREATE INDEX ix_benchmark_runs_deployment_id ON benchmark_runs (deployment_id);
CREATE INDEX ix_benchmark_runs_model_ref ON benchmark_runs (model_ref);
CREATE TABLE metric_rollups (
	id INTEGER NOT NULL, 
	deployment_id INTEGER, 
	minute DATETIME NOT NULL, 
	data JSON NOT NULL, 
	PRIMARY KEY (id)
);
CREATE INDEX ix_metric_rollups_minute ON metric_rollups (minute);
CREATE INDEX ix_metric_rollups_deployment_id ON metric_rollups (deployment_id);
CREATE TABLE downloads (
	id INTEGER NOT NULL, 
	repo_id VARCHAR(256) NOT NULL, 
	revision VARCHAR(64), 
	files JSON NOT NULL, 
	bytes_done BIGINT NOT NULL, 
	bytes_total BIGINT NOT NULL, 
	state VARCHAR(16) NOT NULL, 
	error TEXT, 
	created_by INTEGER, 
	created_at DATETIME NOT NULL, 
	PRIMARY KEY (id)
);
CREATE TABLE favourites (
	ref VARCHAR(512) NOT NULL, 
	PRIMARY KEY (ref)
);
CREATE TABLE audit_log (
	id INTEGER NOT NULL, 
	user_id INTEGER, 
	action VARCHAR(64) NOT NULL, 
	target VARCHAR(256) NOT NULL, 
	detail JSON NOT NULL, 
	ts DATETIME NOT NULL, 
	PRIMARY KEY (id)
);
CREATE INDEX ix_audit_log_ts ON audit_log (ts);
CREATE TABLE decisions (
	id INTEGER NOT NULL, 
	created_at DATETIME NOT NULL, 
	feature VARCHAR(32) NOT NULL, 
	subject VARCHAR(256) NOT NULL, 
	source VARCHAR(8) NOT NULL, 
	choice VARCHAR(256), 
	confidence FLOAT, 
	probabilities JSON NOT NULL, 
	latency_ms FLOAT, 
	model VARCHAR(128), 
	revision VARCHAR(64), 
	input_hash VARCHAR(64), 
	error VARCHAR(64), 
	feedback VARCHAR(8), 
	feedback_by INTEGER, 
	feedback_at DATETIME, 
	PRIMARY KEY (id)
);
CREATE INDEX ix_decisions_created_at ON decisions (created_at);
CREATE INDEX ix_decisions_feature ON decisions (feature);
CREATE TABLE app_settings (
	"key" VARCHAR(128) NOT NULL, 
	value TEXT NOT NULL, 
	updated_at DATETIME NOT NULL, 
	PRIMARY KEY ("key")
);
CREATE TABLE artifact_kinds (
	ref VARCHAR(1024) NOT NULL, 
	mtime FLOAT NOT NULL, 
	kind VARCHAR(16), 
	source VARCHAR(8) NOT NULL, 
	confidence FLOAT, 
	decision_id INTEGER, 
	updated_at DATETIME NOT NULL, 
	PRIMARY KEY (ref)
);
CREATE TABLE sessions (
	id VARCHAR(64) NOT NULL, 
	user_id INTEGER NOT NULL, 
	csrf_token VARCHAR(64) NOT NULL, 
	ip VARCHAR(64) NOT NULL, 
	expires_at DATETIME NOT NULL, 
	PRIMARY KEY (id), 
	FOREIGN KEY(user_id) REFERENCES users (id) ON DELETE CASCADE
);
CREATE TABLE cli_tokens (
	id VARCHAR(64) NOT NULL, 
	user_id INTEGER NOT NULL, 
	scope VARCHAR(32) NOT NULL, 
	created_at DATETIME NOT NULL, 
	expires_at DATETIME NOT NULL, 
	PRIMARY KEY (id), 
	FOREIGN KEY(user_id) REFERENCES users (id) ON DELETE CASCADE
);
CREATE TABLE deployment_events (
	id INTEGER NOT NULL, 
	deployment_id INTEGER NOT NULL, 
	ts DATETIME NOT NULL, 
	from_state VARCHAR(16) NOT NULL, 
	to_state VARCHAR(16) NOT NULL, 
	message TEXT NOT NULL, 
	log_tail TEXT, 
	PRIMARY KEY (id), 
	FOREIGN KEY(deployment_id) REFERENCES deployments (id) ON DELETE CASCADE
);
CREATE INDEX ix_deployment_events_deployment_id ON deployment_events (deployment_id);
CREATE TABLE diagnoses (
	id INTEGER NOT NULL, 
	deployment_id INTEGER NOT NULL, 
	event_id INTEGER, 
	created_at DATETIME NOT NULL, 
	cause VARCHAR(64), 
	source VARCHAR(8) NOT NULL, 
	confidence FLOAT, 
	decision_id INTEGER, 
	summary TEXT NOT NULL, 
	fix_hint TEXT NOT NULL, 
	patch JSON, 
	log_excerpt TEXT NOT NULL, 
	PRIMARY KEY (id), 
	CONSTRAINT uq_diagnoses_event UNIQUE (deployment_id, event_id), 
	FOREIGN KEY(deployment_id) REFERENCES deployments (id) ON DELETE CASCADE
);
CREATE INDEX ix_diagnoses_deployment_id ON diagnoses (deployment_id);
CREATE TABLE schema_version (version INTEGER PRIMARY KEY, name TEXT NOT NULL, applied_at TEXT NOT NULL);
