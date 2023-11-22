drop schema if exists launch_test_elpc cascade;
CREATE SCHEMA launch_test_elpc;

CREATE TABLE launch_test_elpc.detections (
    detection_id SERIAL,
    run_id INT,
    run_nickname VARCHAR,
    image_date TIMESTAMP, -- timestamp when the image was actually captured
    detection_timestamp TIMESTAMP, -- timestamp when model was used to make inference
    image_location NUMERIC, -- that is, the location identifier to use across runs
    image_path VARCHAR, -- path to the image on disk (though will only be relevant for current run?)
    lat_center FLOAT, -- center of the detected application in lat/lon
    lon_center FLOAT,
    lat_max FLOAT,
    lat_min FLOAT,
    lon_max FLOAT,
    lon_min FLOAT,
    est_size_acres FLOAT,
    -- bbox cols in image coords (e.g., 0-1)
    bbox_center_x FLOAT,
    bbox_center_y FLOAT,
    bbox_width FLOAT,
    bbox_height FLOAT,
    county_name VARCHAR,
    city_town_name VARCHAR,
    farm_name VARCHAR,
    field_name VARCHAR,
    score FLOAT
);
ALTER TABLE launch_test_elpc.detections ADD PRIMARY KEY (detection_id);
CREATE INDEX ON launch_test_elpc.detections(detection_timestamp);

CREATE TABLE launch_test_elpc.pipeline_runs (
    run_id SERIAL,
    run_nickname VARCHAR,
    run_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    config JSONB,
    config_hash VARCHAR,
    imagery_date DATE,
    imagery_days_range INT,
    imagery_months_range INT,
    run_status VARCHAR
);
ALTER TABLE launch_test_elpc.pipeline_runs ADD PRIMARY KEY (run_id);

CREATE TABLE launch_test_elpc.sent_to_elpc (
    run_id INT,
    detection_id INT,
    detection_timestamp TIMESTAMP,
    lat_center FLOAT,
    lon_center FLOAT,
    est_size_acres FLOAT,
    county_name VARCHAR,
    city_town_name VARCHAR,
    score FLOAT,
    gdrive_image_url VARCHAR,
    -- Distances to three closest verifiers (not necessarily assignments)
    verifier1_name VARCHAR, 
    verifier1_addr VARCHAR,
    verifier1_distance	FLOAT,
    verifier2_name VARCHAR,
    verifier2_addr VARCHAR,
    verifier2_distance FLOAT,
    verifier3_name VARCHAR,
    verifier3_addr VARCHAR,
    verifier3_distance FLOAT
);
CREATE INDEX ON launch_test_elpc.sent_to_elpc(run_id);
CREATE INDEX ON launch_test_elpc.sent_to_elpc(detection_id);

CREATE TABLE launch_test_elpc.raw_elpc_results(
    response_timestamp TIMESTAMP,
    email VARCHAR,
    location_id VARCHAR,
    verify_reason VARCHAR,
    visible VARCHAR,
    visible_desc VARCHAR,
    manure_visible VARCHAR,
    verification_confidence VARCHAR,
    confidence_desc VARCHAR,
    visit_date DATE,
    visit_time TIME,
    farm_equipment VARCHAR,
    equipment_desc VARCHAR,
    odor_present VARCHAR,
    field_status VARCHAR,
    weather VARCHAR,
    notes VARCHAR,
    photo_links VARCHAR,
    hours_worked VARCHAR
);

CREATE TABLE launch_test_elpc.elpc_results(
    response_timestamp TIMESTAMP,
    email VARCHAR,
    location_id VARCHAR,
    verify_reason VARCHAR,
    visible VARCHAR,
    visible_desc VARCHAR,
    manure_visible VARCHAR,
    verification_confidence VARCHAR,
    confidence_desc VARCHAR,
    visit_date DATE,
    visit_time TIME,
    farm_equipment VARCHAR,
    equipment_desc VARCHAR,
    odor_present VARCHAR,
    field_status VARCHAR,
    weather VARCHAR,
    notes VARCHAR,
    photo_links VARCHAR,
    hours_worked VARCHAR,
    detection_id INT
);

CREATE INDEX ON launch_test_elpc.elpc_results(detection_id);
