CREATE TABLE persons (
    id character varying(255) NOT NULL,
    directory character varying(255) NOT NULL,
    vector double precision[] NOT NULL,
    filename character varying(255) NOT NULL,
    filenameuid character varying(255) NOT NULL,
    box integer[] NOT NULL,
    score double precision NOT NULL
);

CREATE INDEX persons_directory ON persons USING btree (directory);
