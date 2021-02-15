-- DROP TABLE IF EXISTS exercises;

CREATE TABLE IF NOT EXISTS exercises (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    ex_name TEXT NOT NULL,
    muscle_group TEXT NOT NULL,
    ex_mech_type TEXT NOT NULL,
    ex_type TEXT NOT NULL,
    ex_equipment TEXT NOT NULL,
    level TEXT NOT NULL,
    reps INTEGER NOT NULL,
    sets INTEGER NOT NULL,
    main-muscle-worked TEXT NOT NULL,
    movement_size TEXT NOT NULL,
    load_size TEXT NOT NULL,
    instructions TEXT
);