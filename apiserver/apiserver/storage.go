package apiserver

import (
	"database/sql"
	"fmt"
	"github.com/lib/pq"
)

type Person struct {
	Id          string
	Directory   string
	Filename    string
	FilenameUid string
	Score       float64
	Box         []uint32
	Vector      []float64
}

func (u Person) String() string {
	return fmt.Sprintf("Person<%s %s %f %s %s %v %v>", u.Id, u.Directory, u.Score, u.Filename, u.FilenameUid, u.Box, u.Vector)
}

type PgStorage struct {
	db *sql.DB
}

func NewStorage(user string, password string, database string, host string) (PgStorage, error) {
	connStr := fmt.Sprintf("user=%s dbname=%s password=%s host=%s", user, database, password, host)
	db, err := sql.Open("postgres", connStr)
	if err != nil {
		return PgStorage{}, err
	}

	pgo := PgStorage{
		db: db,
	}

	return pgo, nil
}

func (pgo *PgStorage) CloseStorage() {
	pgo.db.Close()
}

func (pgo *PgStorage) Store(person Person) error {
	_, err := pgo.db.Exec("insert into persons (id, directory, filename, filenameuid, score, box, vector) values ($1, $2, $3, $4, $5, $6, $7)",
		person.Id, person.Directory, person.Filename, person.FilenameUid, person.Score, pq.Array(person.Box), pq.Array(person.Vector))

	return err
}

func (pgo *PgStorage) GetDirectory(directory string) ([]Person, error) {
	var persons []Person
	rows, err := pgo.db.Query("select id, filename, filenameuid, score, box, vector from persons where directory=$1", directory)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	for rows.Next() {
		var id string
		var filename string
		var filenameuid string
		var score float64
		var box []int64
		var vector []float64
		err = rows.Scan(&id, &filename, &filenameuid, &score, pq.Array(&box), pq.Array(&vector))
		if err != nil {
			return nil, err
		}

		rebox := make([]uint32, len(box))
		for i, v := range box {
			rebox[i] = uint32(v)
		}

		persons = append(persons, Person{
			Id:          id,
			Directory:   directory,
			Filename:    filename,
			FilenameUid: filenameuid,
			Score:       score,
			Box:         rebox,
			Vector:      vector,
		})
	}

	err = rows.Err()
	if err != nil {
		return nil, err
	}

	return persons, nil
}
