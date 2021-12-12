package main

import (
	"os"
	"strings"
)

type Dictionary struct {
	tokens2ids map[string]int
	ids2tokens []string
	counts []int32
}

func NewDictionary() Dictionary {
	dictionary := Dictionary{}
	dictionary.tokens2ids = make(map[string]int)
	dictionary.ids2tokens = make([]string,0)
	return dictionary
}

func(d *Dictionary) Write(file *os.File) {
	file.WriteString(strings.Join(d.ids2tokens, "\n"))
	file.Sync()
}

func ReadDictionary(path string) Dictionary {
	data, err := os.ReadFile(path)
	if err != nil {
		panic(err.Error())
	}
	ids2tokens := strings.Split(string(data), "\n")
	tokens2ids := make(map[string]int)
	for i, token := range ids2tokens {
		tokens2ids[token] = i
	}
	dictionary := NewDictionary()
	dictionary.ids2tokens = ids2tokens
	dictionary.tokens2ids = tokens2ids
	return dictionary
}

func(d *Dictionary) Add(token string) (id int, wasAdded bool) {

	// Check if it's already there
	id, ok := d.tokens2ids[token]
	if ok {
		return id, false
	}

	// If not, add it.
	d.ids2tokens = append(d.ids2tokens, token)
	id = len(d.ids2tokens)-1
	d.counts = append(d.counts, 1)
	d.tokens2ids[token] = id

	return id, true
}

func(d *Dictionary) GetId(token string) int {
	return d.tokens2ids[token]
}

func(d *Dictionary) GetToken(id int) string {
	return d.ids2tokens[id]
}

