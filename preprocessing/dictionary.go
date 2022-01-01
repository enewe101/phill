package main

import (
	"os"
	"strings"
	"strconv"
	"fmt"
	"sort"
)

type Dictionary struct {
	tokens2ids map[string]int
	ids2tokens []string
	counts []int64
	numAuxTokens  int
}


func NewDictionary(auxTokens []string) *Dictionary {
	dictionary := Dictionary{}
	dictionary.tokens2ids = make(map[string]int)
	dictionary.Add(auxTokens...)
	dictionary.numAuxTokens = len(auxTokens)
	return &dictionary
}

func(d *Dictionary) Sort() {

	// We'll zip the counts with corresponding tokens together, then sort them
	// by count.
	type sortable struct {
		count int64
		token string
	}

	// Make a sorter list containing the counts and tokens zipped together.
	// We want to leave the first few aux tokens in place, so we omit them from
	// the sorter.
	sorter := make([]sortable, len(d.counts))
	for i:=0; i<len(d.counts); i++ {
		if i < d.numAuxTokens {continue}
		sorter[i] = sortable{count:d.counts[i], token:d.ids2tokens[i]}
	}

	// Sort it.
	sort.Slice(sorter, func(i,j int) bool {
		return sorter[i].count > sorter[j].count
	})

	// Now remake the dictionary with sorted values.
	for sorterIndex:=0; sorterIndex<len(d.counts)-d.numAuxTokens; sorterIndex++ {
		dictIndex := sorterIndex + d.numAuxTokens
		d.counts[dictIndex] = sorter[sorterIndex].count
		d.ids2tokens[dictIndex] = sorter[sorterIndex].token
		d.tokens2ids[sorter[sorterIndex].token] = dictIndex
	}
}

func(d *Dictionary) Write(path string) {
	file, err := os.Create(path)
	if err != nil {panic(err.Error())}
	for id, token := range d.ids2tokens {
		fmt.Fprintf(file, "%s\t%d\n", token, d.counts[id])
	}
	file.Close()
}

func ReadDictionary(path string, numAuxTokens int) (*Dictionary, error) {
	data, err := os.ReadFile(path)
	if err != nil {return nil, err}
	lines := strings.Split(string(data), "\n")
	ids2tokens := make([]string, len(lines))
	tokens2ids := make(map[string]int, len(lines))
	counts := make([]int64, len(lines))
	for i, line := range lines {
		if strings.TrimSpace(line) == "" {continue}
		split := strings.Split(line, "\t")
		token := split[0]
		count, err := strconv.ParseInt(split[1], 10, 64)
		if err != nil {panic(err.Error())}
		ids2tokens[i] = token
		tokens2ids[token] = i
		counts[i] = int64(count)
	}
	auxTokens := ids2tokens[:numAuxTokens]
	dictionary := NewDictionary(auxTokens)
	dictionary.ids2tokens = ids2tokens
	dictionary.tokens2ids = tokens2ids
	dictionary.counts = counts
	return dictionary, nil
}


func(d *Dictionary) Add(tokens ...string) {

	for _, token := range tokens {
		// Check if it's already there
		id, ok := d.tokens2ids[token]
		if ok {
			d.counts[id] += 1
			continue
		}

		// If not, add it.
		d.ids2tokens = append(d.ids2tokens, token)
		id = len(d.ids2tokens)-1
		d.counts = append(d.counts, 1)
		d.tokens2ids[token] = id
	}
}

func(d *Dictionary) GetId(token string) int {
	id, valid := d.tokens2ids[token]
	if !valid {
		panic("\"" + token + "\" not found in dictionary.")
	}
	return id
}

func(d *Dictionary) GetToken(id int) string {
	return d.ids2tokens[id]
}

