package main

import (
	"strings"
	"strconv"
)

const ConlluDelimiter = "\n\n"

func conlluDictParser(data *[]byte) (*[]string, *[]string) {
		sentences := strings.Split(string(*data), ConlluDelimiter)
		tokens := make([]string, 0)
		relations := make([]string, 0)
		for _, sentence := range sentences {
			if len(strings.TrimSpace(sentence)) == 0 {continue}
			lines := strings.Split(sentence, "\n")
			for _, line := range lines {
				if len(line) == 0 || line[:1] == "#" {continue}
				fields := strings.Split(line, "\t")
				if strings.Index(fields[0], "-") >= 0 ||
					strings.Index(fields[0], ".") >= 0 {
					continue
				}
				token := fields[1]
				tokens = append(tokens, token)
				relation := fields[7]
				relations = append(relations, relation)
			}
		}
		tokens = *releaseMemory(&tokens)
		relations = *releaseMemory(&relations)
		return &tokens, &relations
}


func conlluSentenceParser(
	data *[]byte,
	tokenDict *Dictionary,
	relationDict *Dictionary) *[]string {

	sentences := strings.Split(string(*data), ConlluDelimiter)
	sentenceStructures := make([]string,0)
	for _, sentence := range sentences {
		if len(strings.TrimSpace(sentence)) == 0 {continue}

		tokenIds := []string{strconv.Itoa(tokenDict.GetId("<ROOT>"))}
		headPos := []string{"0"}
		relationIds := []string{strconv.Itoa(relationDict.GetId("self"))}
		lines := strings.Split(sentence, "\n")
		for _, line := range lines {
			if len(line) == 0 || line[:1] == "#"  {continue}
			fields := strings.Split(line, "\t")
			if strings.Index(fields[0], "-") >= 0 ||
				strings.Index(fields[0], ".") >= 0 {
				continue
			}

			// Extract the desired tokens, relations, and head positions.
			// These are converted to Ids, and then stringified.
			tokenId := strconv.Itoa(tokenDict.GetId(fields[1]))
			tokenIds = append(tokenIds, tokenId)
			relationId := strconv.Itoa(relationDict.GetId(fields[7]))
			relationIds = append(relationIds, relationId)
			headPos = append(headPos, fields[6])
		}

		sentenceStructure := (
			strings.Join(tokenIds, ",") + ";" +
			strings.Join(headPos, ",") + ";" +
			strings.Join(relationIds, ",") + "\n")
		sentenceStructures = append(sentenceStructures, sentenceStructure)
	}
	return &sentenceStructures
}

