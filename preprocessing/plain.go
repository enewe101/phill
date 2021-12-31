package main

import (
	"strings"
	"strconv"
)

const PlainDelimiter = "\n"


func plainDictParser(data *[]byte) (newTokens *[]string, newRelations *[]string){
	sentences := strings.Split(string(*data), PlainDelimiter)
	tokens := make([]string, 0)
	for _, sentence := range sentences {
		sentence = strings.TrimSpace(sentence)
		if len(sentence) == 0 {continue}
		sentenceTokens := strings.Split(sentence, " ")
		tokens = append(tokens, sentenceTokens...)
	}
	tokens = *releaseMemory(&tokens)
	relationsDummy := make([]string,0)
	return &tokens, &relationsDummy
}


// This seems to be necessary to release data dragged along with 
// the tokens. Perhaps those tokens reference the original file data.
func releaseMemory(tokens *[]string) (*[]string) {
	newStrings := make([]string, len(*tokens))
	for i, token := range *tokens {
		var copier strings.Builder
		copier.Grow(len(token))
		copier.WriteString(token)
		newStrings[i] = copier.String()
	}
	return &newStrings
}


func plainSentenceParser(
	data *[]byte,
	tokenDict *Dictionary,
	relationDict *Dictionary) (*[]string) {

	sentences := strings.Split(string(*data), PlainDelimiter)
	sentenceStructures := make([]string,len(sentences))
	for i, sentence := range sentences {
		sentence = strings.TrimSpace(sentence)
		if len(sentence) == 0 {continue }
		tokens := strings.Split(sentence, " ")
		tokenIds := make([]string, len(tokens)+1)
		tokenIds[0] = strconv.Itoa(tokenDict.GetId("<ROOT>"))
		for j, token := range tokens {
			tokenId := strconv.Itoa(tokenDict.GetId(token))
			tokenIds[j+1] = tokenId
		}
		sentenceStructure := strings.Join(tokenIds, ",") + "\n"
		sentenceStructures[i] = sentenceStructure
	}
	return &sentenceStructures
}

