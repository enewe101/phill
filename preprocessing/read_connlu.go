package main

import (
	"io"
	"os"
	"sync"
	"strings"
	"fmt"
	"strconv"
	"io/fs"
	"errors"
)

const delim = "\n\n"
const sentenceFname = "sentences.index"
const tokenDictFname = "tokens.dict"
const relationDictFname = "relations.dict"

// Workers expect a callback that will process chunks of bytes.
type chunkProcessor func([]byte)

// Worker accepts a chunker which yields slices of bytes to be procesed, and
// applies a callback, processFunc, to each slice of bytes
func worker(
	chunker *Chunker,
	wg *sync.WaitGroup,
	processFunc chunkProcessor) {

	defer wg.Done()
	for {
		chunk, err := chunker.next()
		if err != nil {
			if err == io.EOF {
				break
			} else {
				panic(err.Error())
			}
		}
		processFunc(chunk)
		//result_channel <- &(processed)
	}
}


// Waits for all workers to be finished, and then closes the channel that 
// they share, so that the consumer of the shared channel knows no more
// results are coming
func watch_progress(
	wg *sync.WaitGroup,
	tokenChannel chan<- *[]string,
	relationChannel chan<- *[]string) {

	defer close(tokenChannel)
	defer close(relationChannel)
	wg.Wait()
}


// Dispatch workers to proces chunks of a file breaking them into tokens
// and putting the tokens on a resultsChannel.  Consume tokens from the 
// resultsChannel and add them to a dictionary.  Write the dictionary to disk.
func makeDictionary(inPaths []string, outDir string) {

	// Create the directory where output will be written, if not exist.
	err := os.Mkdir(outDir, 0777)
	if err != nil && !errors.Is(err, fs.ErrExist) {panic(err.Error())}

	// File for writing token dictionary
	tokenDictFile, err := os.Create(outDir + "/" + tokenDictFname)
	if err != nil {panic(err.Error())}
	defer tokenDictFile.Close()

	// File for writing relation dictionary.
	relationDictFile, err := os.Create(outDir + "/" + relationDictFname)
	if err != nil {panic(err.Error())}
	defer relationDictFile.Close()

	tokenChannel := make(chan *[]string, 100)
	relationChannel := make(chan *[]string, 100)

	// Start workers on each of the input files.
	processFunc := getDictChunkProcessor(tokenChannel, relationChannel)
	numWorkersPerFile := 2
	var producerGroup sync.WaitGroup
	for _, inPath := range inPaths {

		// Open the file.
		file, err := os.Open(inPath)
		if err != nil {panic(err.Error())}
		defer file.Close()

		// Set up for producers
		chunk_group := NewChunkerGroup(
			file, 4*1024, 1024, []byte(delim), numWorkersPerFile)

		// Start worker producers to read and process chunks of file
		for i:=0; i<numWorkersPerFile; i++ {
			producerGroup.Add(1)
			go worker(chunk_group[i], &producerGroup, processFunc)
		}
	}


	// Create and  initialize dictionaries for tokens and relations.  Add a
	// couple special tokens.  <ROOT> represents the root of sentences, and
	// <UNK> represents an unknown token, which can be used when an out-of-
	// vocabulary token is seen at test time.  Add "self" to the relation dict,
	// which will be used to stand in for <ROOT>'s outgoing relation, which you
	// can think of as a self-loop.  This makes it easier to have <ROOT> in the
	// representation of a sentence while modelling the sentence with three
	// parallel lists: token ids, head-pointers, and relation types.  We need
	// to assign something to <ROOT> in the head-pointers and relation types,
	// so we model it as a self loop.  This also helps to calculate
	// "rootedness", since <ROOT> will be a sink when following head-pointers
	// from any rooted token.
	tokenDict := NewDictionary()
	tokenDict.Add("<ROOT>")
	tokenDict.Add("<UNK>")
	tokenDict.Add("<PAD>")
	relationDict := NewDictionary()
	relationDict.Add("self")

	// Start consumers
	var consumerGroup sync.WaitGroup
	consumerGroup.Add(2)
	go writeDictionary(
		&tokenDict, tokenDictFile, tokenChannel, &consumerGroup)
	go writeDictionary(
		&relationDict, relationDictFile, relationChannel, &consumerGroup)

	// this process will watch the waitgroup and close the channels
	// once workers are all done.
	go watch_progress(&producerGroup, tokenChannel, relationChannel)

	// Wait for the consumers to finish (they will only finish once producers
	// finish).
	consumerGroup.Wait()
}

func writeDictionary(
	dict *Dictionary,
	file *os.File,
	tokenSlices <-chan *[]string,
	group *sync.WaitGroup) {

	defer group.Done()

	for tokenSlice := range tokenSlices {
		for _, token := range *tokenSlice {
			dict.Add(token)
		}
	}
	dict.Write(file)
}


// Callback for a worker.  Slices of bytes provided from the worker are
// split into tokens, and these are placed onto a results channel for further
// processing
func getDictChunkProcessor(
	tokenChannel chan<- *[]string,
	relationChannel chan<- *[]string) chunkProcessor {

	processChunk := func (chunk []byte) {

		sentences := strings.Split(string(chunk), delim)
		tokens := make([]string, 0)
		relations := make([]string, 0)
		for _, sentence := range sentences {
			if len(strings.TrimSpace(sentence)) == 0 {
				continue
			}
			lines := strings.Split(sentence, "\n")
			for _, line := range lines {
				if len(line) == 0 || line[:1] == "#" {
					continue
				}
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
		tokenChannel <- &tokens
		relationChannel <- &relations
	}
	return processChunk
}


// Processes Slices of bytes containing sentences in conllu format.  Extract
// the ids of the tokens, the positions of the heads, and the id of the 
// dependency relations
func getIndexChunkProcessor(
	resultChannel chan<- *[]string,
	tokenDict *Dictionary,
	relationDict *Dictionary) chunkProcessor {

	processChunk := func (chunk []byte) {
		sentences := strings.Split(string(chunk), delim)
		sentenceStructures := make([]string,0)
		for _, sentence := range sentences {
			if len(strings.TrimSpace(sentence)) == 0 {
				continue
			}
			tokenIds := []string{strconv.Itoa(tokenDict.GetId("<ROOT>"))}
			headPos := []string{"0"}
			relationIds := []string{strconv.Itoa(relationDict.GetId("self"))}
			lines := strings.Split(sentence, "\n")
			for _, line := range lines {
				if len(line) == 0 || line[:1] == "#"  {
					continue
				}
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
			if len(tokenIds) != len(headPos) || len(headPos) != len(relationIds){
				panic(
					"Extracted sentence structure should consist of three " +
					"parallel arrays of same length, but got:\n" + 
					fmt.Sprintf("%v\n%v\n%v\n", tokenIds, headPos, relationIds))
			}
			sentenceStructure := (
				strings.Join(tokenIds, ",") + ";" +
				strings.Join(headPos, ",") + ";" +
				strings.Join(relationIds, ",") + "\n")
			sentenceStructures = append(sentenceStructures, sentenceStructure)
		}
		resultChannel <- &sentenceStructures
	}
	return processChunk
}

// Dispatch workers to proces chunks of a file breaking them into tokens
// and putting the tokens on a results channel.  Consume tokens from the results
// channel and add them to a dictionary.  Write the dictionary to disk.
func indexSentences(inPaths []string, outDir string) {

	// Create file for sentence index.  outDir must exist.
	outfile, err := os.Create(outDir + "/" + sentenceFname)
	if err != nil {panic(err.Error())}
	defer outfile.Close()

	// Read dictionaries that will be used to encode lexical items as ids.
	tokenDictPath := outDir + "/" + tokenDictFname
	relationDictPath := outDir + "/" + relationDictFname
	tokenDict := ReadDictionary(tokenDictPath)
	relationDict := ReadDictionary(relationDictPath)

	// Set up channels and waitroup for producers.
	resultChannel := make(chan *[]string, 1000)
	num_workers := 1
	var producerGroup sync.WaitGroup

	// Curry channel and dictionaries into the chunk processing function
	processFunc := getIndexChunkProcessor(
		resultChannel, &tokenDict, &relationDict)

	// Start workers on each conllu file.
	for _, inPath := range inPaths {

		// Open file and create a chunk group to read it.
		file, err := os.Open(inPath)
		if err != nil {panic(err.Error())}
		defer file.Close()
		chunk_group := NewChunkerGroup(
			file, 10*1024, 1024, []byte(delim), num_workers)

		// Start worker producers to read and process chunks of file.
		for i:=0; i<num_workers; i++ {
			producerGroup.Add(1)
			go worker(chunk_group[i], &producerGroup, processFunc)
		}
	}

	// this process will watch the waitgroup and close the channel 
	// once workers are all done.
	go watchIndexProgress(&producerGroup, resultChannel)

	// This continues until watchIndexProgress closes the resultChannel.
	for sentenceStructures := range resultChannel {
		for _, sentenceStructure := range *sentenceStructures {
			outfile.WriteString(sentenceStructure)
		}
	}
}

// Waits for all workers to be finished, then closes the resultChannel
func watchIndexProgress(wg *sync.WaitGroup, resultChannel chan<- *[]string) {
	defer close(resultChannel)
	wg.Wait()
}

